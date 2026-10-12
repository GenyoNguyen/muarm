"""Adaptive multi-start collision-boundary data for CDFlow.

The retained target is a *set* of near-nearest boundary configurations.  The
optimizer operates in physical joint radians; obstacle gaps are MuJoCo metres.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
from scipy.optimize import minimize

from .collision import PandaCollisionOracle
from .scene import SphereScene


DATA_VERSION = 1


@dataclass(frozen=True)
class BoundaryDataConfig:
    """Finite budgets and explicit numerical interpretations of Algorithm 1."""

    queries_per_scene: int = 16
    global_starts: int = 16
    local_starts: int = 4             # per retained global solution
    local_radius: float = 0.16        # joint-space epsilon ball, radians
    near_distance_tolerance: float = 0.06  # absolute radians above global d_min
    boundary_tolerance: float = 0.002  # metres
    dedup_tolerance: float = 0.012    # joint radians
    min_query_gap: float = 0.012      # metres
    max_query_gap: float = 0.22       # metres
    max_query_attempts: int = 2500   # per scene
    max_seed_attempts: int = 80       # random joint draws per global start
    optimizer_maxiter: int = 100
    optimizer_ftol: float = 1e-8
    bisection_iters: int = 28

    def __post_init__(self) -> None:
        if min(self.queries_per_scene, self.global_starts, self.local_starts,
               self.max_query_attempts, self.max_seed_attempts,
               self.optimizer_maxiter, self.bisection_iters) < 1:
            raise ValueError("all sampling and optimizer counts must be positive")
        if not (0 < self.min_query_gap < self.max_query_gap):
            raise ValueError("query gap interval must be positive and increasing")
        if min(self.local_radius, self.near_distance_tolerance,
               self.boundary_tolerance, self.dedup_tolerance) <= 0:
            raise ValueError("radii and numerical tolerances must be positive")


@dataclass(frozen=True)
class _Candidate:
    q: np.ndarray
    gap: float
    status: int
    source: int  # 0 global, 1 local


def _bisect_boundary(
    oracle: PandaCollisionOracle,
    free_q: np.ndarray,
    colliding_q: np.ndarray,
    iterations: int,
) -> np.ndarray:
    """Bracket the first zero of the signed obstacle gap on a joint segment."""
    free = free_q.copy()
    hit = colliding_q.copy()
    if oracle.gap(free) <= 0 or oracle.gap(hit) > 0:
        raise ValueError("bisection needs free and colliding endpoints")
    for _ in range(iterations):
        middle = 0.5 * (free + hit)
        if oracle.gap(middle) > 0:
            free = middle
        else:
            hit = middle
    return 0.5 * (free + hit)


def _random_colliding_seed(
    oracle: PandaCollisionOracle,
    query: np.ndarray,
    rng: np.random.Generator,
    config: BoundaryDataConfig,
) -> np.ndarray | None:
    lower, upper = oracle.joint_limits.T
    for _ in range(config.max_seed_attempts):
        seed = rng.uniform(lower, upper)
        if oracle.gap(seed) <= 0:
            return _bisect_boundary(oracle, query, seed, config.bisection_iters)
        # A free endpoint may still cross an obstacle on the way from query.
        previous = query
        for alpha in np.linspace(0.1, 1.0, 10):
            probe = query + alpha * (seed - query)
            if oracle.gap(probe) <= 0:
                return _bisect_boundary(oracle, previous, probe, config.bisection_iters)
            previous = probe
    return None


def _optimize_to_boundary(
    oracle: PandaCollisionOracle,
    query: np.ndarray,
    seed: np.ndarray,
    config: BoundaryDataConfig,
    *,
    source: int,
) -> _Candidate | None:
    """Minimize joint distance to query subject to exact sphere contact.

    SLSQP uses finite differences for the nonsmooth minimum geometry gap.
    The final solution is independently checked with MuJoCo, regardless of
    SciPy's success flag.  A collision-side result is repaired by bisection.
    """
    lower, upper = oracle.joint_limits.T
    seed = np.clip(seed, lower, upper)

    def objective(x: np.ndarray) -> float:
        delta = x - query
        return 0.5 * float(delta @ delta)

    def objective_jac(x: np.ndarray) -> np.ndarray:
        return x - query

    def contact(x: np.ndarray) -> float:
        # SLSQP may ask for a finite-difference point a few ulps outside a
        # bound even with bounds supplied.  Keep only those *trial* points in
        # range; the returned solution is independently validated below.
        return oracle.gap(np.clip(x, lower, upper))

    result = minimize(
        objective, seed, jac=objective_jac, method="SLSQP",
        bounds=list(map(tuple, oracle.joint_limits)),
        constraints=({"type": "eq", "fun": contact},),
        options={"maxiter": config.optimizer_maxiter,
                 "ftol": config.optimizer_ftol, "disp": False},
    )
    candidate = np.clip(np.asarray(result.x, dtype=np.float64), lower, upper)
    if not oracle.in_limits(candidate):
        return None
    gap = oracle.gap(candidate)
    if gap < -config.boundary_tolerance and oracle.gap(query) > 0:
        candidate = _bisect_boundary(oracle, query, candidate, config.bisection_iters)
        gap = oracle.gap(candidate)
    if not oracle.boundary_valid(candidate, tolerance=config.boundary_tolerance):
        return None
    return _Candidate(candidate.copy(), float(gap), int(result.status), source)


def _unique(
    candidates: Sequence[_Candidate], tolerance: float, query: np.ndarray
) -> list[_Candidate]:
    """Remove numerical repeats without averaging distinct contact modes."""
    ordered = sorted(candidates, key=lambda c: np.linalg.norm(c.q - query))
    kept: list[_Candidate] = []
    for candidate in ordered:
        if all(np.linalg.norm(candidate.q - other.q) > tolerance for other in kept):
            kept.append(candidate)
    return kept


def _split_by_scene(count: int, rng: np.random.Generator) -> np.ndarray:
    """Assign whole scenes to splits 0=train, 1=validation, 2=test."""
    result = np.zeros(count, dtype=np.int8)
    permutation = rng.permutation(count)
    if count >= 3:
        n_test = max(1, round(0.1 * count))
        n_validation = max(1, round(0.1 * count))
        result[permutation[-n_test:]] = 2
        result[permutation[-(n_test + n_validation):-n_test]] = 1
    elif count == 2:
        result[permutation[-1]] = 1
    return result


def generate_dataset(
    model: mujoco.MjModel,
    scenes: Sequence[SphereScene],
    config: BoundaryDataConfig | None = None,
    *,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Generate a variable-target NPZ-ready dataset with scene-level splits.

    Query ``i`` has targets ``boundary_q[offsets[i]:offsets[i+1]]``.  Failed
    queries are counted in metadata and omitted.  A scene with zero successful
    queries raises to avoid a silent invalid split.
    """
    cfg = config or BoundaryDataConfig()
    if not scenes:
        raise ValueError("at least one scene is required")
    rng = np.random.default_rng(seed)
    oracle = PandaCollisionOracle(model)
    lower, upper = oracle.joint_limits.T
    max_spheres = max(len(scene.spheres) for scene in scenes)
    scene_splits = _split_by_scene(len(scenes), rng)
    scene_seeds = rng.integers(0, np.iinfo(np.int64).max, len(scenes), dtype=np.int64)

    rows: dict[str, list] = {key: [] for key in (
        "q", "scene", "scene_mask", "y_d", "y_g", "scene_id", "split",
        "query_gap", "nearest_distance", "global_attempts", "global_feasible",
        "local_attempts", "local_feasible", "mode_count", "distance_spread",
    )}
    boundary_q: list[np.ndarray] = []
    boundary_gap: list[float] = []
    boundary_status: list[int] = []
    boundary_source: list[int] = []
    offsets = [0]
    audits: list[dict] = []
    for scene_index, scene in enumerate(scenes):
        scene_rng = np.random.default_rng(int(scene_seeds[scene_index]))
        oracle.set_scene(scene)
        features, mask = scene.padded(max_spheres)
        completed = 0
        query_draws = 0
        failed_query_optimizations = 0
        global_failed = local_failed = 0
        global_attempts_total = local_attempts_total = local_not_near = 0
        while completed < cfg.queries_per_scene and query_draws < cfg.max_query_attempts:
            query_draws += 1
            # The actual model key pose is useful for a small diagnostic run.
            query = (model.key_qpos[0, oracle.qpos_indices].copy()
                     if query_draws == 1 and model.nkey else scene_rng.uniform(lower, upper))
            if not oracle.in_limits(query):
                continue
            query_gap = oracle.gap(query)
            if not (cfg.min_query_gap <= query_gap <= cfg.max_query_gap):
                continue
            if oracle.constraint_gap(query) <= 0:
                continue

            global_candidates: list[_Candidate] = []
            for _ in range(cfg.global_starts):
                global_attempts_total += 1
                start = _random_colliding_seed(oracle, query, scene_rng, cfg)
                candidate = (None if start is None else
                             _optimize_to_boundary(oracle, query, start, cfg, source=0))
                if candidate is None:
                    global_failed += 1
                else:
                    global_candidates.append(candidate)
            if not global_candidates:
                failed_query_optimizations += 1
                continue
            global_candidates = _unique(global_candidates, cfg.dedup_tolerance, query)
            d_min = min(np.linalg.norm(query - item.q) for item in global_candidates)
            retained = [item for item in global_candidates
                        if np.linalg.norm(query - item.q) <=
                        d_min + cfg.near_distance_tolerance]

            local_candidates: list[_Candidate] = []
            local_attempts = 0
            for base in retained:
                for _ in range(cfg.local_starts):
                    local_attempts += 1
                    local_attempts_total += 1
                    direction = scene_rng.normal(size=7)
                    direction /= np.linalg.norm(direction)
                    radius = cfg.local_radius * scene_rng.random() ** (1.0 / 7.0)
                    local_seed = np.clip(base.q + radius * direction, lower, upper)
                    candidate = _optimize_to_boundary(
                        oracle, query, local_seed, cfg, source=1
                    )
                    if candidate is None:
                        local_failed += 1
                    elif np.linalg.norm(query - candidate.q) <= d_min + cfg.near_distance_tolerance:
                        local_candidates.append(candidate)
                    else:
                        local_not_near += 1

            selected = _unique([*retained, *local_candidates], cfg.dedup_tolerance, query)
            if not selected:
                failed_query_optimizations += 1
                continue
            targets = np.stack([item.q for item in selected])
            distances = np.linalg.norm(targets - query, axis=1)
            directions = (query[None, :] - targets) / distances[:, None]
            if not (np.all(np.isfinite(directions)) and np.all(distances > 1e-8)):
                failed_query_optimizations += 1
                continue
            rows["q"].append(query.copy())
            rows["scene"].append(features)
            rows["scene_mask"].append(mask)
            rows["y_d"].append(float(np.mean(distances)))
            rows["y_g"].append(np.mean(directions, axis=0))
            rows["scene_id"].append(scene_index)
            rows["split"].append(int(scene_splits[scene_index]))
            rows["query_gap"].append(query_gap)
            rows["nearest_distance"].append(d_min)
            rows["global_attempts"].append(cfg.global_starts)
            rows["global_feasible"].append(len(global_candidates))
            rows["local_attempts"].append(local_attempts)
            rows["local_feasible"].append(len(local_candidates))
            rows["mode_count"].append(len(selected))
            rows["distance_spread"].append(float(np.ptp(distances)))
            for item in selected:
                boundary_q.append(item.q)
                boundary_gap.append(item.gap)
                boundary_status.append(item.status)
                boundary_source.append(item.source)
            offsets.append(len(boundary_q))
            completed += 1
        audits.append({
            "scene_id": scene_index,
            "random_seed": int(scene_seeds[scene_index]),
            "split": int(scene_splits[scene_index]),
            "query_draws": query_draws,
            "completed_queries": completed,
            "failed_query_optimizations": failed_query_optimizations,
            "global_optimizer_failures": global_failed,
            "global_optimizer_attempts": global_attempts_total,
            "local_optimizer_failures": local_failed,
            "local_optimizer_attempts": local_attempts_total,
            "local_candidates_not_near": local_not_near,
        })
        if completed == 0:
            raise RuntimeError(
                f"scene {scene_index} produced no boundary sets after {query_draws} query draws; "
                f"audit={audits[-1]}"
            )

    dataset = {
        "q": np.asarray(rows["q"], dtype=np.float32),
        "scene": np.asarray(rows["scene"], dtype=np.float32),
        "scene_mask": np.asarray(rows["scene_mask"], dtype=bool),
        "boundary_q": np.asarray(boundary_q, dtype=np.float32),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "y_d": np.asarray(rows["y_d"], dtype=np.float32),
        "y_g": np.asarray(rows["y_g"], dtype=np.float32),
        "boundary_gap": np.asarray(boundary_gap, dtype=np.float32),
        "boundary_status": np.asarray(boundary_status, dtype=np.int16),
        "boundary_source": np.asarray(boundary_source, dtype=np.int8),
        "joint_limits": oracle.joint_limits.astype(np.float32),
    }
    for key, dtype in (
        ("scene_id", np.int32), ("split", np.int8), ("query_gap", np.float32),
        ("nearest_distance", np.float32), ("global_attempts", np.int16),
        ("global_feasible", np.int16), ("local_attempts", np.int16),
        ("local_feasible", np.int16), ("mode_count", np.int16),
        ("distance_spread", np.float32),
    ):
        dataset[key] = np.asarray(rows[key], dtype=dtype)
    metadata = {
        "version": DATA_VERSION,
        "robot": "Franka Panda, 7 arm joints, radians",
        "scene_schema": "ordered reach_obstacle_i spheres: xyz metres, radius metres",
        "mask_schema": "True means the corresponding sphere is present",
        "gap_units": "metres, signed MuJoCo robot–obstacle surface distance",
        "direction_schema": "mean unit vector from boundary sample toward free query",
        "split_schema": {"0": "train", "1": "validation", "2": "test"},
        "seed": seed,
        "config": asdict(cfg),
        "scene_audit": audits,
        "boundary_residual_max_abs_m": float(np.max(np.abs(boundary_gap))),
        "boundary_residual_mean_abs_m": float(np.mean(np.abs(boundary_gap))),
        "nearest_distance_spread_mean_rad": float(np.mean(dataset["distance_spread"])),
        "mode_count_mean": float(np.mean(dataset["mode_count"])),
        "global_failure_rate": float(sum(a["global_optimizer_failures"] for a in audits) /
                                     max(1, sum(a["global_optimizer_attempts"] for a in audits))),
        "local_failure_rate": float(sum(a["local_optimizer_failures"] for a in audits) /
                                    max(1, sum(a["local_optimizer_attempts"] for a in audits))),
    }
    dataset["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return dataset


def save_dataset(path: str | Path, dataset: dict[str, np.ndarray]) -> Path:
    """Write compressed NumPy arrays without pickle or arbitrary object data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    return path


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load and check the offset-indexed format used by CDFlow training."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    required = {"q", "scene", "scene_mask", "boundary_q", "offsets", "y_d",
                "y_g", "boundary_gap", "scene_id", "split", "joint_limits", "metadata_json"}
    if required - result.keys():
        raise ValueError(f"missing dataset arrays: {sorted(required - result.keys())}")
    count = len(result["q"])
    target_count = len(result["boundary_q"])
    scene = result["scene"]
    if (result["q"].shape != (count, 7)
        or scene.ndim != 3 or scene.shape[0] != count or scene.shape[2] != 4
        or result["scene_mask"].shape != scene.shape[:2]
        or not np.all(np.any(result["scene_mask"], axis=1))
        or result["boundary_q"].shape != (target_count, 7)
        or result["boundary_gap"].shape != (target_count,)
        or result["y_d"].shape != (count,)
        or result["y_g"].shape != (count, 7)
        or result["scene_id"].shape != (count,)
        or result["split"].shape != (count,)
        or result["joint_limits"].shape != (7, 2)
        or result["offsets"].shape != (count + 1,)
        or result["offsets"][0] != 0
        or result["offsets"][-1] != target_count
        or np.any(np.diff(result["offsets"]) < 1)
        or not np.all(np.isfinite(result["q"]))
        or not np.all(np.isfinite(result["boundary_q"]))
        or not np.all(np.isfinite(result["y_d"]))
        or not np.all(np.isfinite(result["y_g"]))
        or not np.all(np.isfinite(result["boundary_gap"]))
        or not np.all(np.isin(result["split"], [0, 1, 2]))):
        raise ValueError("invalid CDFlow dataset array shapes or values")
    if json.loads(str(result["metadata_json"]))["version"] != DATA_VERSION:
        raise ValueError("unsupported CDFlow dataset version")
    return result
