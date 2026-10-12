#!/usr/bin/env python3
"""Evaluate CDFlow whole-arm button contact at one or many sphere positions.

This is a geometric joint-space contact task, not the end-effector XYZ reach
task.  It uses one active sphere from ``reach_obstacles`` and stops when any
Panda arm/hand/finger collision geom contacts that sphere in MuJoCo.  The
negative of each field's outward direction is the commanded direction.

Example::

    .venv/bin/python scripts/eval_cdflow_contact.py \
        --checkpoint learning/runs/panda_cdflow.pt --episodes 10 \
        --sphere-center 0.55 0.05 0.31 --sphere-radius 0.01 \
        --mc-samples 64 --output learning/runs/contact_eval.json

Pass ``--scene-data`` to run a separate trial at each distinct recorded button
position, selecting a scene split with ``--split``. Pass ``--training-data`` to
audit whether the checkpoint's training split included the evaluated geometry.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from learning.cdflow.collision import PandaCollisionOracle
from learning.cdflow.field import CDFlowField
from learning.cdflow.scene import SphereScene
from learning.algos.cdflow_projection import (
    CDFlowProjectionConfig, CDFlowProjectionPolicy, arm_sphere_contact,
)
from learning.robots.panda import _build_model


@dataclass(frozen=True)
class ContactConfig:
    episodes: int = 10
    max_steps: int = 100
    time_limit_s: float = 3.0
    step_size: float = 0.05             # joint-space Euclidean radians
    max_joint_step: float = 0.04        # radians per joint per projection
    edge_resolution: float = 0.005      # maximum per-joint interpolation gap
    min_direction_norm: float = 1e-5
    safety_margin: float = 0.0          # non-obstacle gap, metres
    start_noise: float = 0.25           # Gaussian joint radians around home
    start_gap_min: float = 0.015        # sphere surface metres
    start_gap_max: float = 0.30
    max_start_attempts: int = 5000
    mc_samples: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.episodes, self.max_steps, self.max_start_attempts,
               self.mc_samples) < 1:
            raise ValueError("episode, step, start-attempt, and sample counts must be positive")
        if min(self.time_limit_s, self.step_size, self.max_joint_step, self.edge_resolution,
               self.min_direction_norm, self.start_gap_min) <= 0:
            raise ValueError("projection and start parameters must be positive")
        if self.start_noise < 0 or self.safety_margin < 0:
            raise ValueError("start noise and safety margin must be nonnegative")
        if self.start_gap_max <= self.start_gap_min:
            raise ValueError("start gap bounds must be increasing")


def _single_sphere_scenes(
    sphere_index: int,
    sphere_center: np.ndarray | None = None,
    sphere_radius: float | None = None,
) -> tuple[SphereScene, SphereScene]:
    """Return full-size oracle mask and compact one-sphere field description."""
    base = SphereScene.from_model(_build_model("reach_obstacles"))
    if not 0 <= sphere_index < len(base.spheres):
        raise ValueError(f"sphere index must be in [0, {len(base.spheres) - 1}]")
    spheres = base.spheres.copy()
    if sphere_center is not None:
        center = np.asarray(sphere_center, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("sphere_center must contain three finite XYZ metres")
        spheres[sphere_index, :3] = center
    if sphere_radius is not None:
        if not np.isfinite(sphere_radius) or sphere_radius <= 0:
            raise ValueError("sphere_radius must be finite and positive")
        spheres[sphere_index, 3] = sphere_radius
    mask = np.zeros(len(base.spheres), dtype=bool)
    mask[sphere_index] = True
    oracle_scene = SphereScene(spheres, mask, base.names)
    field_scene = SphereScene(
        spheres[sphere_index:sphere_index + 1],
        names=(base.names[sphere_index],),
    )
    return oracle_scene, field_scene


def _check_checkpoint_robot(field: CDFlowField, oracle: PandaCollisionOracle) -> None:
    expected = oracle.joint_limits
    received = np.asarray(field.model.config.joint_limits, dtype=np.float64)
    if not np.allclose(received, expected, rtol=0.0, atol=2e-4):
        raise ValueError(
            "checkpoint joint limits differ from the reach_obstacles Panda model; "
            "train a model for this robot before contact evaluation"
        )
    robot = field.metadata.get("robot")
    if robot is not None and "Franka Panda" not in str(robot):
        raise ValueError(f"checkpoint dataset describes {robot!r}, not Franka Panda")
    scene_schema = field.metadata.get("scene_schema")
    expected_scene_schema = "ordered reach_obstacle_i spheres: xyz metres, radius metres"
    if scene_schema is not None and scene_schema != expected_scene_schema:
        raise ValueError(
            f"checkpoint scene schema {scene_schema!r} differs from the "
            f"expected {expected_scene_schema!r}"
        )


def _training_coverage(
    field: CDFlowField, training_data: Path | None, sphere: np.ndarray,
) -> dict[str, Any]:
    """Use the actual training split; checkpoint metadata lacks scene masks."""
    training_spheres = _load_training_spheres(field, training_data)
    return _coverage_from_training_spheres(training_spheres, sphere)


def _load_training_spheres(
    field: CDFlowField, training_data: Path | None,
) -> tuple[np.ndarray, int] | None:
    """Read training geometry once for a potentially large scene sweep."""
    if training_data is None:
        return None
    with np.load(training_data, allow_pickle=False) as archive:
        for key in ("scene", "scene_mask", "split", "metadata_json"):
            if key not in archive:
                raise ValueError(f"training dataset lacks {key!r}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata != field.metadata:
            raise ValueError(
                "--training-data metadata does not match checkpoint dataset metadata"
            )
        scenes = np.asarray(archive["scene"], dtype=np.float64)
        masks = np.asarray(archive["scene_mask"], dtype=bool)
        split = np.asarray(archive["split"])
    if scenes.ndim != 3 or scenes.shape[-1] != 4 or masks.shape != scenes.shape[:2]:
        raise ValueError("training dataset has invalid scene/mask dimensions")
    if split.shape != (len(scenes),):
        raise ValueError("training dataset split shape is invalid")
    one_sphere = (split == 0) & (masks.sum(axis=1) == 1)
    count = int(one_sphere.sum())
    represented = scenes[one_sphere][masks[one_sphere]]
    return np.unique(represented, axis=0), count


def _coverage_from_training_spheres(
    training_spheres: tuple[np.ndarray, int] | None, sphere: np.ndarray,
) -> dict[str, Any]:
    if training_spheres is None:
        return {
            "status": "unverified",
            "warning": (
                "Checkpoint metadata does not record active-sphere counts. "
                "Supply --training-data to check whether this single-sphere "
                "scene was represented during training."
            ),
        }
    represented, count = training_spheres
    if count == 0:
        return {
            "status": "unseen_active_count",
            "single_sphere_training_queries": 0,
            "warning": "The checkpoint's training split contains no single-active-sphere queries; contact results are out of distribution.",
        }
    geometry_distance = np.linalg.norm(represented - sphere[None, :], axis=1)
    exact = bool(np.any(np.all(np.isclose(represented, sphere, atol=1e-5, rtol=0.0), axis=1)))
    return {
        "status": "matched_scene" if exact else "unseen_geometry",
        "single_sphere_training_queries": count,
        "same_sphere_in_training": exact,
        "nearest_training_sphere_geometry_l2": float(geometry_distance.min()),
        "warning": None if exact else (
            "The training split contains single-sphere scenes but not this exact "
            "sphere geometry; results measure scene generalization."
        ),
    }


def _arm_sphere_contact(oracle: PandaCollisionOracle) -> bool:
    """Require a MuJoCo contact pair, including arm links, hand, and fingers."""
    return arm_sphere_contact(oracle)


def _sample_starts(
    oracle: PandaCollisionOracle, config: ContactConfig,
    *,
    anchor_q: np.ndarray | None = None,
    candidate_q: np.ndarray | None = None,
) -> list[tuple[np.ndarray, float]]:
    rng = np.random.default_rng(config.seed)
    lower, upper = oracle.joint_limits.T
    home = (
        oracle.model.key_qpos[0, oracle.qpos_indices].copy()
        if oracle.model.nkey else np.mean(oracle.joint_limits, axis=1)
    )
    if anchor_q is not None:
        home = np.asarray(anchor_q, dtype=np.float64)
        if not oracle.in_limits(home):
            raise ValueError("button contact witness is outside Panda joint limits")
    starts: list[tuple[np.ndarray, float]] = []

    def add_if_free(q: np.ndarray) -> bool:
        if not oracle.in_limits(q):
            return False
        gap = oracle.gap(q)
        if (config.start_gap_min <= gap <= config.start_gap_max
                and oracle.constraint_gap(q) > config.safety_margin
                and not _arm_sphere_contact(oracle)):
            starts.append((q.copy(), gap))
            return True
        return False

    if candidate_q is not None:
        candidates = np.asarray(candidate_q, dtype=np.float64)
        if candidates.ndim != 2 or candidates.shape[1] != 7:
            raise ValueError("recorded button queries must have shape [N,7]")
        for index in rng.permutation(len(candidates)):
            add_if_free(candidates[index])
            if len(starts) == config.episodes:
                return starts
    for attempt in range(config.max_start_attempts):
        q = (home.copy() if attempt == 0 and anchor_q is None else
             np.clip(home + rng.normal(0, config.start_noise, size=7), lower, upper))
        add_if_free(q)
        if len(starts) == config.episodes:
            return starts
    raise RuntimeError(
        f"sampled {len(starts)}/{config.episodes} safe, near-sphere starts "
        f"after {config.max_start_attempts} attempts; adjust start-gap/noise settings"
    )


def _episode(
    field: CDFlowField,
    oracle: PandaCollisionOracle,
    scene: SphereScene,
    start: np.ndarray,
    config: ContactConfig,
    *,
    mode: str,
    episode_seed: int,
) -> dict[str, Any]:
    """Run the reusable projection controller and keep the JSON row schema."""
    # Keep paired Monte Carlo starts reproducible without changing another
    # caller's CPU or CUDA random stream.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(episode_seed)
        policy = CDFlowProjectionPolicy(
            field, oracle, scene,
            CDFlowProjectionConfig(
                max_steps=config.max_steps,
                time_limit_s=config.time_limit_s,
                step_size=config.step_size,
                max_joint_step=config.max_joint_step,
                edge_resolution=config.edge_resolution,
                min_direction_norm=config.min_direction_norm,
                safety_margin=config.safety_margin,
                mode=mode,
                mc_samples=config.mc_samples,
            ),
            contact_checker=_arm_sphere_contact,
        )
        policy.start_episode(start)
        rollout = policy.rollout()
    return {
        "mode": mode,
        "start_q": start.tolist(),
        "initial_gap_m": rollout.initial_gap_m,
        "success": rollout.success,
        "steps": rollout.attempted_steps,
        "reason": rollout.status,
        "final_q": rollout.trajectory[-1].tolist(),
        "final_gap_m": rollout.final_gap_m,
        "path_length_rad": rollout.path_length_rad,
        "field_queries": rollout.field_queries,
        "mean_field_query_ms": rollout.mean_field_query_ms,
        "total_field_query_ms": rollout.total_field_query_ms,
    }


def _summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in episodes if row["success"]]
    latencies = [row["total_field_query_ms"] for row in episodes]
    query_count = sum(row["field_queries"] for row in episodes)
    return {
        "episodes": len(episodes),
        "successes": len(successful),
        "success_rate": len(successful) / len(episodes) if episodes else 0.0,
        "mean_steps_to_contact": (float(np.mean([row["steps"] for row in successful]))
                                  if successful else None),
        "mean_path_length_to_contact_rad": (
            float(np.mean([row["path_length_rad"] for row in successful]))
            if successful else None
        ),
        "mean_field_query_ms": (float(sum(latencies) / query_count)
                                if query_count else None),
        "failure_reasons": {
            reason: sum(row["reason"] == reason for row in episodes)
            for reason in sorted({row["reason"] for row in episodes if not row["success"]})
        },
    }


def evaluate(
    checkpoint: Path,
    *,
    sphere_index: int = 1,
    sphere_center: np.ndarray | None = None,
    sphere_radius: float | None = None,
    mode: str = "mc",
    config: ContactConfig | None = None,
    device: str = "cpu",
    training_data: Path | None = None,
) -> dict[str, Any]:
    """Run paired, geometric projection episodes; return a JSON-ready report."""
    config = config or ContactConfig()
    if mode not in {"both", "regression", "mc"}:
        raise ValueError("mode must be both, regression, or mc")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    oracle_scene, field_scene = _single_sphere_scenes(
        sphere_index, sphere_center=sphere_center, sphere_radius=sphere_radius,
    )
    oracle = PandaCollisionOracle(_build_model("reach_obstacles"), oracle_scene)
    field = CDFlowField.load(checkpoint, device=device)
    _check_checkpoint_robot(field, oracle)
    sphere = field_scene.spheres[0]
    coverage = _training_coverage(field, training_data, sphere)
    starts = _sample_starts(oracle, config)
    modes = ("regression", "mc") if mode == "both" else (mode,)
    episodes_by_mode: dict[str, list[dict[str, Any]]] = {}
    for selected_mode in modes:
        # Exclude one-time model/autograd/ODE setup from the per-query latency.
        field.distance_and_gradient(
            field_scene, starts[0][0], mode=selected_mode,
            samples=config.mc_samples,
        )
        if field.device.type == "cuda":
            torch.cuda.synchronize(field.device)
        episodes_by_mode[selected_mode] = [
            _episode(
                field, oracle, field_scene, q, config,
                mode=selected_mode, episode_seed=config.seed + index,
            ) for index, (q, _) in enumerate(starts)
        ]
    return {
        "task": "Panda any-arm-geom contact with one selected sphere",
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "device": str(field.device),
        "scene": {
            "source_model": "reach_obstacles",
            "sphere_index": sphere_index,
            "sphere_name": field_scene.names[0],
            "sphere_xyz_radius_m": sphere.tolist(),
            "active_spheres": 1,
        },
        "training_coverage": coverage,
        "configuration": asdict(config),
        "contact_definition": "MuJoCo arm/hand/finger-to-selected-sphere contact pair with dist <= 0 m",
        "start_q": [q.tolist() for q, _ in starts],
        "start_gap_m": [float(gap) for _, gap in starts],
        "summary": {selected_mode: _summarize(rows)
                    for selected_mode, rows in episodes_by_mode.items()},
        "episodes": episodes_by_mode,
    }


@dataclass(frozen=True)
class _ButtonSceneRecord:
    scene_id: int
    split: str
    sphere_index: int
    sphere: np.ndarray
    queries: np.ndarray
    witness_q: np.ndarray | None


def _load_button_scenes(
    scene_data: Path, split: str,
) -> tuple[list[_ButtonSceneRecord], dict[str, Any]]:
    """Read each distinct single-button scene once, with its nearby free queries."""
    split_codes = {"train": 0, "validation": 1, "test": 2}
    if split not in {*split_codes, "all"}:
        raise ValueError("split must be train, validation, test, or all")
    with np.load(scene_data, allow_pickle=False) as archive:
        required = {"q", "scene", "scene_mask", "scene_id", "split", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"button scene dataset lacks {sorted(missing)}")
        q = np.asarray(archive["q"], dtype=np.float64)
        scene = np.asarray(archive["scene"], dtype=np.float64)
        mask = np.asarray(archive["scene_mask"], dtype=bool)
        ids = np.asarray(archive["scene_id"], dtype=np.int64)
        splits = np.asarray(archive["split"], dtype=np.int8)
        metadata = json.loads(str(archive["metadata_json"].item()))
    count = len(q)
    if (count == 0 or q.shape != (count, 7) or scene.ndim != 3
            or scene.shape[0] != count or scene.shape[-1] != 4
            or mask.shape != scene.shape[:2] or ids.shape != (count,)
            or splits.shape != (count,) or not np.all(np.isin(splits, (0, 1, 2)))
            or not np.all(np.isfinite(q)) or not np.all(np.isfinite(scene))):
        raise ValueError("button scene dataset has invalid array shapes or values")
    witnesses = {
        int(row["scene_id"]): np.asarray(row["witness_q_rad"], dtype=np.float64)
        for row in metadata.get("button_scene_audit", [])
        if "witness_q_rad" in row
    }
    records: list[_ButtonSceneRecord] = []
    for scene_id in np.unique(ids):
        rows = np.flatnonzero(ids == scene_id)
        first = int(rows[0])
        if (np.any(splits[rows] != splits[first])
                or not np.all(mask[rows] == mask[first])
                or not np.allclose(scene[rows], scene[first], rtol=0, atol=1e-6)):
            raise ValueError(f"button scene {scene_id} has inconsistent query rows")
        if split != "all" and splits[first] != split_codes[split]:
            continue
        active = np.flatnonzero(mask[first])
        if len(active) != 1:
            raise ValueError(f"button scene {scene_id} must have one active sphere")
        sphere = scene[first, active[0]].copy()
        if sphere[3] <= 0:
            raise ValueError(f"button scene {scene_id} has invalid sphere radius")
        witness = witnesses.get(int(scene_id))
        if witness is not None and (witness.shape != (7,) or not np.all(np.isfinite(witness))):
            raise ValueError(f"button scene {scene_id} has an invalid contact witness")
        records.append(_ButtonSceneRecord(
            int(scene_id),
            next(name for name, code in split_codes.items() if code == splits[first]),
            int(active[0]), sphere, q[rows].copy(), witness,
        ))
    if not records:
        raise ValueError(f"button scene dataset has no scenes in split {split!r}")
    return records, metadata


def evaluate_dataset(
    checkpoint: Path,
    scene_data: Path,
    *,
    split: str = "test",
    mode: str = "mc",
    config: ContactConfig | None = None,
    device: str = "cpu",
    training_data: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one or more trials at every distinct recorded button position."""
    config = config or ContactConfig(episodes=1)
    if mode not in {"both", "regression", "mc"}:
        raise ValueError("mode must be both, regression, or mc")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    records, metadata = _load_button_scenes(scene_data, split)
    field = CDFlowField.load(checkpoint, device=device)
    first = records[0]
    oracle_scene, _ = _single_sphere_scenes(
        first.sphere_index, first.sphere[:3], float(first.sphere[3]),
    )
    oracle = PandaCollisionOracle(_build_model("reach_obstacles"), oracle_scene)
    _check_checkpoint_robot(field, oracle)
    coverage_data = (training_data if training_data is not None else
                     scene_data if metadata == field.metadata else None)
    training_spheres = _load_training_spheres(field, coverage_data)
    modes = ("regression", "mc") if mode == "both" else (mode,)
    episodes_by_mode: dict[str, list[dict[str, Any]]] = {name: [] for name in modes}
    scene_results: list[dict[str, Any]] = []
    warmed: set[str] = set()
    for record in records:
        oracle_scene, field_scene = _single_sphere_scenes(
            record.sphere_index, record.sphere[:3], float(record.sphere[3]),
        )
        oracle.set_scene(oracle_scene)
        scene_config = replace(config, seed=config.seed + record.scene_id * 1009)
        starts = _sample_starts(
            oracle, scene_config, anchor_q=record.witness_q,
            candidate_q=record.queries,
        )
        scene_episodes: dict[str, list[dict[str, Any]]] = {}
        for selected_mode in modes:
            if selected_mode not in warmed:
                field.distance_and_gradient(
                    field_scene, starts[0][0], mode=selected_mode,
                    samples=config.mc_samples,
                )
                if field.device.type == "cuda":
                    torch.cuda.synchronize(field.device)
                warmed.add(selected_mode)
            rows = [
                _episode(
                    field, oracle, field_scene, q, scene_config,
                    mode=selected_mode,
                    episode_seed=scene_config.seed + index,
                ) for index, (q, _) in enumerate(starts)
            ]
            scene_episodes[selected_mode] = rows
            episodes_by_mode[selected_mode].extend(rows)
        scene_results.append({
            "scene_id": record.scene_id,
            "split": record.split,
            "sphere_index": record.sphere_index,
            "sphere_xyz_radius_m": record.sphere.tolist(),
            "start_source": "recorded_queries_and_witness" if record.witness_q is not None
                            else "recorded_queries_and_home",
            "training_coverage": _coverage_from_training_spheres(
                training_spheres, record.sphere,
            ),
            "summary": {name: _summarize(rows)
                        for name, rows in scene_episodes.items()},
            "episodes": scene_episodes,
        })
    summary = {name: {
        **_summarize(rows),
        "distinct_scenes": len(records),
        "scenes_with_contact": sum(
            any(episode["success"] for episode in scene["episodes"][name])
            for scene in scene_results
        ),
    } for name, rows in episodes_by_mode.items()}
    for mode_summary in summary.values():
        mode_summary["scene_success_rate"] = (
            mode_summary["scenes_with_contact"] / len(records)
        )
    return {
        "task": "Panda any-arm-geom button contact over distinct dataset scenes",
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "scene_data": str(Path(scene_data).expanduser().resolve()),
        "split": split,
        "device": str(field.device),
        "configuration": asdict(config),
        "trials_per_scene": config.episodes,
        "scene_count": len(records),
        "summary": summary,
        "scenes": scene_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-data", type=Path,
                        help="original NPZ dataset to verify single-sphere training coverage")
    parser.add_argument("--scene-data", type=Path,
                        help="evaluate every distinct button scene in this NPZ dataset")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"),
                        default="test", help="dataset scenes to evaluate with --scene-data")
    parser.add_argument("--trials-per-scene", type=int, default=1,
                        help="contact trials at each selected dataset button position")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument("--sphere-index", type=int, help="0, 1, or 2; fixed-scene mode defaults to 1")
    parser.add_argument("--sphere-center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="override the selected sphere centre in metres")
    parser.add_argument("--sphere-radius", type=float,
                        help="override the selected sphere radius in metres (paper button: 0.01)")
    parser.add_argument("--mode", choices=("both", "regression", "mc"), default="mc")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=10,
                        help="trials at the one fixed sphere (without --scene-data)")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--time-limit-s", type=float, default=3.0,
                        help="paper button-pressing limit per trial, in seconds")
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--max-joint-step", type=float, default=0.04)
    parser.add_argument("--edge-resolution", type=float, default=0.005)
    parser.add_argument("--min-direction-norm", type=float, default=1e-5)
    parser.add_argument("--safety-margin", type=float, default=0.0)
    parser.add_argument("--start-noise", type=float, default=0.25)
    parser.add_argument("--start-gap-min", type=float, default=0.015)
    parser.add_argument("--start-gap-max", type=float, default=0.30)
    parser.add_argument("--max-start-attempts", type=int, default=5000)
    parser.add_argument("--mc-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.trials_per_scene < 1:
        parser.error("--trials-per-scene must be positive")
    if args.scene_data is not None and any(value is not None for value in (
        args.sphere_index, args.sphere_center, args.sphere_radius,
    )):
        parser.error("--scene-data selects recorded button geometry; omit sphere overrides")
    config = ContactConfig(
        episodes=args.trials_per_scene if args.scene_data is not None else args.episodes,
        max_steps=args.max_steps,
        time_limit_s=args.time_limit_s,
        step_size=args.step_size, max_joint_step=args.max_joint_step,
        edge_resolution=args.edge_resolution,
        min_direction_norm=args.min_direction_norm,
        safety_margin=args.safety_margin, start_noise=args.start_noise,
        start_gap_min=args.start_gap_min, start_gap_max=args.start_gap_max,
        max_start_attempts=args.max_start_attempts,
        mc_samples=args.mc_samples, seed=args.seed,
    )
    if args.scene_data is not None:
        report = evaluate_dataset(
            args.checkpoint, args.scene_data, split=args.split,
            mode=args.mode, config=config, device=args.device,
            training_data=args.training_data,
        )
    else:
        report = evaluate(
            args.checkpoint, sphere_index=1 if args.sphere_index is None else args.sphere_index,
            sphere_center=args.sphere_center, sphere_radius=args.sphere_radius,
            mode=args.mode, config=config, device=args.device,
            training_data=args.training_data,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
