"""Generate MuJoCo collision-boundary sets for Panda CDFlow.

Example (small diagnostic):
    .venv/bin/python scripts/generate_cdflow_data.py \
        --output /tmp/cdflow_smoke.npz --num-scenes 1 \
        --queries-per-scene 1 --global-starts 3 --local-starts 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from learning.cdflow.collision import PandaCollisionOracle
from learning.cdflow.data import BoundaryDataConfig, generate_dataset, save_dataset
from learning.cdflow.scene import SphereScene, sample_scene_variants
from learning.robots.panda import _build_model


_BUTTON_LINKS = ("link6", "link4", "link7", "link5")
_BUTTON_BOUNDS_M = {
    "x": (-0.03, 0.8),
    "y": (-0.65, 0.65),
    "z": (0.10, 0.95),
}


def _has_button_contact(oracle: PandaCollisionOracle) -> bool:
    obstacles = set(oracle._obstacle_ids)
    arms = set(oracle._arm_geoms)
    return any(
        contact.dist <= 0.0
        and ((contact.geom1 in obstacles and contact.geom2 in arms)
             or (contact.geom2 in obstacles and contact.geom1 in arms))
        for contact in oracle.data.contact
    )


def _contact_scene_along_ray(
    oracle: PandaCollisionOracle,
    base: SphereScene,
    sphere_index: int,
    radius: float,
    witness_q: np.ndarray,
    geom_center: np.ndarray,
    rng: np.random.Generator,
) -> tuple[SphereScene, float] | None:
    """Place a button just inside the *outer* MuJoCo contact boundary.

    Placing it exactly at a mesh origin can yield zero distance without a
    collision pair. Ray marching out from the moving link and bisecting the
    last contact instead gives a verified physical contact witness.
    """
    active = np.zeros(len(base.spheres), dtype=bool)
    active[sphere_index] = True

    def check(center: np.ndarray) -> tuple[SphereScene, float, bool]:
        spheres = base.spheres.copy()
        spheres[sphere_index, :3] = center
        spheres[sphere_index, 3] = radius
        scene = SphereScene(spheres, active, base.names)
        oracle.set_scene(scene)
        gap = oracle.gap(witness_q)
        return scene, gap, _has_button_contact(oracle)

    directions = [rng.normal(size=3) for _ in range(12)]
    directions.extend(np.eye(3))
    directions.extend(-np.eye(3))
    for direction in directions:
        direction = np.asarray(direction, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        last_contact_t: float | None = None
        for distance in np.arange(0.01, 0.41, 0.01):
            _, gap, contact = check(geom_center + distance * direction)
            if contact:
                last_contact_t = float(distance)
            elif last_contact_t is not None and gap > 0.0:
                # The sphere has just left the robot silhouette. Bisect the
                # outer crossing, retaining the contacting side.
                lower, upper = last_contact_t, float(distance)
                for _ in range(16):
                    midpoint = 0.5 * (lower + upper)
                    _, _, touches = check(geom_center + midpoint * direction)
                    if touches:
                        lower = midpoint
                    else:
                        upper = midpoint
                center = geom_center + max(0.0, lower - min(0.002, radius / 5)) * direction
                if any(not bounds[0] <= value <= bounds[1]
                       for value, bounds in zip(center, _BUTTON_BOUNDS_M.values())):
                    break
                scene, gap, touches = check(center)
                if touches and gap <= 0.0:
                    return scene, gap
                break
    return None


def sample_button_scenes(
    model,
    base: SphereScene,
    count: int,
    *,
    sphere_index: int = 1,
    radius: float = 0.01,
    joint_noise: float = 0.7,
    min_center_separation: float = 0.05,
    max_attempts_per_scene: int = 2500,
    seed: int = 0,
) -> tuple[list[SphereScene], list[dict]]:
    """Sample varied button locations with a safe contacting Panda pose each.

    Each center comes from the world position of a collision geom at a valid
    joint configuration. This provides a concrete contact witness instead of
    merely sampling a box that may contain unreachable positions. The fixed
    MuJoCo obstacle count and names are preserved; only one sphere is active.
    """
    if count < 1 or max_attempts_per_scene < 1:
        raise ValueError("scene count and button attempts must be positive")
    if radius <= 0 or joint_noise < 0 or min_center_separation < 0:
        raise ValueError("button radius must be positive; noise and separation nonnegative")
    if not 0 <= sphere_index < len(base.spheres):
        raise ValueError("button sphere index is outside the obstacle scene")
    if not base.names:
        raise ValueError("button scenes need named MuJoCo obstacle geoms")

    oracle = PandaCollisionOracle(model)
    lower, upper = oracle.joint_limits.T
    home = (model.key_qpos[0, oracle.qpos_indices].copy() if model.nkey
            else np.mean(oracle.joint_limits, axis=1))
    representative_geoms: dict[str, int] = {}
    for body_name in _BUTTON_LINKS:
        body_id = model.body(body_name).id
        collision_geoms = [
            geom_id for geom_id in oracle._arm_geoms
            if model.geom_bodyid[geom_id] == body_id
        ]
        if not collision_geoms:
            raise ValueError(f"{body_name} has no collision geom")
        representative_geoms[body_name] = max(
            collision_geoms, key=lambda geom_id: model.geom_rbound[geom_id]
        )

    rng = np.random.default_rng(seed)
    scenes: list[SphereScene] = []
    audit: list[dict] = []
    for scene_index in range(count):
        body_name = _BUTTON_LINKS[scene_index % len(_BUTTON_LINKS)]
        geom_id = representative_geoms[body_name]
        for attempt in range(1, max_attempts_per_scene + 1):
            witness_q = (home.copy() if scene_index == 0 and attempt == 1 else
                         np.clip(home + rng.normal(0.0, joint_noise, 7), lower, upper))
            if oracle.constraint_gap(witness_q) <= 0.0:
                continue
            geom_center = oracle.data.geom_xpos[geom_id].copy()
            contact_scene = _contact_scene_along_ray(
                oracle, base, sphere_index, radius, witness_q, geom_center, rng,
            )
            if contact_scene is None:
                continue
            scene, witness_gap = contact_scene
            center = scene.spheres[sphere_index, :3]
            if any(np.linalg.norm(center - previous.spheres[sphere_index, :3])
                   < min_center_separation for previous in scenes):
                continue
            scenes.append(scene)
            audit.append({
                "scene_id": scene_index,
                "center_xyz_m": center.tolist(),
                "radius_m": radius,
                "witness_q_rad": witness_q.tolist(),
                "witness_obstacle_gap_m": float(witness_gap),
                "source_link": body_name,
                "source_geom_id": geom_id,
                "sampling_attempts": attempt,
            })
            break
        else:
            raise RuntimeError(
                f"could not sample reachable button scene {scene_index} after "
                f"{max_attempts_per_scene} attempts; relax center separation "
                "or increase button joint noise"
            )
    return scenes, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("reach_obstacles", "reach_obstacles_2"),
                        default="reach_obstacles")
    parser.add_argument(
        "--single-sphere-index", type=int, default=None, metavar="INDEX",
        help=("Keep only reach_obstacle_INDEX active for single-sphere contact data. "
              "The saved scene still has the task's full sphere cardinality; "
              "inactive rows are marked false in scene_mask."),
    )
    parser.add_argument(
        "--button-scenes", action="store_true",
        help=("Sample varied, reachable one-sphere button positions from Panda "
              "contact witnesses; defaults to radius 0.01 m."),
    )
    parser.add_argument("--button-radius", type=float, default=0.01,
                        help="button sphere radius in metres (default: 0.01)")
    parser.add_argument("--button-joint-noise", type=float, default=0.7,
                        help="joint radians around home for contact-witness sampling")
    parser.add_argument("--button-min-center-separation", type=float, default=0.05,
                        help="minimum distance between generated button centers, metres")
    parser.add_argument("--button-max-attempts", type=int, default=2500,
                        help="maximum contact-witness draws per button scene")
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--queries-per-scene", type=int, default=16)
    parser.add_argument("--global-starts", type=int, default=16)
    parser.add_argument("--local-starts", type=int, default=4)
    parser.add_argument("--local-radius", type=float, default=0.16)
    parser.add_argument("--near-distance-tolerance", type=float, default=0.06)
    parser.add_argument("--boundary-tolerance", type=float, default=0.002)
    parser.add_argument("--min-query-gap", type=float, default=0.012)
    parser.add_argument("--max-query-gap", type=float, default=0.22)
    parser.add_argument("--max-query-attempts", type=int, default=2500)
    parser.add_argument("--max-seed-attempts", type=int, default=80)
    parser.add_argument("--optimizer-maxiter", type=int, default=100)
    parser.add_argument("--center-jitter", type=float, default=0.04)
    parser.add_argument("--radius-jitter", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = _build_model(args.task)
    base = SphereScene.from_model(model)
    button_audit = None
    if args.button_scenes:
        button_index = (args.single_sphere_index
                        if args.single_sphere_index is not None else min(1, len(base.spheres) - 1))
        try:
            scenes, button_audit = sample_button_scenes(
                model, base, args.num_scenes,
                sphere_index=button_index, radius=args.button_radius,
                joint_noise=args.button_joint_noise,
                min_center_separation=args.button_min_center_separation,
                max_attempts_per_scene=args.button_max_attempts,
                seed=args.seed,
            )
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    else:
        button_index = args.single_sphere_index
    if not args.button_scenes and button_index is not None:
        if not 0 <= args.single_sphere_index < len(base.spheres):
            parser.error(
                f"--single-sphere-index must be between 0 and {len(base.spheres) - 1} "
                f"for {args.task}"
            )
        active = np.zeros(len(base.spheres), dtype=bool)
        active[args.single_sphere_index] = True
        base = SphereScene(base.spheres, mask=active, names=base.names)
    if not args.button_scenes:
        scenes = sample_scene_variants(
            base, args.num_scenes, seed=args.seed,
            center_jitter=args.center_jitter, radius_jitter=args.radius_jitter,
        )
    config = BoundaryDataConfig(
        queries_per_scene=args.queries_per_scene,
        global_starts=args.global_starts,
        local_starts=args.local_starts,
        local_radius=args.local_radius,
        near_distance_tolerance=args.near_distance_tolerance,
        boundary_tolerance=args.boundary_tolerance,
        min_query_gap=args.min_query_gap,
        max_query_gap=args.max_query_gap,
        max_query_attempts=args.max_query_attempts,
        max_seed_attempts=args.max_seed_attempts,
        optimizer_maxiter=args.optimizer_maxiter,
    )
    dataset = generate_dataset(model, scenes, config, seed=args.seed)
    metadata = json.loads(str(dataset["metadata_json"]))
    metadata["task"] = args.task
    metadata["single_sphere_index"] = button_index
    if args.button_scenes:
        for row in button_audit:
            query_index = int(np.flatnonzero(dataset["scene_id"] == row["scene_id"])[0])
            boundary_index = int(dataset["offsets"][query_index])
            row["boundary_witness_q_rad"] = dataset["boundary_q"][boundary_index].tolist()
            row["boundary_witness_gap_m"] = float(dataset["boundary_gap"][boundary_index])
        metadata["scene_generation"] = "fk_witness_button"
        metadata["button_scene_settings"] = {
            "radius_m": args.button_radius,
            "sphere_index": button_index,
            "joint_noise_rad": args.button_joint_noise,
            "min_center_separation_m": args.button_min_center_separation,
            "max_attempts_per_scene": args.button_max_attempts,
            "source_links": list(_BUTTON_LINKS),
            "workspace_bounds_m": _BUTTON_BOUNDS_M,
            "seed": args.seed,
        }
        metadata["button_scene_audit"] = button_audit
    else:
        metadata["scene_generation"] = "jittered_task_obstacles"
    dataset["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    save_dataset(args.output, dataset)
    audit = json.loads(str(dataset["metadata_json"]))
    print(f"Saved {len(dataset['q'])} queries, {len(dataset['boundary_q'])} "
          f"boundary configurations, {args.num_scenes} scenes to {args.output}")
    print(json.dumps({
        "scene_audit": audit["scene_audit"],
        "boundary_residual_max_abs_m": audit["boundary_residual_max_abs_m"],
        "mode_count_mean": audit["mode_count_mean"],
        "global_failure_rate": audit["global_failure_rate"],
        "local_failure_rate": audit["local_failure_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
