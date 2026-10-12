#!/usr/bin/env python3
"""Paired, seeded Panda comparison of CDF-MPPI and CDFlow-MPPI.

Both planners receive the same task goal, initial pose, MPPI settings, and
MuJoCo edge checker. The result distinguishes a rejected colliding proposal
from a collision on the accepted trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.algos.cdf_mppi import CDFMPPIConfig, CDFMPPIPolicy
from learning.algos.cdflow_mppi import CDFlowMPPIPolicy
from learning.cdflow.field import CDFlowField
from learning.envs.registry import make_env


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _episode(env, policy, *, trial_seed: int, max_steps: int) -> dict:
    env.reset(seed=trial_seed)
    policy.cfg.seed = trial_seed
    torch.manual_seed(trial_seed)
    query_times: list[float] = []
    step_times: list[float] = []
    original_query = policy._distance_and_gradient

    def timed_query(q):
        if policy.device.type == "cuda":
            torch.cuda.synchronize(policy.device)
        start = time.perf_counter()
        result = original_query(q)
        if policy.device.type == "cuda":
            torch.cuda.synchronize(policy.device)
        query_times.append(time.perf_counter() - start)
        return result

    policy._distance_and_gradient = timed_query
    try:
        start = time.perf_counter()
        policy.start_episode()
        setup_time = time.perf_counter() - start
        initial_gap = policy._safety_gap(policy.current_q.cpu().numpy())
        goal_xyz = policy.desired_goal.copy()
        start_q = policy.current_q.cpu().numpy().copy()
        ee_positions = [env.get_ee_pos().copy()]

        def on_step(elapsed_s, ee_pos):
            step_times.append(float(elapsed_s))
            ee_positions.append(np.asarray(ee_pos, dtype=np.float64).copy())

        result = policy.rollout(max_iter=max_steps, on_step=on_step)
        accepted_clearance = min(
            initial_gap,
            result.minimum_gap if result.minimum_gap is not None else initial_gap,
        )
        return {
            "seed": trial_seed,
            "status": result.status,
            "success": result.success,
            "attempted_steps": result.attempted_steps,
            "accepted_steps": len(result.trajectory) - 1,
            "goal_xyz_m": goal_xyz.tolist(),
            "start_q_rad": start_q.tolist(),
            "ee_error_m": result.ee_error,
            "initial_clearance_m": initial_gap,
            "minimum_accepted_clearance_m": accepted_clearance,
            "rejected_proposal_gap_m": result.collision_gap,
            "accepted_trajectory_collision": accepted_clearance < 0,
            "joint_path_length_rad": float(np.linalg.norm(
                np.diff(result.trajectory, axis=0), axis=1,
            ).sum()),
            "ee_path_length_m": float(np.linalg.norm(
                np.diff(np.asarray(ee_positions), axis=0), axis=1,
            ).sum()),
            "setup_time_s": setup_time,
            "total_control_time_s": float(sum(step_times)),
            "mean_control_step_ms": 1000 * _mean(step_times) if step_times else None,
            "field_queries": len(query_times),
            "total_field_query_time_s": float(sum(query_times)),
            "mean_field_query_ms": 1000 * _mean(query_times) if query_times else None,
        }
    finally:
        policy._distance_and_gradient = original_query


def _summary(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "success_rate": _mean([float(row["success"]) for row in rows]),
        "rejected_proposal_rate": _mean([
            float(row["status"] == "collision") for row in rows
        ]),
        "accepted_trajectory_collision_rate": _mean([
            float(row["accepted_trajectory_collision"]) for row in rows
        ]),
        "mean_minimum_accepted_clearance_m": _mean([
            row["minimum_accepted_clearance_m"] for row in rows
        ]),
        "mean_ee_error_m": _mean([
            row["ee_error_m"] for row in rows if row["ee_error_m"] is not None
        ]),
        "mean_ee_path_length_m": _mean([row["ee_path_length_m"] for row in rows]),
        "mean_joint_path_length_rad": _mean([
            row["joint_path_length_rad"] for row in rows
        ]),
        "mean_control_step_ms": _mean([
            row["mean_control_step_ms"] for row in rows
            if row["mean_control_step_ms"] is not None
        ]),
        "mean_field_query_ms": _mean([
            row["mean_field_query_ms"] for row in rows
            if row["mean_field_query_ms"] is not None
        ]),
        "failure_status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows if not row["success"]})
        },
    }


def evaluate(
    model_path: str | Path,
    *,
    task: str = "reach_obstacles",
    episodes: int = 10,
    seed: int = 0,
    max_steps: int = 500,
    mode: str = "regression",
    samples: int = 64,
    device: str = "cpu",
    n_samples: int = 200,
    activate: float = 0.5,
    edge_resolution: float = 0.02,
) -> dict:
    if task not in {"reach_obstacles", "reach_obstacles_2"}:
        raise ValueError("evaluation requires a Panda obstacle-reaching task")
    if min(episodes, max_steps, samples, n_samples) <= 0:
        raise ValueError("episodes, steps, field samples, and MPPI samples must be positive")
    if activate <= 0 or edge_resolution <= 0:
        raise ValueError("activation and edge resolution must be positive")
    field = CDFlowField.load(model_path, device=device)
    env = make_env("panda", task)
    try:
        common = dict(n_samples=n_samples, activate=activate,
                      edge_resolution=edge_resolution)
        planners = {
            "cdf_mppi": CDFMPPIPolicy(env, CDFMPPIConfig(**common)),
            "cdflow_mppi": CDFlowMPPIPolicy(
                env, field, CDFMPPIConfig(**common), mode=mode, samples=samples,
            ),
        }
        rows: dict[str, list[dict]] = {name: [] for name in planners}
        for episode in range(episodes):
            trial_seed = seed + episode
            for name, policy in planners.items():
                row = _episode(env, policy, trial_seed=trial_seed, max_steps=max_steps)
                rows[name].append(row)
                print(
                    f"{name} seed={trial_seed} status={row['status']} "
                    f"steps={row['accepted_steps']} "
                    f"min_clearance={row['minimum_accepted_clearance_m']:.4f}m "
                    f"ee_error={row['ee_error_m']:.4f}m",
                    flush=True,
                )
            baseline, learned = rows["cdf_mppi"][-1], rows["cdflow_mppi"][-1]
            if (not np.array_equal(baseline["start_q_rad"], learned["start_q_rad"])
                    or not np.array_equal(baseline["goal_xyz_m"], learned["goal_xyz_m"])):
                raise RuntimeError("paired episodes did not receive the same start and goal")
        return {
            "format": "muarm-cdflow-mppi-eval-v1",
            "model": str(Path(model_path).expanduser().resolve()),
            "task": task,
            "settings": {
                "episodes": episodes, "seed": seed, "max_steps": max_steps,
                "mode": mode, "samples": samples, "device": device,
                "mppi_n_samples": n_samples, "activate_rad": activate,
                "edge_resolution_rad": edge_resolution,
            },
            "episodes": rows,
            "summary": {name: _summary(group) for name, group in rows.items()},
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="trained CDFlow .pt")
    parser.add_argument("--task", choices=("reach_obstacles", "reach_obstacles_2"),
                        default="reach_obstacles")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--mode", choices=("regression", "mc"), default="regression")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="identical MPPI velocity samples for each planner")
    parser.add_argument("--activate", type=float, default=0.5,
                        help="joint-space distance threshold in radians")
    parser.add_argument("--edge-resolution", type=float, default=0.02,
                        help="maximum joint increment between MuJoCo edge checks")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    report = evaluate(
        args.model, task=args.task, episodes=args.episodes, seed=args.seed,
        max_steps=args.max_steps, mode=args.mode, samples=args.samples,
        device=args.device, n_samples=args.n_samples, activate=args.activate,
        edge_resolution=args.edge_resolution,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Saved paired report: {args.output}")


if __name__ == "__main__":
    main()
