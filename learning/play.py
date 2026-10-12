"""Play trained policies and planners; run CDFlow contact projection headlessly.

For viewer-enabled policies, rendering follows the simulation timestep.

Examples
--------
# Play RL model
python learning/play.py --robot panda --task reach --algo sac \
    --model learning/runs/panda_reach_sac.zip

# Play BC model
python learning/play.py --robot panda --task reach --algo bc \
    --model learning/runs/panda_reach_bc.pt

# 10 episodes
python learning/play.py --robot panda --task push --algo sac \
    --model learning/runs/panda_push_sac.zip --episodes 10

# CDF-MPPI planner
python learning/play.py --robot panda --task reach_obstacles --algo cdf_mppi \
    --model learning/runs/panda_reach_obstacles_cdf_mppi.pt

# Paper-style CDFlow button contact (headless geometric projection)
python learning/play.py --robot panda --task cdflow_button --algo cdflow \
    --model learning/runs/panda_cdflow_button_cdflow.pt \
    --cdflow-scene-data learning/runs/cdflow_button.npz --headless
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.envs.registry import make_env
from learning.algos.rl_trainer import load_rl_model
from learning.algos.il.bc import BCPolicy
from learning.algos.cdf_mppi import load_cdf_mppi
from learning.algos.planning import PLANNING_ALGOS, make_planning_policy
from learning.utils.inference_metrics import InferenceEpisodeMetrics


def _cdflow_scene_from_dataset(
    dataset_path: str, scene_id: int | None,
) -> tuple[int, np.ndarray, float]:
    """Select one single-button scene from a generated CDFlow dataset."""
    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {"scene", "scene_mask", "scene_id"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"CDFlow scene dataset lacks {sorted(missing)}")
        scene_ids = np.asarray(archive["scene_id"], dtype=np.int64)
        rows = np.asarray(archive["scene"], dtype=np.float64)
        masks = np.asarray(archive["scene_mask"], dtype=bool)
    if scene_ids.ndim != 1 or rows.ndim != 3 or rows.shape[-1] != 4:
        raise ValueError("CDFlow scene dataset has invalid shapes")
    if masks.shape != rows.shape[:2] or len(scene_ids) != len(rows) or not len(rows):
        raise ValueError("CDFlow scene dataset has inconsistent scene rows")
    selected_id = int(scene_ids[0]) if scene_id is None else scene_id
    matches = np.flatnonzero(scene_ids == selected_id)
    if not len(matches):
        raise ValueError(f"CDFlow scene {selected_id} is absent from {dataset_path}")
    row = int(matches[0])
    active = np.flatnonzero(masks[row])
    if len(active) != 1:
        raise ValueError("CDFlow button playback requires exactly one active sphere")
    sphere_index = int(active[0])
    sphere = rows[row, sphere_index]
    if not np.all(np.isfinite(sphere)) or sphere[3] <= 0:
        raise ValueError("CDFlow button geometry is invalid")
    return sphere_index, sphere[:3].copy(), float(sphere[3])


def _play_cdflow_contact(
    model_path: str, episodes: int, max_steps: int | None,
    seed: int | None, mode: str, samples: int, device: str,
    scene_data: str | None, scene_id: int | None,
    button_center: tuple[float, float, float] | None,
    button_radius: float, time_limit_s: float,
) -> None:
    """Run the paper's direct contact projection, outside the reach env."""
    from scripts.eval_cdflow_contact import ContactConfig, evaluate

    if scene_data is not None:
        if button_center is not None:
            raise ValueError("Use either --cdflow-scene-data or --cdflow-button-center")
        sphere_index, sphere_center, sphere_radius = _cdflow_scene_from_dataset(
            scene_data, scene_id,
        )
    else:
        if scene_id is not None:
            raise ValueError("--cdflow-scene-id requires --cdflow-scene-data")
        sphere_index = 1
        sphere_center = (None if button_center is None else
                         np.asarray(button_center, dtype=np.float64))
        sphere_radius = button_radius
    report = evaluate(
        Path(model_path), sphere_index=sphere_index,
        sphere_center=sphere_center, sphere_radius=sphere_radius,
        mode=mode, device=device,
        training_data=Path(scene_data) if scene_data is not None else None,
        config=ContactConfig(
            episodes=episodes, max_steps=max_steps or 100,
            time_limit_s=time_limit_s, mc_samples=samples,
            seed=0 if seed is None else seed,
        ),
    )
    sphere = report["scene"]["sphere_xyz_radius_m"]
    print(f"CDFlow whole-body contact: sphere={sphere}, mode={mode}")
    for index, row in enumerate(report["episodes"][mode], 1):
        label = "SUCCESS" if row["success"] else row["reason"]
        print(
            f"  ep {index:3d}/{episodes}  {label}  steps={row['steps']:4d}"
            f"  gap={row['final_gap_m']:.5f} m"
            f"  field_time={row['total_field_query_ms']:.1f} ms"
        )
    summary = report["summary"][mode]
    print(
        f"\nContact success rate: {summary['success_rate']:.2f} "
        f"({summary['successes']}/{episodes})"
    )


# ---------------------------------------------------------------------------
# Real-time step helper
# ---------------------------------------------------------------------------

def _rt_step(env, action: np.ndarray) -> tuple:
    """Step env and sleep to match wall-clock simulation time."""
    t0 = time.perf_counter()
    result = env.step(action)
    # env.dt = model.opt.timestep * n_substeps  (set in MuJocoRobotEnv)
    dt = getattr(env, "dt", 0.02)
    elapsed = time.perf_counter() - t0
    sleep_time = dt - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
    return result


# ---------------------------------------------------------------------------
# Policy wrappers
# ---------------------------------------------------------------------------

def _load_policy(
    algo: str, model_path: str | None, env=None, *, seed: int | None = None,
    cdflow_mode: str = "regression", cdflow_samples: int = 64,
    cdflow_device: str = "cpu",
):
    """Load or construct a planner or observation-to-action policy."""
    algo_key = algo.lower()

    if algo_key in PLANNING_ALGOS:
        if env is None:
            raise ValueError(f"{algo_key} needs a Panda environment.")
        return make_planning_policy(algo_key, env, seed=seed)

    if model_path is None:
        raise ValueError(f"{algo_key} requires --model.")

    if algo_key == "cdf_mppi":
        if env is None:
            raise ValueError("CDF-MPPI needs a Panda environment for pose checks.")
        return load_cdf_mppi(env, model_path)

    elif algo_key == "cdflow_mppi":
        if env is None:
            raise ValueError("CDFlow-MPPI needs a Panda environment for pose checks.")
        from learning.algos.cdflow_mppi import load_cdflow_mppi
        return load_cdflow_mppi(
            env, model_path, device=cdflow_device,
            mode=cdflow_mode, samples=cdflow_samples,
        )

    elif algo_key == "bc":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        policy_net = BCPolicy(ckpt["obs_dim"], ckpt["act_dim"], hidden=tuple(ckpt["hidden"]))
        policy_net.load_state_dict(ckpt["state_dict"])
        policy_net.eval()

        def _bc_predict(obs_dict):
            x = torch.from_numpy(obs_dict["observation"]).unsqueeze(0)
            with torch.no_grad():
                return policy_net(x).squeeze(0).numpy()

        return _bc_predict

    else:
        sb3_model = load_rl_model(algo_key, model_path)

        def _sb3_predict(obs_dict):
            action, _ = sb3_model.predict(obs_dict, deterministic=True)
            return action

        return _sb3_predict


# ---------------------------------------------------------------------------
# Play loop
# ---------------------------------------------------------------------------

def _log_rollout_metrics(writer, episode: int,
                         metrics: InferenceEpisodeMetrics, steps: int) -> None:
    """Write computation time and trajectory length for one completed episode."""
    if writer is None:
        return
    writer.add_scalar("rollout/computation_time_s", metrics.computation_time_s, episode)
    writer.add_scalar("rollout/trajectory_length_m", metrics.trajectory_length_m, episode)
    writer.add_scalar("rollout/num_steps", steps, episode)


def play(
    robot: str, task: str, algo: str, model_path: str | None, episodes: int,
    *, render: bool = True, max_steps: int | None = None, seed: int | None = None,
    tensorboard: bool = False, tensorboard_dir: str = "learning/runs/tb/inference",
    horizon: int | None = None, n_substeps: int | None = None,
    physics_timestep: float | None = None,
    distance_threshold: float | None = None,
    cdflow_mode: str = "mc", cdflow_samples: int = 64,
    cdflow_device: str = "cpu",
    cdflow_scene_data: str | None = None, cdflow_scene_id: int | None = None,
    cdflow_button_center: tuple[float, float, float] | None = None,
    cdflow_button_radius: float = 0.01, cdflow_time_limit_s: float = 3.0,
) -> None:
    if algo.lower() == "cdflow":
        if robot != "panda" or task != "cdflow_button":
            raise ValueError("CDFlow contact projection requires --robot panda --task cdflow_button")
        if model_path is None:
            raise ValueError("CDFlow requires --model")
        if render:
            raise ValueError("CDFlow geometric contact playback is headless; pass --headless")
        if tensorboard:
            raise ValueError("CDFlow contact playback does not support --tensorboard")
        _play_cdflow_contact(
            model_path, episodes, max_steps, seed, cdflow_mode, cdflow_samples,
            cdflow_device, cdflow_scene_data, cdflow_scene_id,
            cdflow_button_center, cdflow_button_radius, cdflow_time_limit_s,
        )
        return
    cdf_mode = algo.lower() in {"cdf_mppi", "cdflow_mppi"}
    env_kwargs = {} if horizon is None else {"horizon": horizon}
    if n_substeps is not None:
        env_kwargs["n_substeps"] = n_substeps
    elif algo.lower() == "differentiable_mpc":
        # The reference MPC model advances 0.2 s per solve. Panda's 0.002 s
        # physics step therefore needs 100 substeps per control action.
        env_kwargs["n_substeps"] = 100
    if physics_timestep is not None:
        env_kwargs["physics_timestep"] = physics_timestep
    task_kwargs = {}
    if distance_threshold is not None:
        task_kwargs["distance_threshold"] = distance_threshold
    if algo.lower() == "cdflow_mppi":
        task_kwargs["use_cdf"] = False
    env = make_env(
        robot, task, render_mode=None if cdf_mode or not render else "human",
        task_kwargs=task_kwargs, **env_kwargs,
    )
    successes = 0
    writer = None
    try:
        predict = _load_policy(
            algo, model_path, env=env, seed=seed,
            cdflow_mode=cdflow_mode, cdflow_samples=cdflow_samples,
            cdflow_device=cdflow_device,
        )
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            run_name = f"{robot}_{task}_{algo}_{datetime.now():%Y%m%d-%H%M%S-%f}"
            log_dir = Path(tensorboard_dir) / run_name
            writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard inference logs: {log_dir}")
        for ep in range(episodes):
            if cdf_mode:
                env.render_mode = None
            obs, _ = env.reset(seed=None if seed is None else seed + ep)
            metrics = InferenceEpisodeMetrics()
            if cdf_mode:
                if seed is not None:
                    predict.cfg.seed = seed + ep
                compute_start = time.perf_counter()
                predict.start_episode()
                metrics.add_computation(time.perf_counter() - compute_start)
                metrics.observe_ee(env.get_ee_pos())
                if render:
                    env.render_mode = "human"
                    env.render()

                def _record_cdf_step(computation_time_s: float, ee_pos: np.ndarray) -> None:
                    metrics.add_computation(computation_time_s)
                    metrics.observe_ee(ee_pos)

                result = predict.rollout(
                    max_iter=max_steps or 500, render=render, on_step=_record_cdf_step,
                )
                successes += int(result.success)
                status = "SUCCESS ✓" if result.success else f"{result.status} ✗"
                detail = (
                    f"  gap={result.collision_gap:.4f} m"
                    if result.collision_gap is not None else ""
                )
                if result.ee_error is not None:
                    detail += f"  ee_error={result.ee_error:.4f} m"
                if result.minimum_gap is not None:
                    detail += f"  min_clearance={result.minimum_gap:.4f} m"
                print(
                    f"  ep {ep+1:3d}/{episodes}  {status}  "
                    f"steps={result.attempted_steps:4d}{detail}  "
                    f"compute_time={metrics.computation_time_s:.6f} s  "
                    f"trajectory_length={metrics.trajectory_length_m:.3f} m"
                )
                _log_rollout_metrics(writer, ep + 1, metrics, result.attempted_steps)
                continue
            if hasattr(predict, "reset"):
                compute_start = time.perf_counter()
                predict.reset()
                metrics.add_computation(time.perf_counter() - compute_start)
            metrics.observe_ee(env.get_ee_pos())
            done = False
            terminated = False
            steps = 0
            total_reward = 0.0
            info = {}
            any_collision = False

            while not done:
                compute_start = time.perf_counter()
                action = predict(obs)
                metrics.add_computation(time.perf_counter() - compute_start)
                if (
                    algo.lower() in PLANNING_ALGOS
                    and getattr(predict, "status", None) in {
                        "goal_in_obstacle", "ik_failed", "planning_failed", "tracking_blocked",
                        "invalid_state", "invalid_solution", "collision_unavoidable",
                    }
                ):
                    break
                obs, reward, terminated, truncated, info = (
                    _rt_step(env, action) if render else env.step(action)
                )
                total_reward += float(reward)
                any_collision |= bool(info.get("is_collision", False))
                metrics.observe_ee(env.get_ee_pos())
                steps += 1
                done = terminated or truncated or (max_steps is not None and steps >= max_steps)

            success = bool(info.get("is_success", terminated))
            successes += int(success)
            status = "SUCCESS ✓" if success else "fail ✗"
            ee_error = np.linalg.norm(env.get_ee_pos() - env.goal)
            planner_status = (
                "success" if success and algo.lower() in PLANNING_ALGOS
                else getattr(predict, "status", None)
            )
            planner_detail = f"  planner={planner_status}" if planner_status else ""
            print(
                f"  ep {ep+1:3d}/{episodes}  {status}  steps={steps:4d}  "
                f"reward={total_reward:.3f}  ee_error={ee_error:.4f} m  "
                f"collision={any_collision}  "
                f"compute_time={metrics.computation_time_s:.6f} s  "
                f"trajectory_length={metrics.trajectory_length_m:.3f} m"
                f"{planner_detail}"
            )
            _log_rollout_metrics(writer, ep + 1, metrics, steps)
    finally:
        try:
            env.close()
        finally:
            if writer is not None:
                writer.close()
    sr = successes / episodes
    print(f"\nFinal success rate: {sr:.2f}  ({successes}/{episodes})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Play a robot policy or planner in MuJoCo.")
    parser.add_argument("--robot",    type=str, default="panda",
                        help="Robot type: panda | so_arm | xarm7 | xarm7_nohand")
    parser.add_argument("--task",     type=str, default="reach",
                        choices=["reach", "reach_obstacles", "reach_obstacles_2", "cdflow_button", "push", "pick_place"],
                        help="Task name")
    parser.add_argument("--algo",     type=str, default="sac",
                        choices=["ppo", "sac", "td3", "ddpg", "bc", "cdf_mppi", "cdflow", "cdflow_mppi", *PLANNING_ALGOS],
                        help="Policy or planner algorithm")
    parser.add_argument("--model",    type=str,
                        help="Saved model path for RL, BC, CDF-MPPI, or CDFlow")
    parser.add_argument("--cdflow-mode", choices=["regression", "mc"], default="mc",
                        help="CDFlow direction: Monte Carlo boundary expectation (paper default) or fast regression")
    parser.add_argument("--cdflow-samples", type=int, default=64,
                        help="Generated boundary samples per CDFlow Monte Carlo query")
    parser.add_argument("--cdflow-device", default="cpu",
                        help="PyTorch device for the CDFlow field")
    parser.add_argument("--cdflow-scene-data", type=str,
                        help="CDFlow NPZ dataset containing a selected single-button scene")
    parser.add_argument("--cdflow-scene-id", type=int,
                        help="Scene ID in --cdflow-scene-data (default: first scene)")
    parser.add_argument("--cdflow-button-center", type=float, nargs=3,
                        metavar=("X", "Y", "Z"),
                        help="Button center in metres when no scene dataset is supplied")
    parser.add_argument("--cdflow-button-radius", type=float, default=0.01,
                        help="Button radius in metres (paper: 0.01)")
    parser.add_argument("--cdflow-time-limit-s", type=float, default=3.0,
                        help="Wall-clock contact trial limit in seconds (paper: 3)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--headless", action="store_true",
                        help="Run without the MuJoCo viewer or real-time sleeps")
    parser.add_argument("--max-steps", type=int,
                        help="Limit steps per episode (useful for a quick smoke run)")
    parser.add_argument("--horizon", type=int,
                        help="Environment episode limit in control steps (default: 200)")
    parser.add_argument("--n-substeps", type=int,
                        help="MuJoCo substeps per control step (default: 20, or 100 for MPC)")
    parser.add_argument("--physics-timestep", type=float,
                        help="Panda physics step in seconds (default: model setting)")
    parser.add_argument("--distance-threshold", type=float,
                        help="Task success distance in metres (default: task setting)")
    parser.add_argument("--seed", type=int,
                        help="Seed goal sampling; episode seeds increase by one")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Log inference metrics to TensorBoard")
    parser.add_argument("--tensorboard-dir", type=str, default="learning/runs/tb/inference",
                        help="Parent directory for inference logs (with --tensorboard)")
    args = parser.parse_args()
    if args.task == "cdflow_button" and args.algo != "cdflow":
        parser.error("cdflow_button is the CDFlow contact task; use --algo cdflow")
    if args.algo == "cdflow" and not args.headless:
        parser.error("CDFlow geometric contact playback is headless; pass --headless")
    if args.algo not in PLANNING_ALGOS and args.model is None:
        parser.error(f"--model is required for {args.algo}")
    if args.episodes < 1 or args.cdflow_samples < 1 or any(
        value is not None and value < 1
        for value in (args.max_steps, args.horizon, args.n_substeps)
    ) or any(
        value is not None and value <= 0
        for value in (args.physics_timestep, args.distance_threshold,
                      args.cdflow_button_radius, args.cdflow_time_limit_s)
    ):
        parser.error("Episode limits, substeps, physics timestep, and distance threshold must be positive")

    play(
        robot=args.robot,
        task=args.task,
        algo=args.algo,
        model_path=args.model,
        episodes=args.episodes,
        render=not args.headless,
        max_steps=args.max_steps,
        seed=args.seed,
        tensorboard=args.tensorboard,
        tensorboard_dir=args.tensorboard_dir,
        horizon=args.horizon,
        n_substeps=args.n_substeps,
        physics_timestep=args.physics_timestep,
        distance_threshold=args.distance_threshold,
        cdflow_mode=args.cdflow_mode,
        cdflow_samples=args.cdflow_samples,
        cdflow_device=args.cdflow_device,
        cdflow_scene_data=args.cdflow_scene_data,
        cdflow_scene_id=args.cdflow_scene_id,
        cdflow_button_center=args.cdflow_button_center,
        cdflow_button_radius=args.cdflow_button_radius,
        cdflow_time_limit_s=args.cdflow_time_limit_s,
    )


if __name__ == "__main__":
    main()
