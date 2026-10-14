"""RL / IL training, CDF-MPPI preparation, and CDFlow training entry point.

Examples
--------
# RL – headless (fast)
python learning/train.py --robot panda --task reach --algo sac --timesteps 200000

# RL – with live MuJoCo viewer (1 env only)
python learning/train.py --robot panda --task reach --algo sac --timesteps 200000 --render

# IL – Behavior Cloning
python learning/train.py --robot panda --task reach --algo bc --family il

# Multi-robot
python learning/train.py --robot so_arm --task reach --algo sac --timesteps 100000
python learning/train.py --robot xarm7 --task reach --algo sac --timesteps 200000
python learning/train.py --robot xarm7_nohand --task reach --algo sac --timesteps 200000

# Panda reach with fixed obstacles and CDF observations
python learning/train.py --robot panda --task reach_obstacles --algo sac --timesteps 200000
python learning/train.py --robot panda --task reach_obstacles_2 --algo sac --timesteps 200000

# CDF-MPPI – save pretrained field + planner configuration (no policy fitting)
python learning/train.py --robot panda --task reach_obstacles --algo cdf_mppi

# CDFlow – fit the paper's scene-conditioned field from offline boundary data
python learning/train.py --robot panda --task cdflow_button --algo cdflow \
    --dataset learning/runs/cdflow_button.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.envs.registry import make_env
from learning.algos.rl_trainer import train_rl
from learning.algos.il.bc import train_bc
from learning.algos.cdf_mppi import train_cdf_mppi
from learning.algos.planning import PLANNING_ALGOS


def _env_fn(robot: str, task: str, render: bool):
    """Return a zero-arg callable that creates an env (optionally with viewer)."""
    def _factory():
        return make_env(robot, task, render_mode="human" if render else None)
    return _factory


def main():
    parser = argparse.ArgumentParser(description="Train RL/IL policies or prepare a planner.")
    parser.add_argument("--robot",      type=str, default="panda",
                        help="Robot type: panda | so_arm | xarm7 | xarm7_nohand")
    parser.add_argument("--task",       type=str, default="reach",
                        choices=["reach", "reach_obstacles", "reach_obstacles_2", "cdflow_button", "push", "pick_place"],
                        help="Task name")
    parser.add_argument("--algo",       type=str, default="sac",
                        choices=["ppo", "sac", "td3", "ddpg", "bc", "cdf_mppi", "cdflow", *PLANNING_ALGOS],
                        help="Algorithm: RL, BC, CDF-MPPI, CDFlow, or a training-free planner")
    parser.add_argument("--family",     type=str, default="rl", choices=["rl", "il", "planning"],
                        help="Algorithm family. Auto-detected for BC and planners.")
    parser.add_argument("--timesteps",  type=int, default=200_000,
                        help="Total env steps for RL training")
    parser.add_argument("--n-envs",     type=int, default=1,
                        help="Number of parallel envs (>1 → SubprocVecEnv, headless only)")
    parser.add_argument("--render",     action="store_true",
                        help="Open MuJoCo viewer during training (forces n-envs=1)")
    parser.add_argument("--save-dir",   type=str, default="learning/runs",
                        help="Directory to save trained model checkpoints")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging under save-dir/tb/")
    parser.add_argument("--dataset", type=str,
                        help="Boundary dataset NPZ (required for CDFlow training)")
    parser.add_argument("--cdflow-epochs", type=int, default=100,
                        help="CDFlow optimizer epochs")
    parser.add_argument("--cdflow-batch-size", type=int, default=32,
                        help="CDFlow query batch size")
    parser.add_argument("--cdflow-device", type=str, default="cpu",
                        help="PyTorch device for CDFlow training")
    args = parser.parse_args()

    # TODO: Uhh... WTF?
    if args.task == "cdflow_button" and args.algo != "cdflow":
        parser.error("cdflow_button is the offline CDFlow contact task; use --algo cdflow")

    # BC always belongs to IL family
    if args.algo == "bc":
        args.family = "il"
    elif args.algo in PLANNING_ALGOS:
        args.family = "planning"
        if args.robot != "panda" or args.task not in {"reach_obstacles", "reach_obstacles_2"}:
            parser.error(f"{args.algo} requires --robot panda and an obstacle-reaching task")
        print(f"[info] {args.algo} is a planning algorithm and requires no training or checkpoint.")
        print(
            f"Run: {sys.executable} learning/play.py --robot panda "
            f"--task {args.task} --algo {args.algo}"
        )
        return
    elif args.algo == "cdf_mppi":
        args.family = "planning"
        # TODO: allow other models and tensorboard logging (only for inference)
        if args.robot != "panda" or args.task not in {"reach_obstacles", "reach_obstacles_2"}:
            parser.error("cdf_mppi requires --robot panda and an obstacle-reaching task")
        if args.n_envs != 1 or args.tensorboard:
            parser.error("cdf_mppi preparation uses one environment and no TensorBoard training")
    elif args.algo == "cdflow":
        args.family = "planning"
        if args.robot != "panda" or args.task not in {
            "cdflow_button", "reach_obstacles", "reach_obstacles_2"
        }:
            parser.error("cdflow requires --robot panda and a supported sphere scene")
        if args.dataset is None or not Path(args.dataset).is_file():
            parser.error("cdflow requires an existing --dataset NPZ file")
        if args.cdflow_epochs < 1 or args.cdflow_batch_size < 1:
            parser.error("CDFlow epochs and batch size must be positive")
        if args.n_envs != 1 or args.render:
            parser.error("cdflow trains offline and does not use environments or rendering")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_stem = save_dir / f"{args.robot}_{args.task}_{args.algo}"

    if args.render and args.n_envs > 1:
        print("[warn] --render forces --n-envs 1")
        args.n_envs = 1

    # TODO: make this a family thing
    if args.algo == "cdf_mppi":
        env = make_env(args.robot, args.task)
        try:
            train_cdf_mppi(env, str(save_stem))
        finally:
            env.close()

    elif args.algo == "cdflow":
        from scripts.train_cdflow import train_cdflow
        train_cdflow(
            args.dataset, str(save_stem) + ".pt",
            epochs=args.cdflow_epochs,
            batch_size=args.cdflow_batch_size,
            device=args.cdflow_device,
        )

    elif args.family == "il" or args.algo == "bc":
        # IL: single env, collect demos then train
        env = make_env(args.robot, args.task,
                       render_mode="human" if args.render else None)
        train_bc(env, str(save_stem))
        env.close()

    else:
        tb_log = str(save_dir / "tb") if args.tensorboard else None
        train_rl(
            env_fn=_env_fn(args.robot, args.task, render=args.render),
            algo=args.algo,
            total_timesteps=args.timesteps,
            save_path=str(save_stem),
            n_envs=args.n_envs,
            tensorboard_log=tb_log,
        )

    suffix = ".zip" if args.algo in {"ppo", "sac", "td3", "ddpg"} else ".pt"
    print(f"\nDone. Model saved to: {save_stem}{suffix}")


if __name__ == "__main__":
    main()
