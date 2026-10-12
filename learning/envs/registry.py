"""Environment registry – maps (robot, task) string keys to env factory functions.

Usage:
    from learning.envs.registry import make_env
    env = make_env("panda", "reach", render_mode="human")
"""
from __future__ import annotations

from typing import Any

from learning.envs.tasks.reach import ReachTask
from learning.envs.tasks.reach_obstacles import ReachObstaclesTask
from learning.envs.tasks.reach_obstacles_2 import ReachObstacles2Task
from learning.envs.tasks.push import PushTask
from learning.envs.tasks.pick_place import PickPlaceTask


# -------------------- task factory --------------------

_TASK_MAP = {
    "reach": ReachTask,
    "reach_obstacles": ReachObstaclesTask,
    "reach_obstacles_2": ReachObstacles2Task,
    "push": PushTask,
    "pick_place": PickPlaceTask,
}


def _make_task(task_name: str, task_kwargs: dict | None = None):
    key = task_name.replace("-", "_").lower()
    if key not in _TASK_MAP:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(_TASK_MAP)}")
    return _TASK_MAP[key](**(task_kwargs or {}))


# -------------------- robot factory --------------------

def make_env(
    robot: str,
    task: str,
    *,
    render_mode: str | None = None,
    task_kwargs: dict | None = None,
    **env_kwargs: Any,
):
    """Create a (robot, task) environment.

    Args:
        robot:        "panda" | "so_arm" | "xarm7" | "xarm7_nohand"
        task:         "reach" | "reach_obstacles" | "reach_obstacles_2" | "push" | "pick_place"
        render_mode:  None (headless) | "human"
        task_kwargs:  extra kwargs forwarded to the task constructor
        **env_kwargs: extra kwargs forwarded to the env constructor

    Returns:
        A gymnasium.Env instance.
    """
    robot_key = robot.lower().replace("-", "_")
    task_key = task.replace("-", "_").lower()
    if task_key in {"reach_obstacles", "reach_obstacles_2"} and robot_key != "panda":
        raise ValueError(f"Task '{task_key}' requires the Panda-specific CDF model.")
    task_obj = _make_task(task_key, task_kwargs)

    if robot_key == "panda":
        from learning.robots.panda import FrankaPandaEnv
        return FrankaPandaEnv(
            task=task_obj,
            task_name=task_key,
            render_mode=render_mode,
            **env_kwargs,
        )
    elif robot_key in {"so_arm", "so_arm100"}:
        from learning.robots.so_arm import SoArm100Env
        return SoArm100Env(
            task=task_obj,
            render_mode=render_mode,
            **env_kwargs,
        )
    elif robot_key in {"xarm7", "xarm7_nohand"}:
        from learning.robots.xarm7 import XArm7Env
        return XArm7Env(
            task=task_obj,
            task_name=task_key,
            with_hand=robot_key == "xarm7",
            render_mode=render_mode,
            **env_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown robot '{robot}'. Available: panda, so_arm, xarm7, xarm7_nohand"
        )
