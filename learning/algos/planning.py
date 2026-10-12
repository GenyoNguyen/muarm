"""Lazy registry for planning policies that run without offline training."""
from __future__ import annotations

from importlib import import_module


PLANNING_ALGOS = ("star_worlds", "differentiable_mpc", "ik_rrt_connect")

_POLICIES = {
    "star_worlds": ("learning.algos.star_worlds", "StarWorldsPolicy"),
    "differentiable_mpc": ("learning.algos.differentiable_mpc", "DifferentiableMPCPolicy"),
    "ik_rrt_connect": ("learning.algos.ik_rrt_connect", "IKRRTConnectPolicy"),
}


def make_planning_policy(algo: str, env, *, seed: int | None = None):
    """Construct a planner for the current MuJoCo environment."""
    try:
        module_name, class_name = _POLICIES[algo.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown planning algorithm: {algo}") from exc
    policy_class = getattr(import_module(module_name), class_name)
    if algo.lower() == "ik_rrt_connect":
        return policy_class(env, seed=seed)
    return policy_class(env)
