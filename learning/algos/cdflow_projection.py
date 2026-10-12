"""Paper-style CDFlow whole-body contact projection for the Panda.

The CDFlow field points outward from the collision boundary.  To make contact
with a selected sphere, project the seven-joint pose along its *negative*
direction.  This controller does not sample MPPI actions or solve an IK goal.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from learning.cdflow.collision import PandaCollisionOracle
from learning.cdflow.field import CDFlowField
from learning.cdflow.scene import SphereScene


@dataclass(frozen=True)
class CDFlowProjectionConfig:
    """Joint-space projection and geometric safety settings."""

    max_steps: int = 100
    time_limit_s: float = 3.0
    step_size: float = 0.05       # Euclidean joint radians
    max_joint_step: float = 0.04  # radians per joint and projection
    edge_resolution: float = 0.005
    min_direction_norm: float = 1e-5
    safety_margin: float = 0.0    # floor/self-contact clearance, metres
    mode: str = "mc"
    mc_samples: int = 64

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.mc_samples < 1:
            raise ValueError("step and Monte Carlo sample counts must be positive")
        if min(self.time_limit_s, self.step_size, self.max_joint_step,
               self.edge_resolution, self.min_direction_norm) <= 0:
            raise ValueError("projection and time parameters must be positive")
        if self.safety_margin < 0:
            raise ValueError("safety margin must be nonnegative")
        if self.mode not in {"mc", "regression"}:
            raise ValueError("mode must be 'mc' or 'regression'")


@dataclass(frozen=True)
class CDFlowProjectionResult:
    trajectory: np.ndarray
    status: str
    attempted_steps: int
    initial_gap_m: float
    final_gap_m: float
    path_length_rad: float
    field_queries: int
    mean_field_query_ms: float | None
    total_field_query_ms: float

    @property
    def success(self) -> bool:
        return self.status == "contact"


def arm_sphere_contact(oracle: PandaCollisionOracle) -> bool:
    """Require an actual arm/hand/finger-to-active-sphere MuJoCo contact."""
    obstacle_ids = set(oracle._obstacle_ids)
    arm_ids = set(oracle._arm_geoms)
    return any(
        contact.dist <= 0.0 and (
            (contact.geom1 in obstacle_ids and contact.geom2 in arm_ids)
            or (contact.geom2 in obstacle_ids and contact.geom1 in arm_ids)
        )
        for contact in oracle.data.contact
    )


def _synchronize_if_cuda(field: CDFlowField) -> None:
    device = torch.device(field.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class CDFlowProjectionPolicy:
    """Repeatedly project toward whole-body sphere contact using CDFlow.

    ``oracle`` must own a separate MuJoCo model.  Its active obstacle set must
    correspond to ``scene``; the oracle is responsible for floor and robot
    self-contact checks.  A callback can copy accepted poses into a live viewer.
    """

    def __init__(
        self,
        field: CDFlowField,
        oracle: PandaCollisionOracle,
        scene: SphereScene,
        config: CDFlowProjectionConfig | None = None,
        *,
        contact_checker: Callable[[PandaCollisionOracle], bool] = arm_sphere_contact,
    ) -> None:
        self.field = field
        self.oracle = oracle
        self.scene = scene
        self.config = config or CDFlowProjectionConfig()
        sphere_count = len(scene.spheres) if hasattr(scene, "spheres") else None
        max_spheres = getattr(field, "max_spheres", None)
        if (sphere_count is not None and max_spheres is not None
                and sphere_count > max_spheres):
            raise ValueError(
                f"checkpoint supports {max_spheres} spheres; "
                f"scene has {sphere_count}"
            )
        model = getattr(field, "model", None)
        if model is not None:
            trained_limits = np.asarray(model.config.joint_limits, dtype=np.float64)
            if not np.allclose(trained_limits, oracle.joint_limits, atol=2e-4, rtol=0):
                raise ValueError("CDFlow checkpoint joint limits differ from the Panda model")
        self.contact_checker = contact_checker
        self.current_q: np.ndarray | None = None
        self.initial_gap_m: float | None = None
        self.path_length_rad = 0.0
        self.query_seconds: list[float] = []

    def start_episode(self, start_q: np.ndarray) -> None:
        """Initialize from a safe free pose, without changing any live env."""
        q = np.asarray(start_q, dtype=np.float64).copy()
        if not self.oracle.in_limits(q):
            raise ValueError("start_q must be finite and within Panda joint limits")
        gap = self.oracle.gap(q)
        if not np.isfinite(gap) or gap <= 0.0 or self.contact_checker(self.oracle):
            raise ValueError("start_q must be free of the target sphere")
        constraint_gap = self.oracle.constraint_gap(q)
        if not np.isfinite(constraint_gap) or constraint_gap <= self.config.safety_margin:
            raise ValueError("start_q violates floor or self-contact safety constraints")
        self.current_q = q
        self.initial_gap_m = float(gap)
        self.path_length_rad = 0.0
        self.query_seconds = []

    def step(self, *, deadline: float | None = None) -> str:
        """Take one bounded projection; return contact, continuing, or a stop reason."""
        if self.current_q is None:
            raise RuntimeError("Call start_episode() before step()")
        q = self.current_q
        _synchronize_if_cuda(self.field)
        before = perf_counter()
        distance, gradient = self.field.distance_and_gradient(
            self.scene, q, mode=self.config.mode, samples=self.config.mc_samples,
        )
        _synchronize_if_cuda(self.field)
        self.query_seconds.append(perf_counter() - before)
        if deadline is not None and perf_counter() >= deadline:
            return "timeout"
        distance_value = float(torch.as_tensor(distance).reshape(-1)[0].detach().cpu())
        direction = np.asarray(
            torch.as_tensor(gradient).reshape(-1, 7)[0].detach().cpu(),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(direction))
        if not (np.isfinite(distance_value) and np.all(np.isfinite(direction))):
            return "nonfinite_field"
        if norm < self.config.min_direction_norm:
            return "vanishing_direction"

        delta = -self.config.step_size * direction / norm
        delta = np.clip(delta, -self.config.max_joint_step, self.config.max_joint_step)
        lower, upper = self.oracle.joint_limits.T
        next_q = np.clip(q + delta, lower, upper)
        if np.max(np.abs(next_q - q)) < 1e-10:
            return "joint_limit_stall"

        count = max(1, int(np.ceil(
            np.max(np.abs(next_q - q)) / self.config.edge_resolution
        )))
        for index in range(1, count + 1):
            if deadline is not None and perf_counter() >= deadline:
                return "timeout"
            trial = q + (index / count) * (next_q - q)
            constraint_gap = self.oracle.constraint_gap(trial)
            if (not np.isfinite(constraint_gap)
                    or constraint_gap <= self.config.safety_margin):
                return "unsafe_non_obstacle_edge"
            if self.contact_checker(self.oracle):
                self.path_length_rad += float(np.linalg.norm(trial - q))
                self.current_q = trial
                return "contact"
            if self.oracle.gap(trial) <= 0.0:
                return "sphere_gap_without_arm_contact"

        self.path_length_rad += float(np.linalg.norm(next_q - q))
        self.current_q = next_q
        return "continuing"

    def rollout(
        self,
        *,
        max_steps: int | None = None,
        time_limit_s: float | None = None,
        on_step: Callable[[float, np.ndarray], None] | None = None,
    ) -> CDFlowProjectionResult:
        """Project until whole-body contact, safety stop, or the 3 s budget."""
        if self.current_q is None or self.initial_gap_m is None:
            raise RuntimeError("Call start_episode() before rollout()")
        steps = self.config.max_steps if max_steps is None else max_steps
        timeout = self.config.time_limit_s if time_limit_s is None else time_limit_s
        if steps < 1 or timeout <= 0:
            raise ValueError("max_steps and time_limit_s must be positive")
        deadline = perf_counter() + timeout
        trajectory = [self.current_q.copy()]
        status = "max_steps"
        attempted = 0
        for attempted in range(1, steps + 1):
            if perf_counter() >= deadline:
                status = "timeout"
                attempted -= 1
                break
            before = perf_counter()
            status = self.step(deadline=deadline)
            if status in {"continuing", "contact"}:
                trajectory.append(self.current_q.copy())
                if on_step is not None:
                    on_step(perf_counter() - before, self.current_q.copy())
            if status != "continuing":
                break
        if status == "continuing":
            status = "max_steps"
        final_gap = float(self.oracle.gap(self.current_q))
        return CDFlowProjectionResult(
            trajectory=np.stack(trajectory), status=status,
            attempted_steps=attempted,
            initial_gap_m=self.initial_gap_m, final_gap_m=final_gap,
            path_length_rad=self.path_length_rad,
            field_queries=len(self.query_seconds),
            mean_field_query_ms=(1000 * float(np.mean(self.query_seconds))
                                 if self.query_seconds else None),
            total_field_query_ms=1000 * float(sum(self.query_seconds)),
        )
