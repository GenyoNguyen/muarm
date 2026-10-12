"""One-step CDF-MPPI planning for the Franka Panda.

The source joint-space MPPI update is applied to the existing Reach task.
Each sampled Cartesian task goal is mapped to a joint goal before planning;
MuJoCo checks and displays proposed poses without stepping position servos.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable
import time

import mujoco
import numpy as np
import torch


_START = np.array([-1.57, 0.4, 0.0, -1.2708, 0.0, 1.8675, 0.0], dtype=np.float32)


class InvalidFieldError(RuntimeError):
    """A learned field cannot safely guide the current MPPI step."""


class AmbiguousFieldError(InvalidFieldError):
    """The local outward direction is too small to guide avoidance."""


@dataclass
class CDFMPPIConfig:
    """Parameters of the source's horizon-one Franka MPPI planner."""

    n_samples: int = 200
    dt: float = 0.01
    beta: float = 1.0
    u_clip: float = 2.7
    alpha_mean: float = 0.5
    alpha_cov: float = 0.5
    cov_reg: float = 1e-3
    init_cov_diag: float = 1.0
    alpha1: float = 2000.0
    alpha2: float = 1000.0
    activate: float = 0.5
    edge_resolution: float = 0.02
    goal_ik_seeds: int = 24
    goal_nearby_ik_seeds: int = 256
    goal_nearby_radius_fraction: float = 0.95
    seed: int | None = None


@dataclass
class CDFMPPIRollout:
    trajectory: np.ndarray
    status: str
    attempted_steps: int
    collision_gap: float | None = None
    ee_error: float | None = None
    minimum_gap: float | None = None

    @property
    def success(self) -> bool:
        return self.status == "success"


class CDFMPPIPolicy:
    """Run the source joint-space planner in a Panda obstacle environment."""

    def __init__(
        self, env, cfg: CDFMPPIConfig | None = None, *, require_cdf: bool = True,
    ):
        from learning.robots.panda import FrankaPandaEnv

        self.env = env.unwrapped
        if (
            not isinstance(self.env, FrankaPandaEnv)
            or self.env.task_name not in {"reach_obstacles", "reach_obstacles_2"}
            or self.env.action_space.shape != (7,)
        ):
            raise ValueError(
                "CDF-MPPI requires --robot panda and --task reach_obstacles "
                "or reach_obstacles_2."
            )
        self.cfg = cfg or CDFMPPIConfig()
        if self.cfg.edge_resolution <= 0:
            raise ValueError("edge_resolution must be positive.")
        if self.cfg.goal_ik_seeds < 0 or self.cfg.goal_nearby_ik_seeds < 0:
            raise ValueError("goal IK seed counts must be nonnegative.")
        if not 0 < self.cfg.goal_nearby_radius_fraction < 1:
            raise ValueError("goal_nearby_radius_fraction must be between zero and one.")
        self.cdf = getattr(self.env.task, "_cdf", None)
        if require_cdf and self.cdf is None:
            raise ValueError("CDF-MPPI requires a task with its pretrained CDF loaded.")
        self.device = self.cdf.device if self.cdf is not None else torch.device("cpu")
        self.points = self.env.task._obstacle_points
        self.limits = torch.as_tensor(
            self.env.model.jnt_range[:7].copy(), dtype=torch.float32, device=self.device
        )
        self.eye = torch.eye(7, dtype=torch.float32, device=self.device)
        self._collision_data = mujoco.MjData(self.env.model)
        self._robot_geoms = tuple(
            i for i in range(self.env.model.ngeom)
            if self.env.model.geom_bodyid[i] > 0
            and (self.env.model.geom_contype[i] or self.env.model.geom_conaffinity[i])
        )
        link1_id = self.env.model.body("link1").id
        self._arm_geoms = tuple(
            i for i in self._robot_geoms
            if self.env.model.geom_bodyid[i] >= link1_id
        )
        self._floor_geom = self.env.model.geom("floor").id
        self.reset()

    def reset(self) -> None:
        """Reset the MPPI distribution, as the source does for a new run."""
        self.mean = torch.zeros(7, dtype=torch.float32, device=self.device)
        self.cov = self.cfg.init_cov_diag * self.eye.clone()
        self.generator = None
        if self.cfg.seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(self.cfg.seed)
        self.current_q: torch.Tensor | None = None
        self.goal: torch.Tensor | None = None
        self.desired_goal: np.ndarray | None = None
        self.invalid_start = False

    def _set_pose(self, data: mujoco.MjData, q: np.ndarray) -> None:
        data.qpos[:7] = q
        data.qvel[:7] = 0.0
        data.ctrl[:7] = q
        mujoco.mj_forward(self.env.model, data)

    def start_episode(self) -> None:
        """Map the displayed task goal to joints without changing the task."""
        self.reset()
        self.points = self.env.task._obstacle_points
        q_start = _START.copy()
        self._set_pose(self.env.data, q_start)
        self.current_q = torch.as_tensor(q_start, dtype=torch.float32, device=self.device)
        self.desired_goal = self.env.goal.copy()
        self.invalid_start = self._safety_gap(q_start) <= 0.0
        if not self.invalid_start:
            limits = self.env.model.jnt_range[:7]
            exact_rng = np.random.default_rng(0)
            nearby_rng = np.random.default_rng(1)
            candidates = [(q_start, self.desired_goal)]
            candidates.extend(
                (q_seed, self.desired_goal)
                for q_seed in exact_rng.uniform(
                    limits[:, 0], limits[:, 1], size=(self.cfg.goal_ik_seeds, 7)
                )
            )
            # The task succeeds inside a Cartesian tolerance ball. A safe
            # pose near the displayed point is preferable to a colliding IK
            # solution exactly at its center.
            for _ in range(self.cfg.goal_nearby_ik_seeds):
                q_seed = nearby_rng.uniform(limits[:, 0], limits[:, 1])
                direction = nearby_rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                offset = direction * nearby_rng.uniform(
                    0.0,
                    self.env.task.distance_threshold
                    * self.cfg.goal_nearby_radius_fraction,
                )
                candidates.append((q_seed, self.desired_goal + offset))
            for q_seed, target_position in candidates:
                target = self.env._kin.fk(q_seed)
                target[:3, 3] = target_position
                q_goal, _ = self.env._kin.ik(target, q_init=q_seed, max_iters=200)
                q_goal = np.asarray(q_goal[:7], dtype=np.float64)
                if (not np.all(np.isfinite(q_goal))
                        or np.any(q_goal < limits[:, 0])
                        or np.any(q_goal > limits[:, 1])):
                    continue
                mujoco.mj_copyData(self._collision_data, self.env.model, self.env.data)
                self._set_pose(self._collision_data, q_goal)
                hand = self._collision_data.body("hand").xpos
                if (self.env.task.is_success(hand, self.desired_goal)
                        and self._safety_gap(q_goal) > 0.0):
                    self.goal = torch.as_tensor(
                        q_goal.copy(), dtype=torch.float32, device=self.device
                    )
                    break
        self.env._ee_traj = []

    @staticmethod
    def _angle(move: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        cosine = (move * reference).sum(-1) / (
            (move.norm(dim=-1) + 1e-6) * (reference.norm() + 1e-6)
        )
        return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))

    def _distance_and_gradient(
        self, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Field query hook shared by the CDF and CDFlow MPPI variants."""
        if self.cdf is None:
            raise RuntimeError("No CDF is loaded for this planner.")
        return self.cdf.distance_and_gradient(self.points, q.unsqueeze(0))

    def _cost(self, q: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        if self.goal is None:
            raise RuntimeError("The episode has no reachable joint goal.")
        distance, gradient = self._distance_and_gradient(q)
        if not torch.isfinite(distance).all() or not torch.isfinite(gradient).all():
            raise InvalidFieldError("CDF-MPPI received a nonfinite distance or gradient.")
        normal = gradient.squeeze(0)
        normal_norm = normal.norm().item()
        if normal_norm > 1e-6:
            normal = normal / normal_norm
        reference = self.goal - q
        near_obstacle = (
            distance.item() < self.cfg.activate
            and distance.item() < reference.norm().item()
        )
        if near_obstacle and normal_norm <= 1e-6:
            raise AmbiguousFieldError("No reliable outward direction near an obstacle.")
        cost = self.cfg.alpha2 * self._angle(velocities, reference)
        if near_obstacle:
            cost = cost + self.cfg.alpha1 * (
                self._angle(velocities, normal) - 90.0
            ).clamp_min(0.0)
        return cost

    def _sample_velocities(self) -> torch.Tensor:
        noise = torch.randn(
            self.cfg.n_samples, 7, dtype=torch.float32,
            device=self.device, generator=self.generator,
        )
        return (self.mean + noise @ torch.linalg.cholesky(self.cov).T).clamp(
            -self.cfg.u_clip, self.cfg.u_clip
        )

    def propose(self, q: torch.Tensor) -> np.ndarray:
        """Update MPPI, project the weighted velocity, and integrate q + dt*u."""
        velocities = self._sample_velocities()
        costs = self._cost(q, velocities)
        if not torch.isfinite(costs).all():
            raise InvalidFieldError("CDF-MPPI received nonfinite costs.")
        weights = torch.exp(-self.cfg.beta * (costs - costs.min()))
        weight_sum = weights.sum()
        raw_mean = (weights[:, None] * velocities).sum(0) / weight_sum
        centered = velocities - raw_mean
        weighted_cov = (
            torch.einsum("n,ni,nj->ij", weights, centered, centered) / weight_sum
        )
        self.mean = self.cfg.alpha_mean * self.mean + (1 - self.cfg.alpha_mean) * raw_mean
        self.cov = (
            self.cfg.alpha_cov * self.cov
            + (1 - self.cfg.alpha_cov) * weighted_cov
            + self.cfg.cov_reg * self.eye
        )

        # Source InputBounds is applied only to the selected weighted action.
        lower = (self.limits[:, 0] - q) / self.cfg.dt
        upper = (self.limits[:, 1] - q) / self.cfg.dt
        velocity = torch.maximum(torch.minimum(raw_mean, upper), lower)
        return (q + self.cfg.dt * velocity).detach().cpu().numpy().reshape(7)

    def _collision_gap(self, q_next: np.ndarray) -> float:
        """Hard-set a pose in the checker and query robot/sphere gap."""
        model, data = self.env.model, self._collision_data
        mujoco.mj_copyData(data, model, self.env.data)
        self._set_pose(data, q_next)
        endpoints = np.zeros(6)
        return min(
            mujoco.mj_geomDistance(model, data, robot, obstacle, 5.0, endpoints)
            for robot in self._robot_geoms
            for obstacle in self.env.task._obstacle_geom_ids
        )

    def _constraint_gap_current(self) -> float:
        """Floor and reported robot/self-contact clearance at the checked pose."""
        model, data = self.env.model, self._collision_data
        endpoints = np.zeros(6)
        floor_gap = min(
            mujoco.mj_geomDistance(model, data, robot, self._floor_geom, 5.0, endpoints)
            for robot in self._arm_geoms
        )
        robot_geoms = set(self._robot_geoms)
        self_gaps = [
            contact.dist for contact in data.contact
            if contact.geom1 in robot_geoms and contact.geom2 in robot_geoms
        ]
        return float(min([floor_gap, *self_gaps]))

    def _safety_gap(self, q: np.ndarray) -> float:
        """Minimum obstacle, floor, and reported self-contact clearance."""
        obstacle_gap = self._collision_gap(q)
        return min(obstacle_gap, self._constraint_gap_current())

    def _edge_collision_gap(self, q_start: np.ndarray, q_next: np.ndarray) -> float:
        """Check intermediate joint poses as well as the proposed endpoint."""
        count = max(1, int(np.ceil(
            np.max(np.abs(q_next - q_start)) / self.cfg.edge_resolution
        )))
        return min(
            self._safety_gap(q_start + (index / count) * (q_next - q_start))
            for index in range(1, count + 1)
        )

    def step(self) -> tuple[bool, float]:
        """Accept q_next only after the whole joint-space edge is collision-free."""
        if self.current_q is None:
            raise RuntimeError("Call start_episode() after env.reset() before planning.")
        q_next = self.propose(self.current_q)
        gap = self._edge_collision_gap(self.current_q.cpu().numpy(), q_next)
        if gap < 0.0:
            return False, gap
        self.current_q = torch.as_tensor(q_next.copy(), device=self.device)
        self._set_pose(self.env.data, q_next)
        return True, gap

    def rollout(
        self, max_iter: int = 500, *, render: bool = False,
        on_step: Callable[[float, np.ndarray], None] | None = None,
    ) -> CDFMPPIRollout:
        """Run until collision, displayed task-goal success, or 500 steps."""
        if self.current_q is None or self.desired_goal is None:
            raise RuntimeError("Call start_episode() after env.reset() before planning.")
        trajectory = [self.current_q.cpu().numpy().copy()]
        ee_error = float(np.linalg.norm(self.env.get_ee_pos() - self.desired_goal))
        if self.invalid_start:
            return CDFMPPIRollout(np.stack(trajectory), "invalid_start", 0, ee_error=ee_error)
        if self.env.task.is_success(self.env.get_ee_pos(), self.desired_goal):
            return CDFMPPIRollout(np.stack(trajectory), "success", 0, ee_error=ee_error)
        if self.goal is None:
            return CDFMPPIRollout(np.stack(trajectory), "unreachable_goal", 0, ee_error=ee_error)
        minimum_gap: float | None = None
        for k in range(max_iter):
            start_time = time.perf_counter()
            try:
                accepted, gap = self.step()
            except AmbiguousFieldError:
                return CDFMPPIRollout(
                    np.stack(trajectory), "ambiguous_field", k + 1,
                    ee_error=ee_error, minimum_gap=minimum_gap,
                )
            except InvalidFieldError:
                return CDFMPPIRollout(
                    np.stack(trajectory), "invalid_field", k + 1,
                    ee_error=ee_error, minimum_gap=minimum_gap,
                )
            if on_step is not None:
                on_step(time.perf_counter() - start_time, self.env.get_ee_pos())
            if not accepted:
                return CDFMPPIRollout(
                    np.stack(trajectory), "collision", k + 1, gap, ee_error,
                    minimum_gap=minimum_gap,
                )
            minimum_gap = gap if minimum_gap is None else min(minimum_gap, gap)
            trajectory.append(self.current_q.cpu().numpy().copy())
            ee_error = float(np.linalg.norm(self.env.get_ee_pos() - self.desired_goal))
            if render:
                self.env.render()
                remaining = self.cfg.dt - (time.perf_counter() - start_time)
                if remaining > 0:
                    time.sleep(remaining)
            if self.env.task.is_success(self.env.get_ee_pos(), self.desired_goal):
                return CDFMPPIRollout(
                    np.stack(trajectory), "success", k + 1,
                    ee_error=ee_error, minimum_gap=minimum_gap,
                )
        return CDFMPPIRollout(
            np.stack(trajectory), "max_iter", max_iter,
            ee_error=ee_error, minimum_gap=minimum_gap,
        )

    def save(self, save_path: str) -> str:
        path = Path(save_path)
        if path.suffix != ".pt":
            path = Path(str(path) + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "algo": "cdf_mppi",
            "version": 2,
            "config": asdict(self.cfg),
            "task_name": self.env.task_name,
            "cdf_state_dict": {
                name: value.detach().cpu() for name, value in self.cdf.model.state_dict().items()
            },
        }, path)
        return str(path)


def train_cdf_mppi(env, save_path: str, cfg: CDFMPPIConfig | None = None) -> str:
    """Save pretrained CDF weights and planner settings; there is no policy fitting."""
    policy = CDFMPPIPolicy(env, cfg)
    path = policy.save(save_path)
    print("[CDF-MPPI] Uses the pretrained Panda CDF; no offline policy training is required.")
    print(f"[CDF-MPPI] Saved planner → {path}")
    return path


def load_cdf_mppi(env, model_path: str) -> CDFMPPIPolicy:
    """Load a current or older planner checkpoint into the source-style loop."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    # TODO: WTF is a version?
    if checkpoint.get("algo") != "cdf_mppi" or checkpoint.get("version") not in {1, 2}:
        raise ValueError("Expected a CDF-MPPI planner checkpoint (version 1 or 2).")
    supported = {field.name for field in fields(CDFMPPIConfig)}
    config = {key: value for key, value in checkpoint["config"].items() if key in supported}
    policy = CDFMPPIPolicy(env, CDFMPPIConfig(**config))
    if checkpoint["task_name"] != policy.env.task_name:
        raise ValueError("CDF-MPPI checkpoint task does not match the environment task.")
    policy.cdf.model.load_state_dict(checkpoint["cdf_state_dict"])
    return policy
