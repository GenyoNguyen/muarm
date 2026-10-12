"""Reactive Star-Worlds reaching for the MuJoCo Franka Panda.

The Franka PyBullet example at https://github.com/saayuj/star-worlds is the
logic reference: inflate/cluster workspace obstacles, modulate an EE velocity,
solve position IK, then bias the arm away from obstacles in the EE null space.
This module uses the spherical MuJoCo obstacles directly.  A cluster's proxy
is the convex hull of its inflated spheres, whose directional support can be
computed exactly without mesh sampling.  The result is conservative for a
non-convex sphere union, so actual robot/obstacle geometry is checked before
the joint action is returned.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


_N_JOINTS = 7
_NOMINAL_Q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.785])
_LINK_RADII = np.array([0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04])
_WORKSPACE_LOW = np.array([0.20, -0.50, 0.05])
_WORKSPACE_HIGH = np.array([0.80, 0.50, 0.75])


@dataclass(frozen=True)
class StarWorldsConfig:
    """Controller settings in SI units except for joint-angle parameters."""

    ee_speed: float = 0.30
    obstacle_inflation: float = 0.05
    safety_radius: float = 0.01
    modulation_eta: float = 1.0
    weight_power: float = 2.0
    gamma_threshold: float = 1.6
    link_repulsion_gain: float = 0.30
    posture_gain: float = 0.08
    correction_ema_alpha: float = 0.10
    max_nullspace_step: float = 0.07
    ik_damping: float = 0.05
    ik_iterations: int = 12
    max_ik_step: float = 0.08
    collision_margin: float = 0.002
    collision_substeps: int = 4


@dataclass(frozen=True)
class _Proxy:
    """Convex-hull proxy for one connected set of inflated spheres."""

    centers: np.ndarray
    radii: np.ndarray
    kernel_center: np.ndarray

    def radius(self, point: np.ndarray) -> float:
        """Source controller's direction-dependent support radius."""
        direction = np.asarray(point, dtype=float) - self.kernel_center
        norm = float(np.linalg.norm(direction))
        if norm < 1e-10:
            return float(np.max(np.linalg.norm(self.centers - self.kernel_center, axis=1) + self.radii))
        direction /= norm
        return float(np.max((self.centers - self.kernel_center) @ direction + self.radii))

    def gamma(self, point: np.ndarray, extra_radius: float = 0.0) -> float:
        radius = max(self.radius(point) + extra_radius, 1e-6)
        return max(float(np.linalg.norm(point - self.kernel_center)) / radius, 1e-3)


def _cluster_proxies(
    centers: np.ndarray, radii: np.ndarray, start: np.ndarray, goal: np.ndarray
) -> list[_Proxy]:
    """Join intersecting inflated spheres and choose off-path kernel points.

    For the sphere-only tasks, each connected component is represented by the
    convex hull of its spheres.  The kernel lies inside that convex proxy;
    moving it slightly away from the direct start-to-goal line selects a
    consistent side for the modulation field, as in the source 3D algorithm.
    """
    parent = list(range(len(centers)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if np.linalg.norm(centers[i] - centers[j]) <= radii[i] + radii[j]:
                parent[root(j)] = root(i)

    groups: dict[int, list[int]] = {}
    for index in range(len(centers)):
        groups.setdefault(root(index), []).append(index)

    path = goal - start
    path_norm = float(np.linalg.norm(path))
    path_dir = path / path_norm if path_norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    proxies = []
    for indices in groups.values():
        group_centers = centers[indices]
        group_radii = radii[indices]
        centroid = group_centers.mean(axis=0)
        # Select a side away from the start-goal line.  Keep the nudge inside
        # every sphere's minimum inflated radius, hence inside the proxy.
        from_line = centroid - (start + np.dot(centroid - start, path_dir) * path_dir)
        from_line -= np.dot(from_line, path_dir) * path_dir
        lateral_norm = float(np.linalg.norm(from_line))
        if lateral_norm < 1e-8:
            from_line = np.cross(path_dir, np.array([0.0, 0.0, 1.0]))
            lateral_norm = float(np.linalg.norm(from_line))
            if lateral_norm < 1e-8:
                from_line = np.cross(path_dir, np.array([0.0, 1.0, 0.0]))
                lateral_norm = float(np.linalg.norm(from_line))
        nudge = 0.25 * float(np.min(group_radii)) * from_line / lateral_norm
        proxies.append(_Proxy(group_centers, group_radii, centroid + nudge))
    return proxies


class StarWorldsPolicy:
    """Return a safe normalized seven-joint delta for a Panda reach task.

    Call ``reset()`` after ``env.reset()`` and then call the policy with each
    Gymnasium observation.  ``status`` describes the last action decision.
    The planner has no learned state or checkpoint.
    """

    def __init__(self, env, cfg: StarWorldsConfig | None = None):
        from learning.robots.panda import FrankaPandaEnv

        self.env = env.unwrapped
        if (
            not isinstance(self.env, FrankaPandaEnv)
            or self.env.task_name not in {"reach_obstacles", "reach_obstacles_2"}
            or self.env.action_space.shape != (_N_JOINTS,)
        ):
            raise ValueError("Star-Worlds requires the Panda reach_obstacles task.")
        self.cfg = cfg or StarWorldsConfig()
        self.model = self.env.model
        self.data = self.env.data
        self._scratch = mujoco.MjData(self.model)
        self._hand_id = self.model.body("hand").id
        self._link_ids = [self.model.body(f"link{i}").id for i in range(1, 8)]
        self._link_ids.append(self._hand_id)
        self._joint_lo = self.model.jnt_range[:_N_JOINTS, 0].copy()
        self._joint_hi = self.model.jnt_range[:_N_JOINTS, 1].copy()
        self._obstacle_ids = tuple(
            i for i in range(self.model.ngeom)
            if self.model.geom(i).name.startswith("reach_obstacle_")
        )
        self._robot_geom_ids = tuple(
            i for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] > 0
            and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
        )
        self._obstacle_centers = np.empty((0, 3))
        self._obstacle_radii = np.empty(0)
        self.proxies: list[_Proxy] = []
        self._smooth_correction = np.zeros(_N_JOINTS)
        self.status = "ready"
        self.reset()

    def reset(self) -> None:
        """Discard controller history and update the obstacle scene."""
        self._smooth_correction[:] = 0.0
        mujoco.mj_copyData(self._scratch, self.model, self.data)
        # The source demo moves the arm to a known start before reactive
        # control.  This MuJoCo task's keyframe intersects the centre sphere
        # in reach_obstacles, so use the collision-free nominal posture for
        # both obstacle scenes to start each episode consistently.
        if self._collision_gap(_NOMINAL_Q) >= self.cfg.collision_margin:
            self.data.qpos[:_N_JOINTS] = _NOMINAL_Q
            self.data.qvel[:_N_JOINTS] = 0.0
            self.data.ctrl[:_N_JOINTS] = _NOMINAL_Q
            mujoco.mj_forward(self.model, self.data)
            self.env._ee_traj = []
        self._update_obstacles(force=True)
        self.status = "ready"

    def _update_obstacles(self, *, force: bool = False, goal: np.ndarray | None = None) -> None:
        centers = np.array([self.data.geom_xpos[i] for i in self._obstacle_ids], dtype=float)
        radii = np.array([self.model.geom_size[i, 0] for i in self._obstacle_ids], dtype=float)
        if centers.size == 0:
            centers = centers.reshape(0, 3)
        if not force and np.array_equal(centers, self._obstacle_centers):
            return
        self._obstacle_centers = centers
        self._obstacle_radii = radii
        ee = self.env.get_ee_pos()
        target = np.asarray(self.env.goal if goal is None else goal, dtype=float)
        self.proxies = _cluster_proxies(
            centers, radii + self.cfg.obstacle_inflation, ee, target
        )

    def _modulated_velocity(self, ee: np.ndarray, goal: np.ndarray) -> np.ndarray:
        to_goal = goal - ee
        distance = float(np.linalg.norm(to_goal))
        if distance < self.env.task.distance_threshold:
            return np.zeros(3)
        direction = to_goal / distance
        if not self.proxies:
            return self.cfg.ee_speed * direction

        gammas = np.array([
            max(proxy.gamma(ee, self.cfg.safety_radius), 1.001)
            for proxy in self.proxies
        ])
        weights = gammas ** (-self.cfg.weight_power)
        weights /= weights.sum()
        modulation = np.zeros((3, 3))
        for weight, gamma, proxy in zip(weights, gammas, self.proxies):
            normal = ee - proxy.kernel_center
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm < 1e-10:
                modulation += weight * np.eye(3)
                continue
            normal /= normal_norm
            lambda_n = max(0.0, 1.0 - self.cfg.modulation_eta / gamma)
            lambda_t = 1.0 + self.cfg.modulation_eta / gamma
            modulation += weight * (
                lambda_t * np.eye(3)
                + (lambda_n - lambda_t) * np.outer(normal, normal)
            )
        velocity = modulation @ direction
        norm = float(np.linalg.norm(velocity))
        if norm < 1e-10:
            velocity = np.cross(direction, np.array([0.0, 0.0, 1.0]))
            norm = float(np.linalg.norm(velocity))
        return self.cfg.ee_speed * velocity / norm if norm > 1e-10 else np.zeros(3)

    def _set_pose(self, q: np.ndarray) -> None:
        self._scratch.qpos[:_N_JOINTS] = q
        self._scratch.qvel[:_N_JOINTS] = 0.0
        mujoco.mj_forward(self.model, self._scratch)

    def _point_jacobian(self, point: np.ndarray, body_id: int) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros_like(jacp)
        mujoco.mj_jac(self.model, self._scratch, jacp, jacr, point, body_id)
        return jacp[:, :_N_JOINTS]

    def _position_ik(self, target: np.ndarray, q_seed: np.ndarray) -> tuple[np.ndarray, float]:
        """Damped position-only IK, seeded at the current arm configuration."""
        q = q_seed.copy()
        for _ in range(self.cfg.ik_iterations):
            self._set_pose(q)
            point = self._scratch.body(self._hand_id).xpos.copy()
            error = target - point
            if np.linalg.norm(error) < 1e-4:
                break
            jacobian = self._point_jacobian(point, self._hand_id)
            damping_sq = self.cfg.ik_damping ** 2
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping_sq * np.eye(3), error
            )
            norm = float(np.linalg.norm(delta))
            if norm > self.cfg.max_ik_step:
                delta *= self.cfg.max_ik_step / norm
            q = np.clip(q + delta, self._joint_lo, self._joint_hi)
        self._set_pose(q)
        residual = float(np.linalg.norm(target - self._scratch.body(self._hand_id).xpos))
        return q, residual

    def _link_correction(self, q: np.ndarray) -> np.ndarray:
        """Source APF on link samples plus a nominal-posture preference."""
        correction = np.zeros(_N_JOINTS)
        self._set_pose(q)
        origins = np.array([self._scratch.body(i).xpos.copy() for i in self._link_ids])
        points: list[tuple[np.ndarray, int, float]] = []
        for index, body_id in enumerate(self._link_ids):
            points.append((origins[index], body_id, _LINK_RADII[index]))
            if index + 1 < len(origins):
                points.append((0.5 * (origins[index] + origins[index + 1]), body_id, _LINK_RADII[index]))
        for point, body_id, link_radius in points:
            for proxy in self.proxies:
                gamma = proxy.gamma(point, link_radius)
                if gamma >= self.cfg.gamma_threshold:
                    continue
                direction = point - proxy.kernel_center
                norm = float(np.linalg.norm(direction))
                if norm < 1e-8:
                    continue
                rho = max(gamma - 1.0, 1e-4)
                threshold = max(self.cfg.gamma_threshold - 1.0, 1e-6)
                weight = np.clip(threshold / rho - 1.0, 0.0, 5.0)
                jacobian = self._point_jacobian(point, body_id)
                correction += self.cfg.link_repulsion_gain * weight * (
                    jacobian.T @ (direction / norm)
                )
        posture_error = _NOMINAL_Q - q
        if np.linalg.norm(posture_error) > 0.05:
            correction += self.cfg.posture_gain * posture_error
        alpha = self.cfg.correction_ema_alpha
        self._smooth_correction = alpha * correction + (1.0 - alpha) * self._smooth_correction
        return self._smooth_correction

    def _collision_gap(self, q: np.ndarray) -> float:
        """Minimum exact MuJoCo robot-to-obstacle geom distance at a pose."""
        self._set_pose(q)
        if not self._obstacle_ids:
            return float("inf")
        endpoints = np.zeros(6)
        return min(
            mujoco.mj_geomDistance(self.model, self._scratch, robot, obstacle, 0.25, endpoints)
            for robot in self._robot_geom_ids
            for obstacle in self._obstacle_ids
        )

    def _safe_action(self, q: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Backtrack a proposed action until the interpolated arm stays clear."""
        current_gap = self._collision_gap(q)
        if np.linalg.norm(action) < 1e-10:
            self.status = "stalled"
            return np.zeros(_N_JOINTS, dtype=np.float32)
        for scale in (1.0, 0.5, 0.25, 0.125):
            scaled = action * scale
            gaps = [
                self._collision_gap(q + fraction * self.env.joint_action_scale * scaled)
                for fraction in np.linspace(1.0 / self.cfg.collision_substeps, 1.0, self.cfg.collision_substeps)
            ]
            # An already colliding initial pose may still move out of contact.
            if all(gap >= self.cfg.collision_margin for gap in gaps) or (
                current_gap < self.cfg.collision_margin and min(gaps) > current_gap + 1e-5
            ):
                self.status = "moving" if scale == 1.0 else "collision_backtrack"
                return scaled.astype(np.float32)
        self.status = "collision_blocked"
        return np.zeros(_N_JOINTS, dtype=np.float32)

    def __call__(self, observation: dict) -> np.ndarray:
        """Compute one normalized joint-delta action from an XYZ goal."""
        q = self.data.qpos[:_N_JOINTS].copy()
        ee = self.env.get_ee_pos()
        goal = np.asarray(observation["desired_goal"], dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Star-Worlds requires a finite XYZ desired_goal.")
        if self.env.task.is_success(ee, goal):
            self.status = "goal_reached"
            return np.zeros(_N_JOINTS, dtype=np.float32)
        self._update_obstacles(goal=goal)
        if self._obstacle_centers.size and np.any(
            np.linalg.norm(self._obstacle_centers - goal, axis=1)
            < self._obstacle_radii + self.cfg.collision_margin
        ):
            self.status = "goal_in_obstacle"
            return np.zeros(_N_JOINTS, dtype=np.float32)

        velocity = self._modulated_velocity(ee, goal)
        target = np.clip(ee + velocity * self.env.dt, _WORKSPACE_LOW, _WORKSPACE_HIGH)
        mujoco.mj_copyData(self._scratch, self.model, self.data)
        q_ik, residual = self._position_ik(target, q)
        if not np.isfinite(q_ik).all() or not np.isfinite(residual) or residual > 0.02:
            self.status = "ik_failed"
            return np.zeros(_N_JOINTS, dtype=np.float32)

        correction = self._link_correction(q)
        self._set_pose(q)
        jacobian = self._point_jacobian(self._scratch.body(self._hand_id).xpos, self._hand_id)
        null_space = np.eye(_N_JOINTS) - np.linalg.pinv(jacobian) @ jacobian
        null_step = null_space @ correction
        norm = float(np.linalg.norm(null_step))
        if norm > self.cfg.max_nullspace_step:
            null_step *= self.cfg.max_nullspace_step / norm
        q_cmd = np.clip(q_ik + null_step, self._joint_lo, self._joint_hi)
        action = np.clip((q_cmd - q) / self.env.joint_action_scale, -1.0, 1.0)
        return self._safe_action(q, action)
