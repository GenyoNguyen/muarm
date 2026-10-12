"""Exact MuJoCo Panda–obstacle gap oracle with isolated simulation state."""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import SphereScene


class PandaCollisionOracle:
    """Evaluate seven-joint poses without touching a live environment's MjData.

    ``gap`` is the minimum signed distance in metres between a Panda collision
    geom and an active task sphere.  ``is_valid`` additionally checks the floor
    and MuJoCo-reported self contacts; neither affects the obstacle boundary
    labels.  The caller owns ``model``; scene updates deliberately change its
    world obstacle geoms, so use a separate model for a live environment.
    """

    def __init__(self, model: mujoco.MjModel, scene: SphereScene | None = None):
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_ids = tuple(model.joint(f"joint{i}").id for i in range(1, 8))
        self.qpos_indices = np.array([model.jnt_qposadr[j] for j in self.joint_ids], dtype=int)
        self.joint_limits = model.jnt_range[list(self.joint_ids)].astype(np.float64).copy()
        if not np.all(model.jnt_limited[list(self.joint_ids)]):
            raise ValueError("all seven Panda joints must have finite limits")
        self._robot_geoms = tuple(
            i for i in range(model.ngeom)
            if model.geom_bodyid[i] != 0
            and model.geom_contype[i] != 0
            and model.geom_conaffinity[i] != 0
        )
        if not self._robot_geoms:
            raise ValueError("model has no Panda collision geoms")
        link1_id = model.body("link1").id
        self._arm_geoms = tuple(i for i in self._robot_geoms if model.geom_bodyid[i] >= link1_id)
        self._floor_geom = model.geom("floor").id
        self._endpoints = np.empty(6, dtype=np.float64)
        self._obstacle_ids: tuple[int, ...] = ()
        self.scene: SphereScene
        self.set_scene(scene or SphereScene.from_model(model))
        self.data.qpos[self.qpos_indices] = np.clip(
            model.key_qpos[0, self.qpos_indices]
            if model.nkey else np.mean(self.joint_limits, axis=1),
            self.joint_limits[:, 0], self.joint_limits[:, 1]
        )
        # Both finger joints are open in the obstacle-reaching tasks.
        for name in ("finger_joint1", "finger_joint2"):
            try:
                joint = model.joint(name)
            except KeyError:
                continue
            self.data.qpos[model.jnt_qposadr[joint.id]] = min(0.04, model.jnt_range[joint.id, 1])
        mujoco.mj_forward(model, self.data)

    def set_scene(self, scene: SphereScene) -> None:
        """Install centers and radii on this oracle's model."""
        if not scene.names:
            names = tuple(f"reach_obstacle_{i}" for i in range(len(scene.spheres)))
        else:
            names = scene.names
        ids = tuple(self.model.geom(name).id for name in names)
        if len(ids) != len(scene.spheres):
            raise ValueError("scene and MuJoCo obstacle counts differ")
        # All model obstacle geoms must be represented: otherwise a masked or
        # omitted obstacle could still take part in MuJoCo contacts.
        actual = SphereScene.from_model(self.model)
        if tuple(actual.names) != names:
            raise ValueError("scene names must match the full MuJoCo obstacle set")
        for geom_id, row, active in zip(ids, scene.spheres, scene.mask):
            self.model.geom_pos[geom_id] = row[:3] if active else np.array([1e3, 1e3, 1e3])
            self.model.geom_size[geom_id, 0] = row[3] if active else 1e-6
            self.model.geom_rbound[geom_id] = row[3] if active else 1e-6
        self._obstacle_ids = tuple(i for i, active in zip(ids, scene.mask) if active)
        if not self._obstacle_ids:
            raise ValueError("at least one active obstacle is required")
        self.scene = scene
        mujoco.mj_forward(self.model, self.data)

    def in_limits(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=np.float64)
        return bool(q.shape == (7,) and np.all(np.isfinite(q)) and
                    np.all(q >= self.joint_limits[:, 0]) and
                    np.all(q <= self.joint_limits[:, 1]))

    def _set_q(self, q: np.ndarray) -> None:
        if not self.in_limits(q):
            raise ValueError("q must be a finite seven-joint vector within Panda limits")
        self.data.qpos[self.qpos_indices] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _obstacle_gap_current(self) -> float:
        # geom_rbound encloses each actual geom, so the sphere-bound distance
        # is a lower bound on the exact distance.  Sorting by this bound makes
        # the 30-sphere cross scene practical without approximating the label.
        robots = np.asarray(self._robot_geoms)
        obstacles = np.asarray(self._obstacle_ids)
        lower_bounds = (
            np.linalg.norm(
                self.data.geom_xpos[robots, None, :]
                - self.data.geom_xpos[None, obstacles, :], axis=2
            )
            - self.model.geom_rbound[robots, None]
            - self.model.geom_rbound[None, obstacles]
        )
        best = 10.0
        for flat_index in np.argsort(lower_bounds, axis=None):
            robot_index, obstacle_index = np.unravel_index(flat_index, lower_bounds.shape)
            if lower_bounds[robot_index, obstacle_index] >= best:
                break
            distance = mujoco.mj_geomDistance(
                self.model, self.data,
                int(robots[robot_index]), int(obstacles[obstacle_index]),
                10.0, self._endpoints,
            )
            best = min(best, distance)
        return float(best)

    def gap(self, q: np.ndarray) -> float:
        """Minimum signed Panda–sphere surface gap; negative means overlap."""
        self._set_q(np.asarray(q, dtype=np.float64))
        return self._obstacle_gap_current()

    def constraint_gap(self, q: np.ndarray) -> float:
        """Minimum non-obstacle safety gap (floor or reported self contact)."""
        self._set_q(np.asarray(q, dtype=np.float64))
        floor_gap = min(
            mujoco.mj_geomDistance(
                self.model, self.data, robot, self._floor_geom, 5.0, self._endpoints
            ) for robot in self._arm_geoms
        )
        robot_set = set(self._robot_geoms)
        self_contacts = [
            contact.dist for contact in self.data.contact
            if contact.geom1 in robot_set and contact.geom2 in robot_set
        ]
        return float(min([floor_gap, *self_contacts]))

    def is_valid(self, q: np.ndarray, *, margin: float = 0.0) -> bool:
        if not self.in_limits(q):
            return False
        self._set_q(np.asarray(q, dtype=np.float64))
        if self._obstacle_gap_current() <= margin:
            return False
        return self.constraint_gap(q) > margin

    def boundary_valid(self, q: np.ndarray, *, tolerance: float = 0.002) -> bool:
        """Validate a candidate obstacle-boundary configuration."""
        if not self.in_limits(q):
            return False
        gap = self.gap(q)
        return bool(np.isfinite(gap) and abs(gap) <= tolerance and
                    self.constraint_gap(q) > -tolerance)
