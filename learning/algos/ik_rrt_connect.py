"""Position IK followed by collision-checked joint-space RRT-Connect.

The external goal is an end-effector XYZ point. IK produces candidate joint
configurations; RRT-Connect plans from the measured joints to one of them.
The planner uses a separate MuJoCo data buffer, so searching does not move the
live robot. Each interpolated edge checks all Panda collision geoms against the
task obstacles. A short action along the checked path is executed each step.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import mujoco
import numpy as np


_N_ARM_JOINTS = 7


@dataclass(frozen=True)
class IKRRTConnectConfig:
    """Planner resolution and finite search budgets."""

    ik_trials: int = 1000
    ik_max_tip_distance: float = 1.5  # source simIK maxDist, metres before IK
    ik_candidates: int = 1  # the source plans to configs[1] only
    ik_max_iters: int = 160
    ik_tolerance: float = 0.008
    ik_damping: float = 0.06
    ik_step: float = 0.25
    rrt_iterations: int = 1800
    rrt_max_time_s: float = 6.0  # source simOMPL.compute maxTime
    rrt_step: float = 0.22
    goal_bias: float = 0.12
    edge_resolution: float = 0.045
    shortcut_attempts: int = 80
    waypoint_tolerance: float = 0.022
    min_tip_height: float = 0.01
    collision_margin: float = 0.0


class _Tree:
    def __init__(self, root: np.ndarray):
        self.nodes = [root.copy()]
        self.parents = [-1]

    def nearest(self, target: np.ndarray) -> int:
        nodes = np.asarray(self.nodes)
        return int(np.argmin(np.sum((nodes - target) ** 2, axis=1)))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(q.copy())
        self.parents.append(parent)
        return len(self.nodes) - 1

    def branch(self, index: int) -> list[np.ndarray]:
        result = []
        while index >= 0:
            result.append(self.nodes[index])
            index = self.parents[index]
        result.reverse()
        return result


class IKRRTConnectPolicy:
    """Training-free Panda point-reaching planner compatible with ``env.step``.

    Call ``reset()`` after each environment reset. The first policy call then
    computes a path to ``obs['desired_goal']``. ``status`` reports ``unplanned``,
    ``tracking``, ``success``, ``goal_in_obstacle``, ``ik_failed``, ``planning_failed``, or
    ``tracking_blocked``. A failed search returns a zero action.
    """

    def __init__(
        self,
        env,
        config: IKRRTConnectConfig | None = None,
        *,
        seed: int | None = None,
    ):
        if getattr(env, "task_name", None) not in {"reach_obstacles", "reach_obstacles_2"}:
            raise ValueError("IK-RRT-Connect requires a Panda obstacle-reaching task.")
        if env.action_space.shape != (_N_ARM_JOINTS,):
            raise ValueError("IK-RRT-Connect requires seven joint-delta actions.")
        self.env = env
        self.model = env.model
        self.data = mujoco.MjData(self.model)
        self._preview_data = mujoco.MjData(self.model)
        self.cfg = config or IKRRTConnectConfig()
        self.rng = np.random.default_rng(seed)
        self._hand_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, env.get_ee_body()
        )
        self._limits = self.model.jnt_range[:_N_ARM_JOINTS].copy()
        self._robot_geoms = tuple(
            i for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] != 0
            and self.model.geom_contype[i] != 0
            and self.model.geom_conaffinity[i] != 0
        )
        self._arm_geoms = frozenset(
            i for i in self._robot_geoms
            if self.model.geom_bodyid[i] >= self.model.body("link1").id
        )
        self._floor_geom = self.model.geom("floor").id
        self._distance_buffer = np.empty(6, dtype=np.float64)
        self._jac_pos = np.empty((3, self.model.nv), dtype=np.float64)
        self._jac_rot = np.empty((3, self.model.nv), dtype=np.float64)
        self._obstacle_geoms: tuple[int, ...] = ()
        self._obstacle_centers = np.empty((0, 3), dtype=np.float64)
        self._obstacle_radii = np.empty(0, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        """Discard the previous episode's goal, plan, and tracking state."""
        self.status = "unplanned"
        self.goal: np.ndarray | None = None
        self.path: np.ndarray | None = None
        self.waypoints: np.ndarray | None = None
        self.goal_configuration: np.ndarray | None = None
        self._waypoint_index = 0

    def _snapshot_world(self) -> None:
        mujoco.mj_copyData(self.data, self.model, self.env.data)
        obstacle_ids = getattr(self.env.task, "_obstacle_geom_ids", ())
        self._obstacle_geoms = tuple(sorted(obstacle_ids))
        if not self._obstacle_geoms:
            raise ValueError("The obstacle task has not been reset or has no obstacle geoms.")
        ids = list(self._obstacle_geoms)
        self._obstacle_centers = self.data.geom_xpos[ids].copy()
        self._obstacle_radii = self.model.geom_rbound[ids].copy()

    def _set_q(self, q: np.ndarray) -> None:
        self.data.qpos[:_N_ARM_JOINTS] = q
        self.data.qvel[:_N_ARM_JOINTS] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _valid(self, q: np.ndarray) -> bool:
        if not np.all(np.isfinite(q)):
            return False
        if np.any(q < self._limits[:, 0]) or np.any(q > self._limits[:, 1]):
            return False
        self._set_q(q)
        if self.data.xpos[self._hand_body, 2] < self.cfg.min_tip_height:
            return False

        margin = self.cfg.collision_margin
        for robot in self._arm_geoms:
            distance = mujoco.mj_geomDistance(
                self.model, self.data, robot, self._floor_geom,
                margin + 0.01, self._distance_buffer,
            )
            if distance <= margin:
                return False
        for robot in self._robot_geoms:
            # MuJoCo's geom_rbound encloses each mesh. This broad-phase test
            # only skips pairs that cannot touch; the narrow phase is exact.
            separation = np.linalg.norm(
                self._obstacle_centers - self.data.geom_xpos[robot], axis=1
            )
            nearby = np.flatnonzero(
                separation <= self.model.geom_rbound[robot] + self._obstacle_radii + margin
            )
            for obstacle_index in nearby:
                distance = mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    robot,
                    self._obstacle_geoms[int(obstacle_index)],
                    margin + 0.01,
                    self._distance_buffer,
                )
                if distance <= margin:
                    return False
        return True

    def _edge_valid(self, start: np.ndarray, end: np.ndarray) -> bool:
        count = max(1, int(np.ceil(np.max(np.abs(end - start)) / self.cfg.edge_resolution)))
        for index in range(1, count + 1):
            q = start + (index / count) * (end - start)
            if not self._valid(q):
                return False
        return True

    def _ik(self, goal: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
        q = np.clip(seed.copy(), self._limits[:, 0], self._limits[:, 1])
        damping_sq = self.cfg.ik_damping ** 2
        # The reach task accepts a small position tolerance. Remember a
        # collision-free near solution when an exact XYZ solution intersects
        # an obstacle; this still satisfies the task's point goal.
        near_tolerance = min(0.025, 0.75 * self.env.task.distance_threshold)
        best_q = None
        best_error = near_tolerance
        for _ in range(self.cfg.ik_max_iters):
            self._set_q(q)
            error = goal - self.data.xpos[self._hand_body]
            error_norm = np.linalg.norm(error)
            if error_norm < best_error and self._valid(q):
                best_q = q.copy()
                best_error = error_norm
            if error_norm <= self.cfg.ik_tolerance:
                return q if self._valid(q) else best_q
            mujoco.mj_jacBody(
                self.model, self.data, self._jac_pos, self._jac_rot, self._hand_body
            )
            jac = self._jac_pos[:, :_N_ARM_JOINTS]
            dq = jac.T @ np.linalg.solve(
                jac @ jac.T + damping_sq * np.eye(3), error
            )
            largest = np.max(np.abs(dq))
            if largest > self.cfg.ik_step:
                dq *= self.cfg.ik_step / largest
            q_next = np.clip(q + dq, self._limits[:, 0], self._limits[:, 1])
            if np.max(np.abs(q_next - q)) < 1e-7:
                break
            q = q_next
        return best_q

    def _goal_candidates(self, goal: np.ndarray, start: np.ndarray) -> list[np.ndarray]:
        candidates = []
        for trial in range(self.cfg.ik_trials):
            if trial == 0:
                seed = start
            elif trial % 4 == 0:
                seed = self.rng.uniform(self._limits[:, 0], self._limits[:, 1])
            else:
                spread = 0.6 if trial % 2 else 1.4
                seed = np.clip(
                    start + self.rng.normal(0.0, spread, _N_ARM_JOINTS),
                    self._limits[:, 0], self._limits[:, 1],
                )
            # simIK's maxDist filters each sampled seed by its tip-to-target
            # distance before attempting IK, not the episode's start pose.
            self._set_q(seed)
            if np.linalg.norm(self.data.xpos[self._hand_body] - goal) > self.cfg.ik_max_tip_distance:
                continue
            q = self._ik(goal, seed)
            if q is not None and all(np.linalg.norm(q - other) > 0.25 for other in candidates):
                candidates.append(q)
                if len(candidates) >= self.cfg.ik_candidates:
                    break
        candidates.sort(key=lambda q: np.linalg.norm(q - start))
        return candidates

    def _extend(self, tree: _Tree, target: np.ndarray) -> tuple[str, int | None]:
        parent = tree.nearest(target)
        near = tree.nodes[parent]
        delta = target - near
        length = np.linalg.norm(delta)
        if length < 1e-10:
            return "reached", parent
        q_new = near + min(1.0, self.cfg.rrt_step / length) * delta
        if not self._edge_valid(near, q_new):
            return "trapped", None
        index = tree.add(q_new, parent)
        return ("reached" if length <= self.cfg.rrt_step else "advanced"), index

    def _connect(self, tree: _Tree, target: np.ndarray, deadline: float) -> tuple[str, int | None]:
        while time.perf_counter() < deadline:
            status, index = self._extend(tree, target)
            if status != "advanced":
                return status, index
        return "trapped", None

    def _rrt_connect(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray] | None:
        deadline = time.perf_counter() + self.cfg.rrt_max_time_s
        if self._edge_valid(start, goal):
            return [start, goal]
        start_tree, goal_tree = _Tree(start), _Tree(goal)
        for iteration in range(self.cfg.rrt_iterations):
            if time.perf_counter() >= deadline:
                break
            start_side = iteration % 2 == 0
            first = start_tree if start_side else goal_tree
            second = goal_tree if start_side else start_tree
            if self.rng.random() < self.cfg.goal_bias:
                target = second.nodes[0]
            else:
                target = self.rng.uniform(self._limits[:, 0], self._limits[:, 1])
            status, first_index = self._extend(first, target)
            if status == "trapped":
                continue
            connected, second_index = self._connect(second, first.nodes[first_index], deadline)
            if connected == "reached":
                first_branch = first.branch(first_index)
                second_branch = second.branch(second_index)
                if start_side:
                    return first_branch + second_branch[-2::-1]
                return second_branch + first_branch[-2::-1]
        return None

    def _shortcut(self, path: list[np.ndarray]) -> list[np.ndarray]:
        if len(path) <= 2:
            return path
        for _ in range(self.cfg.shortcut_attempts):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), size=2, replace=False))
            if j - i < 2:
                continue
            if self._edge_valid(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path

    def _densify(self, path: list[np.ndarray]) -> np.ndarray:
        points = [path[0]]
        action_step = min(self.env.joint_action_scale, self.cfg.edge_resolution)
        for start, end in zip(path, path[1:]):
            count = max(1, int(np.ceil(np.max(np.abs(end - start)) / action_step)))
            points.extend(start + (i / count) * (end - start) for i in range(1, count + 1))
        return np.asarray(points)

    def _action_safe(self, q: np.ndarray, action: np.ndarray) -> bool:
        """Check the command edge and one MuJoCo control step before sending it."""
        commanded = np.clip(
            q + self.env.joint_action_scale * action,
            self._limits[:, 0], self._limits[:, 1],
        )
        if not self._edge_valid(q, commanded):
            return False
        preview = self._preview_data
        mujoco.mj_copyData(preview, self.model, self.env.data)
        preview.ctrl[:_N_ARM_JOINTS] = commanded
        if self.model.nu > _N_ARM_JOINTS:
            preview.ctrl[_N_ARM_JOINTS] = 255.0
        obstacle_ids = set(self._obstacle_geoms)
        for _ in range(self.env.n_substeps):
            mujoco.mj_step(self.model, preview)
            if any(
                contact.dist <= self.cfg.collision_margin
                and (
                    contact.geom1 in obstacle_ids or contact.geom2 in obstacle_ids
                    or (
                        contact.geom1 == self._floor_geom
                        and contact.geom2 in self._arm_geoms
                    )
                    or (
                        contact.geom2 == self._floor_geom
                        and contact.geom1 in self._arm_geoms
                    )
                )
                for contact in preview.contact
            ):
                return False
        return True

    def plan(self, goal: np.ndarray) -> bool:
        """Plan from measured joints to an XYZ target without moving the robot."""
        goal = np.asarray(goal, dtype=np.float64).reshape(3)
        self.reset()
        self.goal = goal.copy()
        self._snapshot_world()
        if np.any(
            np.linalg.norm(self._obstacle_centers - goal, axis=1)
            <= self._obstacle_radii
        ):
            self.status = "goal_in_obstacle"
            return False
        start = self.env.data.qpos[:_N_ARM_JOINTS].copy()
        if not self._valid(start):
            self.status = "planning_failed"
            return False
        if self.env.task.is_success(self.env.get_ee_pos(), goal):
            self.path = np.asarray([start])
            self.waypoints = self.path.copy()
            self.goal_configuration = start.copy()
            self.status = "success"
            return True
        candidates = self._goal_candidates(goal, start)
        if not candidates:
            self.status = "ik_failed"
            return False
        for candidate in candidates:
            path = self._rrt_connect(start, candidate)
            if path is None:
                continue
            path = self._shortcut(path)
            self.path = np.asarray(path)
            self.waypoints = self._densify(path)
            self.goal_configuration = candidate.copy()
            self._waypoint_index = 1
            self.status = "tracking"
            return True
        self.status = "planning_failed"
        return False

    def __call__(self, obs: dict) -> np.ndarray:
        """Return the next normalized seven-joint action for ``env.step``."""
        goal = np.asarray(obs["desired_goal"], dtype=np.float64).reshape(3)
        zero = np.zeros(_N_ARM_JOINTS, dtype=np.float32)
        if self.goal is None or np.linalg.norm(goal - self.goal) > 1e-6:
            if not self.plan(goal):
                return zero
        if self.status in {"goal_in_obstacle", "ik_failed", "planning_failed", "tracking_blocked"}:
            return zero
        if self.env.task.is_success(self.env.get_ee_pos(), goal):
            self.status = "success"
            return zero
        assert self.waypoints is not None
        q = self.env.data.qpos[:_N_ARM_JOINTS].copy()
        while (
            self._waypoint_index < len(self.waypoints) - 1
            and np.max(np.abs(self.waypoints[self._waypoint_index] - q))
            <= self.cfg.waypoint_tolerance
        ):
            self._waypoint_index += 1
        target = self.waypoints[self._waypoint_index]
        action = np.clip(
            (target - q) / self.env.joint_action_scale, -1.0, 1.0
        )
        if self._action_safe(q, action):
            return action.astype(np.float32)

        # Servo inertia can leave the measured joints behind a valid geometric
        # path. Test stronger motion in the current velocity direction before
        # the next simulation step; this can clear a near obstacle while still
        # checking the resulting command and simulated substeps.
        velocity_sign = np.sign(self.env.data.qvel[:_N_ARM_JOINTS])
        for fraction in (0.25, 0.5, 0.75, 1.0):
            candidate = (1.0 - fraction) * action + fraction * velocity_sign
            if self._action_safe(q, candidate):
                return candidate.astype(np.float32)
        for fraction in (0.5, 0.25, 0.0):
            candidate = fraction * action
            if self._action_safe(q, candidate):
                return candidate.astype(np.float32)

        # In a narrow passage, independent joint corrections can be needed to
        # counter coupled actuator motion. Keep the safe correction that makes
        # the best predicted progress toward the current checked waypoint.
        best_action = None
        best_error = np.inf
        for attempt in range(48):
            if attempt % 4 == 0:
                candidate = self.rng.uniform(-1.0, 1.0, _N_ARM_JOINTS)
            else:
                candidate = np.clip(
                    action + self.rng.normal(0.0, 0.5, _N_ARM_JOINTS), -1.0, 1.0
                )
            if not self._action_safe(q, candidate):
                continue
            error = np.linalg.norm(
                target - self._preview_data.qpos[:_N_ARM_JOINTS]
            )
            if error < best_error:
                best_error = error
                best_action = candidate.copy()
        if best_action is not None:
            return best_action.astype(np.float32)
        self.status = "tracking_blocked"
        return zero
