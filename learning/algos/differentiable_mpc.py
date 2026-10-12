"""Point-goal, collision-constrained MPC for the Franka obstacle tasks.

The planning model follows the acceleration-controlled, double-integrator
formulation in GiantMGG/robot-optimal-trajectory-planning. Its original UR5e
kinematics and collision shapes are replaced with symbolic kinematics and
capsule approximations derived from muarm's Panda MuJoCo model. The goal is an
XYZ point, so no goal IK or orientation target is needed. Only the first joint
position target is issued before the optimization is solved again.
"""
from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import mujoco
import numpy as np


_NQ = 7
# Franka joint velocity limits (rad/s). The MJCF does not specify these.
_VELOCITY_LIMITS = np.array([2.175] * 4 + [2.61] * 3)
_START_Q = np.array([-0.186, -0.133, -0.363, -2.436, -0.340, 1.426, -0.930])


@dataclass(frozen=True)
class DifferentiableMPCConfig:
    # Match the official repository's default cost configuration.  The Panda
    # joint limits and geometry safeguards below remain MuJoCo-specific.
    horizon: int = 8
    model_step_s: float = 0.2
    acceleration_limit: float = 4.0 * np.pi  # source UR5e bound, rad/s^2
    collision_margin: float = 0.005  # metres beyond fitted link capsules
    position_weight: float = 100.0
    terminal_position_weight: float = 100.0
    velocity_weight: float = 0.1
    acceleration_weight: float = 0.02
    slack_weight: float = 50_000.0
    max_solver_iterations: int = 60
    max_solver_seconds: float = 0.6
    swept_checks: int = 5
    execution_clearance: float = 0.001


def _transform(pos: np.ndarray, quat: np.ndarray) -> ca.SX:
    """A fixed MuJoCo body/geom transform in CasADi coordinates."""
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat, dtype=np.float64))
    result = ca.SX.eye(4)
    result[:3, :3] = ca.DM(rotation.reshape(3, 3))
    result[:3, 3] = ca.DM(pos)
    return result


def _joint_transform(axis: np.ndarray, anchor: np.ndarray, angle: ca.SX) -> ca.SX:
    """Rotation about the joint's body-local axis and anchor."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    skew = ca.DM([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    rotation = ca.SX.eye(3) + ca.sin(angle) * skew + (1 - ca.cos(angle)) * (skew @ skew)
    result = ca.SX.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = ca.DM(anchor) - rotation @ ca.DM(anchor)
    return result


def _translation(offset: np.ndarray) -> ca.SX:
    result = ca.SX.eye(4)
    result[:3, 3] = ca.DM(offset)
    return result


class _PandaGeometry:
    """Symbolic FK and obstacle capsules built from the active MJCF geometry."""

    def __init__(self, model: mujoco.MjModel):
        q = ca.SX.sym("q", _NQ)
        transforms: dict[int, ca.SX] = {0: ca.SX.eye(4)}
        for body_id in range(1, model.nbody):
            parent = int(model.body_parentid[body_id])
            fixed = _transform(model.body_pos[body_id], model.body_quat[body_id])
            transform = transforms[parent] @ fixed
            for index in range(int(model.body_jntnum[body_id])):
                joint_id = int(model.body_jntadr[body_id]) + index
                if joint_id < _NQ:
                    transform = transform @ _joint_transform(
                        model.jnt_axis[joint_id], model.jnt_pos[joint_id], q[joint_id]
                    )
            transforms[body_id] = transform

        hand_id = model.body("hand").id
        ee = transforms[hand_id][:3, 3]
        self.ee_function = ca.Function("panda_mpc_ee", [q], [ee]).expand()

        # Each collision mesh is represented by an axial capsule whose size
        # follows the MuJoCo compiler's mesh bounding box. Ignore the fixed
        # base and the actuated fingers; the arm and hand are covered.
        capsules: list[tuple[ca.Function, float, int]] = []
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            if body_id < model.body("link1").id or body_id > hand_id:
                continue
            if int(model.geom_group[geom_id]) != 3:
                continue
            size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
            axis = int(np.argmax(size))
            radius = float(np.max(np.delete(size, axis)))
            half_segment = max(float(size[axis]) - radius, 0.0)
            local_a = np.asarray(model.geom_pos[geom_id], dtype=np.float64).copy()
            local_b = local_a.copy()
            direction = np.zeros(3)
            direction[axis] = half_segment
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
            offset = rotation.reshape(3, 3) @ direction
            local_a -= offset
            local_b += offset
            transform = transforms[body_id]
            world_a = transform[:3, :3] @ ca.DM(local_a) + transform[:3, 3]
            world_b = transform[:3, :3] @ ca.DM(local_b) + transform[:3, 3]
            capsule = ca.Function(
                f"panda_mpc_capsule_{geom_id}", [q], [ca.vertcat(world_a, world_b)]
            ).expand()
            capsules.append((capsule, radius, body_id))

        # Obstacle tasks command the gripper fully open. Its collision meshes
        # and fingertip pads extend beyond the hand mesh, so model each open
        # finger separately. The pad boxes are enclosed by one small sphere.
        for name in ("left_finger", "right_finger"):
            body_id = model.body(name).id
            joint_id = int(model.body_jntadr[body_id])
            finger_transform = transforms[body_id] @ _translation(
                model.jnt_axis[joint_id] * model.jnt_range[joint_id, 1]
            )
            mesh_id = next(
                i for i in range(int(model.body_geomadr[body_id]),
                                 int(model.body_geomadr[body_id] + model.body_geomnum[body_id]))
                if model.geom_group[i] == 3 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
            )
            mesh_center = model.geom_pos[mesh_id]
            mesh_size = model.geom_size[mesh_id]
            mesh_radius = float(max(mesh_size[0], mesh_size[1]))
            for suffix, local_a, local_b, radius in (
                ("mesh", mesh_center + [0, 0, -mesh_size[2]],
                 mesh_center + [0, 0, mesh_size[2]], mesh_radius),
                ("pad", np.array([0.0, 0.004, 0.046]),
                 np.array([0.0, 0.004, 0.046]), 0.014),
            ):
                a = finger_transform[:3, :3] @ ca.DM(local_a) + finger_transform[:3, 3]
                b = finger_transform[:3, :3] @ ca.DM(local_b) + finger_transform[:3, 3]
                capsule = ca.Function(
                    f"panda_mpc_{name}_{suffix}", [q], [ca.vertcat(a, b)]
                ).expand()
                capsules.append((capsule, radius, body_id))
        if not capsules:
            raise RuntimeError("The Panda model has no arm collision geometry.")
        self.capsules = capsules
        first_capsule_by_body = {
            body_id: index for index, (_, _, body_id) in reversed(list(enumerate(capsules)))
        }
        self.self_collision_pairs = [
            (first_capsule_by_body[model.body(first).id],
             first_capsule_by_body[model.body(second).id])
            for first, second in (
                ("link1", "link6"), ("link1", "hand"),
                ("link2", "link6"), ("link2", "hand"),
                ("link3", "hand"), ("link4", "hand"),
                ("link5", "hand"),
                ("link1", "left_finger"), ("link1", "right_finger"),
                ("link2", "left_finger"), ("link2", "right_finger"),
                ("link3", "left_finger"), ("link3", "right_finger"),
            )
        ]


def _capsule_sphere_clearance(
    start: ca.SX, end: ca.SX, radius: float, center: np.ndarray,
    obstacle_radius: float, margin: float,
) -> ca.SX:
    """Differentiable squared capsule-to-sphere clearance expression."""
    segment = end - start
    to_center = ca.DM(center) - start
    projection = ca.dot(to_center, segment) / (ca.sumsqr(segment) + 1e-9)
    # Smooth approximation of clip(projection, 0, 1); unlike fmin/fmax this
    # gives IPOPT a continuous derivative at the capsule endpoints.
    eps = 1e-4
    positive = 0.5 * (projection + ca.sqrt(projection * projection + eps * eps))
    excess = positive - 1.0
    bounded = positive - 0.5 * (excess + ca.sqrt(excess * excess + eps * eps))
    closest = start + bounded * segment
    safe_radius = radius + obstacle_radius + margin
    return ca.sumsqr(closest - ca.DM(center)) - safe_radius * safe_radius


def _capsule_capsule_clearances(
    a: ca.MX, b: ca.MX, c: ca.MX, d: ca.MX, safe_radius: float,
) -> tuple[ca.MX, ...]:
    """Five closest-point candidates for two finite line segments."""
    u, v, w = b - a, d - c, a - c
    uu, uv, vv = ca.dot(u, u), ca.dot(u, v), ca.dot(v, v)
    uw, vw = ca.dot(u, w), ca.dot(v, w)
    denominator = uu * vv - uv * uv + 1e-10
    s = ca.fmin(1, ca.fmax(0, (uv * vw - vv * uw) / denominator))
    t = ca.fmin(1, ca.fmax(0, (uu * vw - uv * uw) / denominator))
    pairs = [(a + s * u, c + t * v)]
    for endpoint in (a, b):
        t_edge = ca.fmin(1, ca.fmax(0, ca.dot(endpoint - c, v) / (vv + 1e-10)))
        pairs.append((endpoint, c + t_edge * v))
    for endpoint in (c, d):
        s_edge = ca.fmin(1, ca.fmax(0, ca.dot(endpoint - a, u) / (uu + 1e-10)))
        pairs.append((a + s_edge * u, endpoint))
    return tuple(ca.sumsqr(p - r) - safe_radius * safe_radius for p, r in pairs)


class DifferentiableMPCPolicy:
    """Receding-horizon Panda point-reaching controller with collision checks.

    ``__call__`` returns the normalized seven-joint delta action expected by
    ``FrankaPandaEnv.step``. ``status`` records solver or safety failures.
    """

    def __init__(self, env, cfg: DifferentiableMPCConfig | None = None):
        from learning.robots.panda import FrankaPandaEnv

        self.env = env.unwrapped
        if (
            not isinstance(self.env, FrankaPandaEnv)
            or self.env.task_name not in {"reach_obstacles", "reach_obstacles_2"}
            or self.env.action_space.shape != (_NQ,)
        ):
            raise ValueError("Differentiable MPC requires Panda reach_obstacles or reach_obstacles_2.")
        self.cfg = cfg or DifferentiableMPCConfig()
        if self.cfg.horizon < 2 or self.cfg.swept_checks < 2 or self.cfg.model_step_s <= 0:
            raise ValueError("MPC needs horizon >= 2, swept_checks >= 2, and model_step_s > 0.")
        if not np.isclose(self.env.dt, self.cfg.model_step_s):
            raise ValueError(
                "MPC model_step_s must equal the environment control step; "
                "use --n-substeps 100 for the reference 0.2 s step."
            )
        self.dt = self.cfg.model_step_s
        self.model = self.env.model
        self.geometry = _PandaGeometry(self.model)
        self.joint_lower = self.model.jnt_range[:_NQ, 0].copy()
        self.joint_upper = self.model.jnt_range[:_NQ, 1].copy()
        self.obstacle_ids = tuple(sorted(self.env.task._obstacle_geom_ids))
        if not self.obstacle_ids:
            self.obstacle_ids = tuple(
                self.model.geom(f"reach_obstacle_{i}").id
                for i in range(len(self.env.task.obstacle_centers))
            )
        self.obstacle_centers = np.stack([self.model.geom_pos[i].copy() for i in self.obstacle_ids])
        self.obstacle_radii = np.array([self.model.geom_size[i, 0] for i in self.obstacle_ids])
        self.robot_geom_ids = tuple(
            i for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] > 0
            and self.model.geom_bodyid[i] <= self.model.body("right_finger").id
            and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
        )
        self.floor_id = self.model.geom("floor").id
        self.check_data = mujoco.MjData(self.model)
        self._build_solver()
        self.reset()

    def _build_solver(self) -> None:
        cfg = self.cfg
        horizon = cfg.horizon
        dt = self.dt
        opti = ca.Opti()
        x = opti.variable(2 * _NQ, horizon + 1)
        u = opti.variable(_NQ, horizon)
        slack = opti.variable(horizon)
        x0 = opti.parameter(2 * _NQ)
        goal = opti.parameter(3)
        opti.subject_to(x[:, 0] == x0)
        opti.subject_to(slack >= 0)

        objective = 0
        for step in range(horizon):
            q, velocity = x[:_NQ, step], x[_NQ:, step]
            q_next, velocity_next = x[:_NQ, step + 1], x[_NQ:, step + 1]
            acceleration = u[:, step]
            opti.subject_to(q_next == q + dt * velocity + 0.5 * dt * dt * acceleration)
            opti.subject_to(velocity_next == velocity + dt * acceleration)
            opti.subject_to(opti.bounded(self.joint_lower, q_next, self.joint_upper))
            opti.subject_to(opti.bounded(-_VELOCITY_LIMITS, velocity_next, _VELOCITY_LIMITS))
            opti.subject_to(opti.bounded(-cfg.acceleration_limit, acceleration, cfg.acceleration_limit))

            endpoints_by_capsule = [capsule(q_next) for capsule, _, _ in self.geometry.capsules]
            for endpoints, (_, radius, body_id) in zip(endpoints_by_capsule, self.geometry.capsules):
                start, end = endpoints[:3], endpoints[3:]
                for center, obstacle_radius in zip(self.obstacle_centers, self.obstacle_radii):
                    clearance = _capsule_sphere_clearance(
                        start, end, radius, center, obstacle_radius, cfg.collision_margin,
                    )
                    opti.subject_to(clearance + slack[step] >= 0)
                if body_id >= self.model.body("link3").id:
                    # The scene floor is the half-space z >= 0.
                    opti.subject_to(start[2] + slack[step] >= radius)
                    opti.subject_to(end[2] + slack[step] >= radius)

            for i, j in self.geometry.self_collision_pairs:
                radius_a = self.geometry.capsules[i][1]
                radius_b = self.geometry.capsules[j][1]
                a, b = endpoints_by_capsule[i][:3], endpoints_by_capsule[i][3:]
                c, d = endpoints_by_capsule[j][:3], endpoints_by_capsule[j][3:]
                for clearance in _capsule_capsule_clearances(
                    a, b, c, d, radius_a + radius_b + cfg.collision_margin,
                ):
                    opti.subject_to(clearance + slack[step] >= 0)

            error = self.geometry.ee_function(q_next) - goal
            position_weight = cfg.terminal_position_weight if step == horizon - 1 else cfg.position_weight
            objective += (
                position_weight * ca.sumsqr(error)
                + cfg.velocity_weight * ca.sumsqr(velocity_next)
                + cfg.acceleration_weight * ca.sumsqr(acceleration)
                + cfg.slack_weight * slack[step]
            )

        scale = self.env.joint_action_scale
        opti.subject_to(opti.bounded(-scale, x[:_NQ, 1] - x0[:_NQ], scale))
        opti.minimize(objective)
        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {
                "print_level": 0,
                "max_iter": cfg.max_solver_iterations,
                "max_cpu_time": cfg.max_solver_seconds,
                "tol": 1e-3,
                "acceptable_tol": 1e-3,
                "sb": "yes",
            },
        )
        self.opti = opti
        self.x, self.u, self.slack = x, u, slack
        self.x0, self.goal = x0, goal

    def reset(self) -> None:
        # The task keyframe's open fingertip has only ~2 mm of clearance from
        # its center obstacle; its uncommanded servo drift creates contact.
        # Begin this planner's episodes from a checked, open-gripper posture.
        self.env.data.qpos[:_NQ] = _START_Q
        self.env.data.qpos[7:9] = self.model.jnt_range[7:9, 1]
        self.env.data.qvel[:] = 0.0
        self.env.data.ctrl[:_NQ] = _START_Q
        self.env.data.ctrl[7] = 255.0
        mujoco.mj_forward(self.model, self.env.data)
        if not self._action_safe(np.zeros(_NQ)):
            raise RuntimeError("The MPC start posture is not collision-free in this scene.")
        self.status = "ready"
        self._warm_x: np.ndarray | None = None
        self._warm_u: np.ndarray | None = None

    def _swept_collision_free(self, current: np.ndarray, candidate: np.ndarray) -> bool:
        mujoco.mj_copyData(self.check_data, self.model, self.env.data)
        endpoints = np.zeros(6, dtype=np.float64)
        for fraction in np.linspace(0.0, 1.0, self.cfg.swept_checks)[1:]:
            self.check_data.qpos[:_NQ] = current + fraction * (candidate - current)
            self.check_data.qvel[:_NQ] = 0.0
            mujoco.mj_forward(self.model, self.check_data)
            for robot in self.robot_geom_ids:
                for obstacle in self.obstacle_ids:
                    if mujoco.mj_geomDistance(
                        self.model, self.check_data, robot, obstacle, 0.0, endpoints
                    ) < 0.0:
                        return False
        return True

    def _action_safe(self, action: np.ndarray) -> bool:
        """Check one real MuJoCo servo interval, including finger motion."""
        mujoco.mj_copyData(self.check_data, self.model, self.env.data)
        q_target = np.clip(
            self.check_data.qpos[:_NQ] + self.env.joint_action_scale * action,
            self.joint_lower, self.joint_upper,
        )
        self.check_data.ctrl[:_NQ] = q_target
        self.check_data.ctrl[_NQ] = 255.0
        endpoints = np.zeros(6, dtype=np.float64)
        for substep in range(self.env.n_substeps):
            mujoco.mj_step(self.model, self.check_data)
            if any(
                contact.dist <= 0.0 and (
                    contact.geom1 in self.obstacle_ids
                    or contact.geom2 in self.obstacle_ids
                    or (
                        contact.geom1 in self.robot_geom_ids
                        and contact.geom2 in self.robot_geom_ids
                        and abs(
                            int(self.model.geom_bodyid[contact.geom1])
                            - int(self.model.geom_bodyid[contact.geom2])
                        ) >= 3
                    )
                    or (
                        contact.geom1 == self.floor_id
                        and self.model.geom_bodyid[contact.geom2] >= self.model.body("link1").id
                    )
                    or (
                        contact.geom2 == self.floor_id
                        and self.model.geom_bodyid[contact.geom1] >= self.model.body("link1").id
                    )
                )
                for contact in self.check_data.contact
            ):
                return False
            if (substep + 1) % max(1, self.env.n_substeps // 4) == 0:
                for robot in self.robot_geom_ids:
                    for obstacle in self.obstacle_ids:
                        if mujoco.mj_geomDistance(
                            self.model, self.check_data, robot, obstacle,
                            self.cfg.execution_clearance, endpoints,
                        ) < self.cfg.execution_clearance:
                            return False
        return True

    def __call__(self, obs: dict) -> np.ndarray:
        q = np.asarray(self.env.data.qpos[:_NQ], dtype=np.float64)
        velocity = np.clip(self.env.data.qvel[:_NQ], -_VELOCITY_LIMITS, _VELOCITY_LIMITS)
        target = np.asarray(obs["desired_goal"], dtype=np.float64).reshape(3)
        zero = np.zeros(_NQ, dtype=np.float32)
        if not np.isfinite(q).all() or not np.isfinite(velocity).all() or not np.isfinite(target).all():
            self.status = "invalid_state"
            return zero
        self.opti.set_value(self.x0, np.concatenate([q, velocity]))
        self.opti.set_value(self.goal, target)
        if self._warm_x is not None:
            self.opti.set_initial(self.x, self._warm_x)
            self.opti.set_initial(self.u, self._warm_u)
        else:
            self.opti.set_initial(self.x, np.tile(np.concatenate([q, velocity])[:, None], (1, self.cfg.horizon + 1)))
            self.opti.set_initial(self.u, 0)
        self.opti.set_initial(self.slack, 0)
        try:
            solution = self.opti.solve()
        except RuntimeError:
            self.status = "solver_failed"
            return zero
        planned = np.asarray(solution.value(self.x), dtype=np.float64)
        controls = np.asarray(solution.value(self.u), dtype=np.float64)
        candidate = planned[:_NQ, 1]
        if not np.isfinite(candidate).all():
            self.status = "invalid_solution"
            return zero
        proposed_action = np.clip(
            (candidate - q) / self.env.joint_action_scale, -1.0, 1.0
        )
        if not self._swept_collision_free(q, candidate):
            self.status = "collision_rejected"
            return zero
        if not self._action_safe(proposed_action):
            brake = np.clip(-0.1 * velocity / self.env.joint_action_scale, -1.0, 1.0)
            for alternative in (0.5 * proposed_action, 0.25 * proposed_action, brake, zero):
                if self._action_safe(alternative):
                    self.status = "collision_avoided"
                    return np.asarray(alternative, dtype=np.float32)
            self.status = "collision_unavoidable"
            return zero
        self._warm_x = np.column_stack([planned[:, 1:], planned[:, -1]])
        self._warm_u = np.column_stack([controls[:, 1:], controls[:, -1]])
        self.status = "ready"
        return proposed_action.astype(np.float32)
