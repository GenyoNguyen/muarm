"""Franka Panda robot wrapper.

Model loading: builds the MuJoCo model programmatically from scene.xml +
task-specific body snippets via from_xml_string, avoiding relative-path
include issues when the XML is not co-located with the robot assets.

Action: 4-D Cartesian delta + gripper, or 7-D joint delta for obstacle tasks
EE body: "hand"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematic.panda_kinematics import PandaKinematics
from learning.envs.base_env import MuJocoRobotEnv
from learning.envs.tasks.reach_obstacles import OBSTACLE_CENTERS, OBSTACLE_RADIUS
from learning.envs.tasks.reach_obstacles_2 import (
    OBSTACLE_CENTERS as OBSTACLE_CENTERS_2,
    OBSTACLE_RADIUS as OBSTACLE_RADIUS_2,
)

# --- Paths ------------------------------------------------------------------
_FRANKA_DIR   = ROOT / "models" / "franka_emika_panda"
_FRANKA_SCENE = _FRANKA_DIR / "scene.xml"
_PANDA_MJCF   = _FRANKA_DIR / "panda.xml"
_MESH_DIR     = _FRANKA_DIR / "assets"

# Workspace limits [m]
_WS_LOW  = np.array([0.20, -0.50, 0.05], dtype=np.float64)
_WS_HIGH = np.array([0.80,  0.50, 0.75], dtype=np.float64)
_N_ARM_JOINTS = 7
# TODO: Joint control tasks should be self-defined, not needing this explicit definition
_JOINT_CONTROL_TASKS = {"reach_obstacles", "reach_obstacles_2"}
_REACH_OBSTACLES_START_Q = np.array(
    [-0.186, -0.133, -0.363, -2.436, -0.340, 1.426, -0.930]
)

# --- Task body snippets (injected into worldbody) ---------------------------
def _obstacle_geoms(centers, radius: float = OBSTACLE_RADIUS) -> str:
    return "\n".join(
        f'    <geom name="reach_obstacle_{i}" type="sphere" '
        f'pos="{x} {y} {z}" size="{radius}" rgba="0.25 0.35 0.85 1"/>'
        for i, (x, y, z) in enumerate(centers)
    )

_TASK_BODIES: dict[str, str] = {
    "reach": """
    <site name="target" type="sphere" size="0.02" rgba="1 0 0 0.5" pos="0.5 0 0.3"/>
""",
    "reach_obstacles": f"""
    <site name="target" type="sphere" size="0.02" rgba="1 0 0 0.5" pos="0.5 0 0.3"/>
{_obstacle_geoms(OBSTACLE_CENTERS)}
""",
    "reach_obstacles_2": f"""
    <site name="target" type="sphere" size="0.02" rgba="1 0 0 0.5" pos="0.5 0 0.3"/>
{_obstacle_geoms(OBSTACLE_CENTERS_2, OBSTACLE_RADIUS_2)}
""",
    "push": """
    <site name="target" type="sphere" size="0.02" rgba="1 0 0 0.5" pos="0.55 0 0.03"/>
    <body name="push_box" pos="0.5 0.0 0.025">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.025" rgba="0.2 0.6 1 1"
            mass="0.12" friction="0.9 0.02 0.001"/>
    </body>
""",
    "pick_place": """
    <site name="target" type="sphere" size="0.02" rgba="1 0 0 0.5" pos="0.55 0 0.25"/>
    <body name="pick_obj" pos="0.5 0.0 0.025">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0.9 0.5 0.2 1"
            mass="0.08" friction="1.0 0.02 0.001"/>
    </body>
""",
}


def _build_model(task_name: str) -> mujoco.MjModel:
    """Build MuJoCo model for panda + task bodies via XML string + asset bytes."""
    body_snippet = _TASK_BODIES.get(task_name, "")
    xml = f"""<mujoco model="panda_{task_name}">
  <include file="{_FRANKA_SCENE}"/>
  <worldbody>
    {body_snippet}
  </worldbody>
</mujoco>"""

    assets: dict[str, bytes] = {}
    if _MESH_DIR.is_dir():
        for fname in os.listdir(_MESH_DIR):
            fpath = _MESH_DIR / fname
            if fpath.is_file():
                with open(fpath, "rb") as fh:
                    assets[fname] = fh.read()

    model = mujoco.MjModel.from_xml_string(xml, assets)
    # TODO: Unify the start configurations for all reach_obstacles environments
    if task_name == "reach_obstacles":
        # The regular Panda home keyframe intersects the taller obstacle stack.
        model.key_qpos[0, :_N_ARM_JOINTS] = _REACH_OBSTACLES_START_Q
        model.key_ctrl[0, :_N_ARM_JOINTS] = _REACH_OBSTACLES_START_Q
    return model


class FrankaPandaEnv(MuJocoRobotEnv):
    """Franka Emika Panda manipulation environment.

    Action:      4-D Cartesian delta + grip, or 7-D joint delta for obstacle tasks
    Observation: Dict{"observation":(obs_dim,), "achieved_goal":(3,), "desired_goal":(3,)}
    """

    def __init__(
        self,
        task,
        task_name: str = "reach",
        model_path: str | None = None,
        render_mode: str | None = None,
        n_substeps: int = 20,
        physics_timestep: float | None = None,
        horizon: int = 200,
        action_scale: float = 0.03,
        joint_action_scale: float = 0.05,
        obs_dim: int = 32,
    ):
        self.action_scale = action_scale
        # TODO: This can be merged with action_scale
        self.joint_action_scale = joint_action_scale
        self.obs_dim = obs_dim
        self.task_name = task_name

        # TODO: [LOW] Check the kinematics mechanism
        # Pinocchio kinematics
        self._kin = PandaKinematics(ee_frame="hand")
        self._kin.build_from_mjcf(str(_PANDA_MJCF))

        # Pre-build MuJoCo model
        if model_path:
            prebuilt = mujoco.MjModel.from_xml_path(model_path)
        else:
            prebuilt = _build_model(task_name)
        if physics_timestep is not None:
            if physics_timestep <= 0:
                raise ValueError("physics_timestep must be positive.")
            prebuilt.opt.timestep = physics_timestep

        super().__init__(
            model_path=None,
            model=prebuilt,
            task=task,
            render_mode=render_mode,
            n_substeps=n_substeps,
            horizon=horizon,
        )

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    def _build_spaces(self) -> None:
        action_dim = _N_ARM_JOINTS if self.task_name in _JOINT_CONTROL_TASKS else 4
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "desired_goal":  spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })

    def _get_obs(self) -> dict:
        ee_pos   = self.get_ee_pos()
        qpos     = self.data.qpos[:9].copy()   # 7 arm + 2 finger
        qvel     = self.data.qvel[:9].copy()
        task_obs = self.task.get_obs(self.model, self.data, self)
        achieved = self.task.achieved_goal(self.model, self.data, self)

        raw     = np.concatenate([ee_pos, qpos, qvel, task_obs]).astype(np.float32)
        obs_vec = np.zeros(self.obs_dim, dtype=np.float32)
        n = min(len(raw), self.obs_dim)
        obs_vec[:n] = raw[:n]

        return {
            "observation":   obs_vec,
            "achieved_goal": achieved.astype(np.float32),
            "desired_goal":  self._goal.astype(np.float32),
        }

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        # TODO: Decision of whether to use joint actions or position actions should be
        # taken from the task itself.
        if self.task_name in _JOINT_CONTROL_TASKS:
            q = self.data.qpos[:_N_ARM_JOINTS]
            q_target = q + self.joint_action_scale * action[:_N_ARM_JOINTS]
            q_target = np.clip(
                q_target,
                self.model.jnt_range[:_N_ARM_JOINTS, 0],
                self.model.jnt_range[:_N_ARM_JOINTS, 1],
            )
            self.data.ctrl[:_N_ARM_JOINTS] = q_target
            # TODO: This sets the control of the gripper to a fixed value. We may not need this.
            if self.model.nu > _N_ARM_JOINTS:
                self.data.ctrl[_N_ARM_JOINTS] = 255.0
            return

        ee     = self.get_ee_pos()
        target = np.clip(ee + self.action_scale * action[:3], _WS_LOW, _WS_HIGH)

        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = np.diag([1.0, -1.0, -1.0])
        tf[:3, 3]  = target

        q_seed = self.data.qpos[:_N_ARM_JOINTS].copy()
        q_sol, _ = self._kin.ik(tf, q_init=q_seed, max_iters=100)
        self.data.ctrl[:_N_ARM_JOINTS] = q_sol[:_N_ARM_JOINTS]

        if self.model.nu > _N_ARM_JOINTS:
            grip = float(np.clip((float(action[3]) + 1.0) * 0.5 * 255.0, 0.0, 255.0))
            self.data.ctrl[_N_ARM_JOINTS] = grip

    def get_ee_pos(self) -> np.ndarray:
        return self.data.body("hand").xpos.copy()

    def get_ee_body(self) -> str:
        return "hand"
