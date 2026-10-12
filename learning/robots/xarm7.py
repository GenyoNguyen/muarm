"""UFACTORY xArm7 robot wrapper.

Supports the MuJoCo Menagerie models with and without the gripper.
Action: [dx, dy, dz, grip] with the hand, [dx, dy, dz] without it.
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

_XARM7_DIR = ROOT / "models" / "ufactory_xarm7"
_MESH_DIR = _XARM7_DIR / "assets"
_N_ARM_JOINTS = 7

_WS_LOW = np.array([0.20, -0.50, 0.05], dtype=np.float64)
_WS_HIGH = np.array([0.80, 0.50, 0.75], dtype=np.float64)

_TASK_BODIES: dict[str, str] = {
    "reach": """
    <site name="target" type="sphere" size="0.02" rgba="1 0 0 0.5" pos="0.5 0 0.3"/>
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


def _build_model(task_name: str, with_hand: bool) -> mujoco.MjModel:
    model_file = _XARM7_DIR / ("xarm7.xml" if with_hand else "xarm7_nohand.xml")
    xml = f"""<mujoco model="xarm7_{task_name}">
  <include file="{model_file}"/>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.2 0.3 0.4 1"/>
    {_TASK_BODIES.get(task_name, "")}
  </worldbody>
</mujoco>"""

    assets = {
        name: (_MESH_DIR / name).read_bytes()
        for name in os.listdir(_MESH_DIR)
        if (_MESH_DIR / name).is_file()
    }
    return mujoco.MjModel.from_xml_string(xml, assets)


class XArm7Env(MuJocoRobotEnv):
    """UFACTORY xArm7 manipulation environment."""

    def __init__(
        self,
        task,
        task_name: str = "reach",
        with_hand: bool = True,
        model_path: str | None = None,
        render_mode: str | None = None,
        n_substeps: int = 20,
        horizon: int = 200,
        action_scale: float = 0.03,
        obs_dim: int = 32,
    ):
        self.with_hand = with_hand
        self.action_scale = action_scale
        self.obs_dim = obs_dim
        self._ee_site = "link_tcp" if with_hand else "attachment_site"

        model_file = _XARM7_DIR / ("xarm7.xml" if with_hand else "xarm7_nohand.xml")
        self._kin = PandaKinematics(ee_frame=self._ee_site)
        self._kin.build_from_mjcf(str(model_file))

        model = mujoco.MjModel.from_xml_path(model_path) if model_path else _build_model(task_name, with_hand)
        super().__init__(
            model_path=None,
            model=model,
            task=task,
            render_mode=render_mode,
            n_substeps=n_substeps,
            horizon=horizon,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self._ee_rotation = self.data.site(self._ee_site).xmat.reshape(3, 3).copy()
        # Pinocchio omits the MJCF root body's translation; calibrate its frame to MuJoCo.
        self._kin_translation_offset = self.get_ee_pos() - self._kin.fk(
            self.data.qpos[:_N_ARM_JOINTS]
        )[:3, 3]
        return observation, info

    def _build_spaces(self) -> None:
        action_dim = 4 if self.with_hand else 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })

    def _get_obs(self) -> dict:
        state_dim = _N_ARM_JOINTS + int(self.with_hand)
        task_obs = self.task.get_obs(self.model, self.data, self)
        achieved = self.task.achieved_goal(self.model, self.data, self)
        raw = np.concatenate([
            self.get_ee_pos(),
            self.data.qpos[:state_dim],
            self.data.qvel[:state_dim],
            task_obs,
        ]).astype(np.float32)
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[:min(len(raw), self.obs_dim)] = raw[:self.obs_dim]
        return {
            "observation": obs,
            "achieved_goal": achieved.astype(np.float32),
            "desired_goal": self._goal.astype(np.float32),
        }

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        target = np.clip(
            self.get_ee_pos() + self.action_scale * action[:3],
            _WS_LOW,
            _WS_HIGH,
        )

        target_tf = np.eye(4, dtype=np.float64)
        target_tf[:3, :3] = self._ee_rotation
        target_tf[:3, 3] = target - self._kin_translation_offset
        q_sol, _ = self._kin.ik(
            target_tf,
            q_init=self.data.qpos[:_N_ARM_JOINTS],
            max_iters=100,
        )
        self.data.ctrl[:_N_ARM_JOINTS] = q_sol[:_N_ARM_JOINTS]

        if self.with_hand:
            self.data.ctrl[_N_ARM_JOINTS] = np.clip((action[3] + 1.0) * 127.5, 0.0, 255.0)

    def get_ee_pos(self) -> np.ndarray:
        return self.data.site(self._ee_site).xpos.copy()

    def get_ee_body(self) -> str:
        return "xarm_gripper_base_link" if self.with_hand else "link7"
