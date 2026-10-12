"""Integration checks for the learned Panda field planner."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from learning.algos.cdf_mppi import CDFMPPIConfig
from learning.algos.cdflow_mppi import CDFlowMPPIPolicy
from learning.cdflow.field import CDFlowField
from learning.cdflow.model import CDFlowConfig, CDFlowModel
from learning.envs.registry import make_env


class CDFlowMPPIIntegrationTests(unittest.TestCase):
    def test_direct_playback_needs_no_legacy_cdf_weights(self) -> None:
        with patch(
            "learning.envs.tasks.reach_obstacles.CDF",
            side_effect=AssertionError("legacy CDF must not load"),
        ):
            env = make_env(
                "panda", "reach_obstacles", task_kwargs={"use_cdf": False},
            )
        try:
            env.reset(seed=0)
            self.assertIsNone(env.task._cdf)
            field = CDFlowField(CDFlowModel(CDFlowConfig(
                joint_limits=env.model.jnt_range[:7].tolist(),
                max_spheres=3, context_dim=12, scene_width=12,
                flow_width=16, flow_layers=1, distance_widths=(20, 12),
                ode_steps=1,
            )))
            policy = CDFlowMPPIPolicy(
                env, field, CDFMPPIConfig(n_samples=4, seed=0),
            )
            policy.start_episode()
            result = policy.rollout(max_iter=1)
            self.assertEqual(result.attempted_steps, 1)
            self.assertTrue(np.isfinite(result.trajectory).all())
        finally:
            env.close()

    def test_checkpoint_joint_limit_mismatch_is_rejected(self) -> None:
        env = make_env("panda", "reach_obstacles", task_kwargs={"use_cdf": False})
        try:
            field = CDFlowField(CDFlowModel(CDFlowConfig(
                joint_limits=[[-2.0, 2.0]] * 7, max_spheres=3,
                context_dim=12, scene_width=12, flow_width=16,
                flow_layers=1, distance_widths=(20, 12), ode_steps=1,
            )))
            with self.assertRaisesRegex(ValueError, "joint limits"):
                CDFlowMPPIPolicy(env, field)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
