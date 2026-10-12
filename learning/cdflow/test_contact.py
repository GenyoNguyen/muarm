"""Contact projection checks: sign, safety, and MuJoCo contact semantics."""
from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from learning.cdflow.collision import PandaCollisionOracle
from learning.algos.cdflow_projection import (
    CDFlowProjectionConfig, CDFlowProjectionPolicy,
)
from learning.robots.panda import _build_model
from scripts.eval_cdflow_contact import (
    ContactConfig, _arm_sphere_contact, _episode, _single_sphere_scenes,
    _load_button_scenes, _sample_starts, _training_coverage, evaluate_dataset,
)


class _ConstantField:
    device = torch.device("cpu")

    def __init__(self):
        self.modes = []

    def distance_and_gradient(self, scene, q, *, mode, samples):
        self.modes.append(mode)
        # Outward direction points away from the synthetic obstacle at q[0]>0.
        return torch.tensor([0.1]), torch.tensor([[-1.0, 0, 0, 0, 0, 0, 0]])


class _SyntheticOracle:
    def __init__(self, unsafe_above: float = float("inf")):
        self.joint_limits = np.asarray([[-1.0, 1.0]] * 7)
        self.model = SimpleNamespace(nkey=1, key_qpos=np.zeros((1, 7)))
        self.qpos_indices = np.arange(7)
        self.current_q = np.zeros(7)
        self.unsafe_above = unsafe_above

    def in_limits(self, q):
        q = np.asarray(q)
        return bool(q.shape == (7,) and np.all(np.isfinite(q))
                    and np.all(q >= self.joint_limits[:, 0])
                    and np.all(q <= self.joint_limits[:, 1]))

    def gap(self, q):
        self.current_q = np.asarray(q).copy()
        return float(0.08 - q[0])

    def constraint_gap(self, q):
        self.current_q = np.asarray(q).copy()
        return -0.01 if q[0] > self.unsafe_above else 1.0


class ContactEvaluationTests(unittest.TestCase):
    def test_controller_defaults_to_paper_mc_projection(self) -> None:
        field = _ConstantField()
        oracle = _SyntheticOracle()
        policy = CDFlowProjectionPolicy(
            field, oracle, object(), CDFlowProjectionConfig(max_steps=4),
            contact_checker=lambda current: current.current_q[0] >= 0.079,
        )
        policy.start_episode(np.zeros(7))
        visited = []
        result = policy.rollout(on_step=lambda elapsed, q: visited.append(q.copy()))
        self.assertTrue(result.success)
        self.assertEqual(result.status, "contact")
        self.assertEqual(result.attempted_steps, 2)
        self.assertEqual(field.modes, ["mc", "mc"])
        self.assertGreater(result.trajectory[-1, 0], 0)
        self.assertEqual(len(visited), 2)

    def test_controller_requires_real_contact_to_succeed(self) -> None:
        policy = CDFlowProjectionPolicy(
            _ConstantField(), _SyntheticOracle(), object(),
            CDFlowProjectionConfig(max_steps=4),
            contact_checker=lambda current: False,
        )
        policy.start_episode(np.zeros(7))
        result = policy.rollout()
        self.assertFalse(result.success)
        self.assertEqual(result.status, "sphere_gap_without_arm_contact")

    def test_controller_respects_three_second_default(self) -> None:
        self.assertEqual(CDFlowProjectionConfig().time_limit_s, 3.0)

    def test_projection_uses_negative_outward_direction(self) -> None:
        oracle = _SyntheticOracle()
        config = ContactConfig(episodes=1, max_steps=4, edge_resolution=0.005)
        with patch(
            "scripts.eval_cdflow_contact._arm_sphere_contact",
            side_effect=lambda current: current.current_q[0] >= 0.079,
        ):
            result = _episode(
                _ConstantField(), oracle, object(), np.zeros(7), config,
                mode="regression", episode_seed=1,
            )
        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "contact")
        self.assertEqual(result["steps"], 2)
        self.assertGreater(result["final_q"][0], 0.0)
        self.assertEqual(result["field_queries"], 2)

    def test_non_obstacle_edge_stops_before_contact(self) -> None:
        oracle = _SyntheticOracle(unsafe_above=0.02)
        config = ContactConfig(episodes=1, max_steps=4, edge_resolution=0.005)
        with patch(
            "scripts.eval_cdflow_contact._arm_sphere_contact",
            return_value=False,
        ):
            result = _episode(
                _ConstantField(), oracle, object(), np.zeros(7), config,
                mode="mc", episode_seed=1,
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "unsafe_non_obstacle_edge")

    def test_one_sphere_mask_and_real_contact_pair(self) -> None:
        oracle_scene, field_scene = _single_sphere_scenes(1)
        self.assertEqual(int(oracle_scene.mask.sum()), 1)
        self.assertEqual(field_scene.spheres.shape, (1, 4))
        oracle = PandaCollisionOracle(_build_model("reach_obstacles"), oracle_scene)
        home = oracle.data.qpos[oracle.qpos_indices].copy()
        self.assertGreater(oracle.gap(home), 0.0)
        self.assertFalse(_arm_sphere_contact(oracle))

        # Search a deterministic nearby pose whose overlap is also reported as
        # a MuJoCo contact pair. This catches a false success based only on a
        # geometric distance from a non-contacting or inactive geom.
        rng = np.random.default_rng(7)
        for _ in range(1000):
            q = np.clip(
                home + rng.normal(0, 0.6, size=7),
                oracle.joint_limits[:, 0], oracle.joint_limits[:, 1],
            )
            if oracle.gap(q) <= 0 and _arm_sphere_contact(oracle):
                break
        else:
            self.fail("no real Panda arm-to-selected-sphere contact was found")
        self.assertLessEqual(oracle.gap(q), 0.0)
        self.assertTrue(_arm_sphere_contact(oracle))

    def test_override_button_geometry(self) -> None:
        oracle_scene, field_scene = _single_sphere_scenes(
            1, sphere_center=np.array([0.55, 0.05, 0.31]), sphere_radius=0.01,
        )
        self.assertEqual(int(oracle_scene.mask.sum()), 1)
        np.testing.assert_allclose(field_scene.spheres[0], [0.55, 0.05, 0.31, 0.01])
        np.testing.assert_allclose(
            oracle_scene.spheres[1], field_scene.spheres[0],
        )

    def test_three_sphere_training_is_reported_as_scene_shift(self) -> None:
        import json

        metadata = {"robot": "Franka Panda, 7 arm joints, radians"}
        field = SimpleNamespace(metadata=metadata)
        scene = np.zeros((2, 3, 4), dtype=np.float32)
        scene[:, :, 3] = 0.12
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "training.npz"
            np.savez(
                path, scene=scene, scene_mask=np.ones((2, 3), dtype=bool),
                split=np.zeros(2, dtype=np.int8),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            coverage = _training_coverage(field, path, scene[0, 0])
        self.assertEqual(coverage["status"], "unseen_active_count")

    def test_witness_based_starts_for_varied_button_scene(self) -> None:
        oracle = _SyntheticOracle()
        witness = np.zeros(7)
        witness[0] = 0.08  # contact, so the home pose is not useful here
        config = ContactConfig(episodes=1, max_start_attempts=100,
                               start_noise=0.05, seed=4)
        with patch("scripts.eval_cdflow_contact._arm_sphere_contact", return_value=False):
            starts = _sample_starts(oracle, config, anchor_q=witness)
        self.assertEqual(len(starts), 1)
        self.assertGreaterEqual(starts[0][1], config.start_gap_min)
        self.assertLessEqual(starts[0][1], config.start_gap_max)
        self.assertLess(np.linalg.norm(starts[0][0] - witness), 0.3)

    def test_dataset_contact_evaluates_each_distinct_scene_once(self) -> None:
        import json

        scene = np.zeros((4, 3, 4), dtype=np.float32)
        scene[:, :, 3] = 0.12
        scene[:2, 1] = [0.48, -0.05, 0.30, 0.01]
        scene[2:, 1] = [0.66, 0.13, 0.43, 0.01]
        mask = np.zeros((4, 3), dtype=bool)
        mask[:, 1] = True
        metadata = {"button_scene_audit": [
            {"scene_id": 0, "witness_q_rad": [0.0] * 7},
            {"scene_id": 1, "witness_q_rad": [0.1] * 7},
        ]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "buttons.npz"
            np.savez(path, q=np.zeros((4, 7)), scene=scene, scene_mask=mask,
                     scene_id=np.array([0, 0, 1, 1]),
                     split=np.array([0, 0, 2, 2]),
                     metadata_json=np.asarray(json.dumps(metadata)))
            records, _ = _load_button_scenes(path, "test")
            self.assertEqual([record.scene_id for record in records], [1])
            self.assertAlmostEqual(records[0].sphere[3], 0.01)

            field = SimpleNamespace(device=torch.device("cpu"), metadata={},
                                    distance_and_gradient=lambda *a, **k: None)

            def fake_episode(field, oracle, selected_scene, start, config, *, mode,
                             episode_seed):
                success = selected_scene.spheres[0, 0] < 0.6
                return {
                    "success": success, "reason": "contact" if success else "max_steps",
                    "steps": 1, "path_length_rad": 0.05,
                    "field_queries": 1, "total_field_query_ms": 1.0,
                }

            with patch("scripts.eval_cdflow_contact.CDFlowField.load", return_value=field), \
                 patch("scripts.eval_cdflow_contact._check_checkpoint_robot"), \
                 patch("scripts.eval_cdflow_contact._sample_starts",
                       return_value=[(np.zeros(7), 0.1)]), \
                 patch("scripts.eval_cdflow_contact._episode", side_effect=fake_episode):
                result = evaluate_dataset(
                    Path(folder) / "fake.pt", path, split="all",
                    config=ContactConfig(episodes=1),
                )
        self.assertEqual(result["scene_count"], 2)
        self.assertEqual([item["scene_id"] for item in result["scenes"]], [0, 1])
        self.assertEqual(result["summary"]["mc"]["episodes"], 2)
        self.assertEqual(result["summary"]["mc"]["scenes_with_contact"], 1)
        self.assertEqual(result["summary"]["mc"]["scene_success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
