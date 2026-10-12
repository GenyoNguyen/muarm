"""Small correctness checks for CDFlow field evaluation bookkeeping."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from scripts.eval_cdflow_field import _check_metadata, _direction_cosines, SCHEMA_KEYS


class FieldEvaluationTests(unittest.TestCase):
    def test_direction_cosine_skips_undefined_reference(self) -> None:
        predicted = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        reference = np.array([[2.0, 0.0], [0.0, 0.0], [0.0, -3.0]])
        cosine, count = _direction_cosines(predicted, reference)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(cosine, 0.0)

    def test_metadata_rejects_incompatible_single_sphere_dataset(self) -> None:
        metadata = {key: f"value-{key}" for key in SCHEMA_KEYS}
        metadata.update(task="reach_obstacles", single_sphere_index=1)
        joint_limits = np.tile(np.array([[-2.0, 2.0]]), (7, 1))
        field = SimpleNamespace(
            metadata=metadata, max_spheres=3,
            model=SimpleNamespace(config=SimpleNamespace(joint_limits=joint_limits)),
        )
        dataset = SimpleNamespace(
            metadata=dict(metadata), scene=np.zeros((2, 3, 4)),
            joint_limits=joint_limits.astype(np.float32),
        )
        self.assertEqual(_check_metadata(field, dataset, None), ("reach_obstacles", True))
        dataset.metadata["single_sphere_index"] = 0
        with self.assertRaisesRegex(ValueError, "single_sphere_index"):
            _check_metadata(field, dataset, None)

    def test_metadata_rejects_joint_limit_mismatch(self) -> None:
        metadata = {key: f"value-{key}" for key in SCHEMA_KEYS}
        metadata["task"] = "reach_obstacles"
        field = SimpleNamespace(
            metadata=metadata, max_spheres=3,
            model=SimpleNamespace(config=SimpleNamespace(
                joint_limits=np.tile(np.array([[-2.0, 2.0]]), (7, 1)),
            )),
        )
        dataset = SimpleNamespace(
            metadata=dict(metadata), scene=np.zeros((1, 3, 4)),
            joint_limits=np.tile(np.array([[-1.0, 1.0]]), (7, 1)),
        )
        with self.assertRaisesRegex(ValueError, "joint limits"):
            _check_metadata(field, dataset, None)


if __name__ == "__main__":
    unittest.main()
