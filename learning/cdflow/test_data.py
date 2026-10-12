"""Geometry and variable-set data checks against the actual Panda MJCF."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np

from learning.cdflow.collision import PandaCollisionOracle
from learning.cdflow.data import (
    BoundaryDataConfig, generate_dataset, load_dataset, save_dataset,
)
from learning.cdflow.scene import SphereScene, sample_scene_variants
from learning.robots.panda import _build_model


def test_mujoco_oracle_gap_sign_and_sphere_radius() -> None:
    model = _build_model("reach_obstacles")
    scene = SphereScene.from_model(model)
    assert scene.spheres.shape == (3, 4)
    np.testing.assert_allclose(scene.spheres[:, 3], 0.12)
    other_data = mujoco.MjData(model)
    before = other_data.qpos.copy()
    oracle = PandaCollisionOracle(model, scene)
    home = model.key_qpos[0, oracle.qpos_indices].copy()
    free_gap = oracle.gap(home)
    assert free_gap > 0
    np.testing.assert_array_equal(other_data.qpos, before)

    bigger = scene.spheres.copy()
    bigger[:, 3] += 0.01
    oracle.set_scene(SphereScene(bigger, scene.mask, scene.names))
    np.testing.assert_allclose(oracle.gap(home), free_gap - 0.01, atol=1e-7)

    rng = np.random.default_rng(4)
    oracle.set_scene(scene)
    hit = None
    for _ in range(500):
        candidate = rng.uniform(oracle.joint_limits[:, 0], oracle.joint_limits[:, 1])
        if oracle.gap(candidate) < -0.005:
            hit = candidate
            break
    assert hit is not None
    assert not oracle.is_valid(hit)


def test_variable_boundary_sets_and_scene_splits(tmp_path) -> None:
    model = _build_model("reach_obstacles")
    scenes = sample_scene_variants(SphereScene.from_model(model), 3, seed=2)
    config = BoundaryDataConfig(
        queries_per_scene=1, global_starts=4, local_starts=2,
        optimizer_maxiter=75,
    )
    dataset = generate_dataset(model, scenes, config, seed=2)
    path = save_dataset(tmp_path / "small.npz", dataset)
    loaded = load_dataset(path)
    assert loaded["q"].shape == (3, 7)
    assert loaded["offsets"].shape == (4,)
    assert set(loaded["split"].tolist()) == {0, 1, 2}
    assert len(np.unique(loaded["scene_id"])) == 3
    assert json.loads(str(loaded["metadata_json"]))["version"] == 1

    oracle = PandaCollisionOracle(model)
    for i, query in enumerate(loaded["q"]):
        oracle.set_scene(scenes[int(loaded["scene_id"][i])])
        start, end = loaded["offsets"][i:i + 2]
        targets = loaded["boundary_q"][start:end]
        assert len(targets) >= 1
        gaps = np.array([oracle.gap(target) for target in targets])
        assert np.max(np.abs(gaps)) <= config.boundary_tolerance
        distances = np.linalg.norm(query - targets, axis=1)
        expected_gradient = np.mean((query - targets) / distances[:, None], axis=0)
        np.testing.assert_allclose(loaded["y_d"][i], np.mean(distances), atol=1e-5)
        np.testing.assert_allclose(loaded["y_g"][i], expected_gradient, atol=1e-5)
        assert np.all(distances <= loaded["nearest_distance"][i] +
                      config.near_distance_tolerance + 1e-5)


def test_single_sphere_cli_keeps_full_scene_cardinality(tmp_path) -> None:
    output = tmp_path / "single_sphere.npz"
    script = Path(__file__).resolve().parents[2] / "scripts" / "generate_cdflow_data.py"
    subprocess.run([
        sys.executable, str(script), "--output", str(output),
        "--task", "reach_obstacles", "--single-sphere-index", "1",
        "--num-scenes", "1", "--queries-per-scene", "1",
        "--global-starts", "3", "--local-starts", "2", "--seed", "9",
    ], check=True, capture_output=True, text=True)
    dataset = load_dataset(output)
    assert dataset["scene"].shape == (1, 3, 4)
    np.testing.assert_array_equal(dataset["scene_mask"], [[False, True, False]])
    assert json.loads(str(dataset["metadata_json"]))["single_sphere_index"] == 1
    model = _build_model("reach_obstacles")
    oracle = PandaCollisionOracle(model)
    oracle.set_scene(SphereScene(
        dataset["scene"][0], dataset["scene_mask"][0],
        tuple(f"reach_obstacle_{i}" for i in range(3)),
    ))
    assert abs(oracle.gap(dataset["boundary_q"][0])) < 0.002


def test_button_cli_samples_reachable_small_spheres(tmp_path) -> None:
    output = tmp_path / "buttons.npz"
    script = Path(__file__).resolve().parents[2] / "scripts" / "generate_cdflow_data.py"
    subprocess.run([
        sys.executable, str(script), "--output", str(output),
        "--task", "reach_obstacles", "--button-scenes",
        "--num-scenes", "3", "--queries-per-scene", "1",
        "--global-starts", "3", "--local-starts", "2", "--seed", "9",
    ], check=True, capture_output=True, text=True)
    dataset = load_dataset(output)
    metadata = json.loads(str(dataset["metadata_json"]))
    settings = metadata["button_scene_settings"]
    assert metadata["scene_generation"] == "fk_witness_button"
    assert settings["radius_m"] == 0.01
    assert settings["sphere_index"] == 1
    assert settings["seed"] == 9
    assert dataset["scene"].shape == (3, 3, 4)
    np.testing.assert_array_equal(dataset["scene_mask"], [[False, True, False]] * 3)
    centers = dataset["scene"][:, 1, :3]
    np.testing.assert_allclose(dataset["scene"][:, 1, 3], 0.01)
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            assert np.linalg.norm(centers[left] - centers[right]) >= 0.05 - 1e-6

    audit = metadata["button_scene_audit"]
    assert len(audit) == 3
    oracle = PandaCollisionOracle(_build_model("reach_obstacles"))
    for scene_id, row in enumerate(audit):
        scene_row = int(np.flatnonzero(dataset["scene_id"] == scene_id)[0])
        scene = SphereScene(
            dataset["scene"][scene_row], dataset["scene_mask"][scene_row],
            tuple(f"reach_obstacle_{i}" for i in range(3)),
        )
        oracle.set_scene(scene)
        np.testing.assert_allclose(row["center_xyz_m"], centers[scene_row], atol=1e-6)
        # Scene rows are stored as float32, so allow a small reconstruction
        # tolerance while also requiring an actual MuJoCo contact pair.
        assert oracle.gap(np.asarray(row["witness_q_rad"])) <= 1e-6
        obstacle_ids = set(oracle._obstacle_ids)
        arm_ids = set(oracle._arm_geoms)
        assert any(
            contact.dist <= 0.0
            and ((contact.geom1 in obstacle_ids and contact.geom2 in arm_ids)
                 or (contact.geom2 in obstacle_ids and contact.geom1 in arm_ids))
            for contact in oracle.data.contact
        )
        assert abs(oracle.gap(np.asarray(row["boundary_witness_q_rad"]))) <= 0.002
        assert abs(row["boundary_witness_gap_m"]) <= 0.002


class CDFlowDataTests(unittest.TestCase):
    """Run the geometry checks with the repository's stdlib test runner."""

    def test_mujoco_oracle_gap_sign_and_sphere_radius(self) -> None:
        test_mujoco_oracle_gap_sign_and_sphere_radius()

    def test_variable_boundary_sets_and_scene_splits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cdflow-data-test-") as folder:
            test_variable_boundary_sets_and_scene_splits(Path(folder))

    def test_single_sphere_cli_keeps_full_scene_cardinality(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cdflow-one-sphere-test-") as folder:
            test_single_sphere_cli_keeps_full_scene_cardinality(Path(folder))

    def test_button_cli_samples_reachable_small_spheres(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cdflow-button-test-") as folder:
            test_button_cli_samples_reachable_small_spheres(Path(folder))


if __name__ == "__main__":
    unittest.main()
