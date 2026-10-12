"""Regression checks for the source-style Franka CDF-MPPI inference loop.

Run: python -m unittest learning.algos.test_cdf_mppi
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import torch

from learning.algos.cdf_mppi import (
    CDFMPPIConfig, CDFMPPIPolicy, CDFMPPIRollout, load_cdf_mppi,
)
from learning.envs.registry import make_env
import learning.play as playback


class CDFMPPIRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)

    def test_episode_maps_displayed_goal_to_joints_and_preserves_task(self):
        for task in ("reach_obstacles", "reach_obstacles_2"):
            with self.subTest(task=task):
                env = make_env("panda", task)
                try:
                    env.reset(seed=7)
                    geometry = env.model.geom_pos.copy()
                    task_points = env.task._obstacle_points
                    points_before = task_points.clone()
                    task_goal = env.goal.copy()
                    policy = CDFMPPIPolicy(env)
                    policy.start_episode()
                    np.testing.assert_allclose(
                        env.data.qpos[:7], [-1.57, .4, 0, -1.2708, 0, 1.8675, 0]
                    )
                    np.testing.assert_allclose(env.data.qvel, 0)
                    self.assertIsNotNone(policy.goal)
                    self.assertGreater(policy._safety_gap(policy.goal.cpu().numpy()), 0.0)
                    np.testing.assert_array_equal(policy.desired_goal, task_goal)
                    mujoco.mj_copyData(policy._collision_data, env.model, env.data)
                    policy._set_pose(policy._collision_data, policy.goal.cpu().numpy())
                    self.assertTrue(env.task.is_success(
                        policy._collision_data.body("hand").xpos, task_goal
                    ))
                    np.testing.assert_array_equal(env.model.geom_pos, geometry)
                    np.testing.assert_array_equal(env.goal, task_goal)
                    torch.testing.assert_close(env.task._obstacle_points, points_before)
                    self.assertIs(env.task._obstacle_points, task_points)
                    self.assertIs(policy.points, task_points)
                    if task == "reach_obstacles_2":
                        np.testing.assert_allclose(policy.points[0].cpu(), [.5, -.14, .3])
                    first_joint_goal = policy.goal.cpu().numpy().copy()
                    env.reset(seed=8)
                    policy.start_episode()
                    self.assertGreater(
                        np.linalg.norm(policy.goal.cpu().numpy() - first_joint_goal), .01
                    )
                    np.testing.assert_array_equal(policy.desired_goal, env.goal)
                finally:
                    env.close()

    def test_mppi_updates_around_raw_mean_then_projects_only_executed_velocity(self):
        env = make_env("panda", "reach_obstacles_2")
        try:
            policy = CDFMPPIPolicy(env, CDFMPPIConfig(n_samples=3, seed=0))
            q = torch.as_tensor([-1.57, .4, 0, -1.2708, 0, 1.8675, 0])
            q[0] = policy.limits[0, 1] - .001
            samples = torch.zeros((3, 7))
            samples[:, 0] = torch.tensor([2.0, 1.0, -1.0])
            costs = torch.tensor([0.0, 1.0, 2.0])
            with (
                patch.object(policy, "_sample_velocities", return_value=samples),
                patch.object(policy, "_cost", return_value=costs) as cost_fn,
            ):
                q_next = policy.propose(q)

            torch.testing.assert_close(cost_fn.call_args.args[1], samples)
            weights = torch.softmax(-costs, dim=0)
            raw_mean = (weights[:, None] * samples).sum(0)
            centered = samples - raw_mean
            raw_cov = torch.einsum("n,ni,nj->ij", weights, centered, centered)
            torch.testing.assert_close(policy.mean, .5 * raw_mean)
            torch.testing.assert_close(policy.cov, .5 * policy.eye + .5 * raw_cov + .001 * policy.eye)
            self.assertGreater(raw_mean[0].item(), .1)
            self.assertAlmostEqual(q_next[0], policy.limits[0, 1].item(), places=6)
            self.assertEqual(q_next.shape, (7,))
        finally:
            env.close()

    def test_cdf_gradient_penalty_only_activates_near_obstacles(self):
        env = make_env("panda", "reach_obstacles_2")
        try:
            env.reset(seed=0)
            policy = CDFMPPIPolicy(env)
            policy.start_episode()
            q = policy.current_q
            velocities = torch.zeros((2, 7))
            velocities[:, 0] = torch.tensor([1.0, -1.0])
            normal = torch.zeros((1, 7))
            normal[0, 0] = 1.0
            with patch.object(
                policy.cdf, "distance_and_gradient", return_value=(torch.tensor([.1]), normal)
            ):
                active = policy._cost(q, velocities)
            with patch.object(
                policy.cdf, "distance_and_gradient", return_value=(torch.tensor([1.0]), normal)
            ):
                inactive = policy._cost(q, velocities)
            self.assertAlmostEqual((active[0] - inactive[0]).item(), 0.0, delta=10.0)
            self.assertGreater((active[1] - inactive[1]).item(), 170_000)
        finally:
            env.close()

    def test_endpoint_collision_aborts_without_accepting_proposal_or_stepping_servo(self):
        env = make_env("panda", "reach_obstacles_2")
        try:
            env.reset(seed=7)
            policy = CDFMPPIPolicy(env)
            policy.start_episode()
            start = policy.current_q.cpu().numpy().copy()
            proposed = start.copy()
            proposed[0] += .01
            policy._set_pose(policy._collision_data, proposed)
            robot_geom = policy._robot_geoms[-1]
            obstacle = next(iter(env.task._obstacle_geom_ids))
            env.model.geom_pos[obstacle] = policy._collision_data.geom_xpos[robot_geom]
            mujoco.mj_forward(env.model, env.data)
            with (
                patch.object(policy, "propose", return_value=proposed),
                patch.object(env, "step", side_effect=AssertionError("No servo stepping")),
            ):
                result = policy.rollout(max_iter=3)
            self.assertEqual(result.status, "collision")
            self.assertLess(result.collision_gap, 0)
            self.assertEqual(result.attempted_steps, 1)
            self.assertIsNone(result.minimum_gap)
            self.assertEqual(len(result.trajectory), 1)
            np.testing.assert_array_equal(result.trajectory[0], start)
            np.testing.assert_array_equal(env.data.qpos[:7], start)
        finally:
            env.close()

    def test_intermediate_collision_rejects_a_free_endpoint(self):
        env = make_env("panda", "reach_obstacles")
        try:
            env.reset(seed=7)
            policy = CDFMPPIPolicy(env, CDFMPPIConfig(edge_resolution=0.01))
            policy.start_episode()
            start = policy.current_q.cpu().numpy().copy()
            proposed = start.copy()
            proposed[0] += 0.04

            def synthetic_gap(q):
                moved = q[0] - start[0]
                return -0.01 if 0.015 < moved < 0.025 else 0.10

            with (
                patch.object(policy, "propose", return_value=proposed),
                patch.object(policy, "_collision_gap", side_effect=synthetic_gap),
            ):
                accepted, gap = policy.step()
            self.assertFalse(accepted)
            self.assertLess(gap, 0.0)
            np.testing.assert_array_equal(policy.current_q.cpu().numpy(), start)
            np.testing.assert_array_equal(env.data.qpos[:7], start)
        finally:
            env.close()

    def test_floor_or_self_collision_rejects_an_edge(self):
        env = make_env("panda", "reach_obstacles")
        try:
            env.reset(seed=7)
            policy = CDFMPPIPolicy(env)
            policy.start_episode()
            start = policy.current_q.cpu().numpy().copy()
            proposed = start.copy()
            proposed[0] += 0.01
            with (
                patch.object(policy, "propose", return_value=proposed),
                patch.object(policy, "_collision_gap", return_value=0.1),
                patch.object(policy, "_constraint_gap_current", return_value=-0.01),
            ):
                accepted, gap = policy.step()
            self.assertFalse(accepted)
            self.assertEqual(gap, -0.01)
            np.testing.assert_array_equal(policy.current_q.cpu().numpy(), start)
        finally:
            env.close()

    def test_ambiguous_near_obstacle_field_stops_without_moving(self):
        env = make_env("panda", "reach_obstacles")
        try:
            env.reset(seed=0)
            policy = CDFMPPIPolicy(env, CDFMPPIConfig(seed=0))
            policy.start_episode()
            start = policy.current_q.cpu().numpy().copy()
            with patch.object(policy, "_distance_and_gradient", return_value=(
                torch.tensor([0.01]), torch.zeros((1, 7)),
            )):
                result = policy.rollout(max_iter=1)
            self.assertEqual(result.status, "ambiguous_field")
            self.assertEqual(result.attempted_steps, 1)
            np.testing.assert_array_equal(result.trajectory[0], start)
            np.testing.assert_array_equal(env.data.qpos[:7], start)
        finally:
            env.close()

    def test_short_real_rollout_tracks_accepted_joint_state(self):
        env = make_env("panda", "reach_obstacles_2")
        try:
            env.reset()
            policy = CDFMPPIPolicy(env, CDFMPPIConfig(seed=0))
            geometry = env.model.geom_pos.copy()
            points = env.task._obstacle_points.clone()
            policy.start_episode()
            with patch.object(env, "step", side_effect=AssertionError("No servo stepping")):
                result = policy.rollout(max_iter=3)
            self.assertEqual(result.status, "max_iter")
            self.assertEqual(result.trajectory.shape, (4, 7))
            np.testing.assert_array_equal(env.data.qpos[:7], result.trajectory[-1])
            self.assertTrue(np.all(np.abs(np.diff(result.trajectory, axis=0)) <= .02701))
            np.testing.assert_array_equal(env.model.geom_pos, geometry)
            torch.testing.assert_close(env.task._obstacle_points, points)
        finally:
            env.close()

    def test_reported_success_matches_the_displayed_cartesian_goal(self):
        for task, seed in (("reach_obstacles", 0), ("reach_obstacles_2", 7)):
            with self.subTest(task=task):
                env = make_env("panda", task)
                try:
                    env.reset(seed=seed)
                    policy = CDFMPPIPolicy(env, CDFMPPIConfig(seed=0))
                    policy.start_episode()
                    initial_error = np.linalg.norm(env.get_ee_pos() - env.goal)
                    result = policy.rollout()
                    actual_error = np.linalg.norm(env.get_ee_pos() - env.goal)
                    self.assertAlmostEqual(result.ee_error, actual_error)
                    self.assertEqual(
                        result.success, env.task.is_success(env.get_ee_pos(), env.goal)
                    )
                    self.assertLess(actual_error, initial_error)
                finally:
                    env.close()

    def test_joint_goal_match_alone_cannot_report_success(self):
        env = make_env("panda", "reach_obstacles_2")
        try:
            env.reset(seed=7)
            policy = CDFMPPIPolicy(env)
            policy.start_episode()
            policy.goal = policy.current_q.clone()
            stationary = policy.current_q.cpu().numpy().copy()
            with patch.object(policy, "propose", return_value=stationary):
                result = policy.rollout(max_iter=1)
            self.assertEqual(result.status, "max_iter")
            self.assertGreater(result.ee_error, env.task.distance_threshold)
        finally:
            env.close()

    def test_old_checkpoint_ignores_previous_collision_fallback_settings(self):
        env = make_env("panda", "reach_obstacles")
        try:
            policy = CDFMPPIPolicy(env)
            with tempfile.TemporaryDirectory(prefix="cdf-mppi-test-") as folder:
                path = policy.save(str(Path(folder) / "planner"))
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                checkpoint["version"] = 1
                checkpoint["config"].pop("dt")
                checkpoint["config"].update(
                    collision_check_distance=.05, settling_steps=3, recovery_samples=1000
                )
                torch.save(checkpoint, path)
                restored = load_cdf_mppi(env, path)
                self.assertEqual(restored.cfg.dt, .01)
                self.assertFalse(hasattr(restored.cfg, "recovery_samples"))
                env.reset(seed=7)
                policy.start_episode()
                restored.start_episode()
                np.testing.assert_allclose(restored.goal.cpu(), policy.goal.cpu())
        finally:
            env.close()

    def test_play_dispatches_learned_and_pretrained_fields_to_joint_rollout(self):
        for algo in ("cdf_mppi", "cdflow_mppi"):
            with self.subTest(algo=algo):
                env = Mock()
                env.reset.return_value = ({}, {})
                env.get_ee_pos.return_value = np.zeros(3)
                policy = Mock()
                policy.rollout.return_value = CDFMPPIRollout(
                    np.zeros((1, 7)), "collision", 1, -.01, .2
                )
                output = io.StringIO()
                with (
                    patch.object(playback, "make_env", return_value=env),
                    patch.object(playback, "_load_policy", return_value=policy),
                    patch.object(playback, "_rt_step", side_effect=AssertionError("No env.step")),
                    contextlib.redirect_stdout(output),
                ):
                    playback.play("panda", "reach_obstacles_2", algo, "unused.pt", 1,
                                  seed=7)
                policy.start_episode.assert_called_once()
                self.assertEqual(policy.cfg.seed, 7)
                policy.rollout.assert_called_once()
                self.assertEqual(policy.rollout.call_args.kwargs["max_iter"], 500)
                self.assertTrue(policy.rollout.call_args.kwargs["render"])
                self.assertTrue(callable(policy.rollout.call_args.kwargs["on_step"]))
                self.assertIn("ee_error=0.2000 m", output.getvalue())
                env.close.assert_called_once()

    def test_other_algorithms_keep_their_environment_step_loop(self):
        for algo in ("sac", "bc"):
            with self.subTest(algo=algo):
                env = Mock()
                env.goal = np.zeros(3)
                env.get_ee_pos.return_value = np.zeros(3)
                obs = {"observation": np.zeros(32)}
                env.reset.return_value = (obs, {})
                policy = Mock(return_value=np.zeros(7))
                transitions = [
                    (obs, -1.0, False, False, {"is_collision": True, "is_success": False}),
                    (obs, 0.0, True, False, {"is_collision": False, "is_success": True}),
                ]
                with (
                    patch.object(playback, "make_env", return_value=env),
                    patch.object(playback, "_load_policy", return_value=policy),
                    patch.object(playback, "_rt_step", side_effect=transitions) as step,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    playback.play("panda", "reach_obstacles", algo, "unused.pt", 1)
                    self.assertEqual(step.call_count, 2)
                env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
