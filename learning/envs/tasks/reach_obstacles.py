"""Reach task with fixed obstacles and Panda CDF observations."""
from __future__ import annotations

import numpy as np
import torch

from learning.cdf import CDF
from learning.envs.tasks.reach import ReachTask


OBSTACLE_RADIUS = 0.12
OBSTACLE_CENTERS = (
    (0.50, -0.02, OBSTACLE_RADIUS),
    (0.50, -0.02, OBSTACLE_RADIUS*3),
    (0.50, -0.02, OBSTACLE_RADIUS*5),
)
SAMPLE_RADIUS = 0.2


class ReachObstaclesTask(ReachTask):
    """ReachTask augmented with obstacles and CDF observations."""

    obstacle_centers = OBSTACLE_CENTERS
    obstacle_radius = OBSTACLE_RADIUS
    sample_radius = SAMPLE_RADIUS

    def __init__(
        self,
        cdf_device: str = "cpu",
        collision_penalty: float = 1.0,
        use_cdf: bool = True,
        **reach_kwargs,
    ):
        super().__init__(**reach_kwargs)
        self.collision_penalty = collision_penalty
        self._cdf = CDF(device=cdf_device) if use_cdf else None
        self._obstacle_points = torch.tensor(
            self.obstacle_centers,
            dtype=torch.float32,
            device=self._cdf.device if self._cdf is not None else "cpu",
        )
        self._obstacle_geom_ids: frozenset[int] = frozenset()

    def reset(self, model, data, np_random) -> None:
        self._obstacle_geom_ids = frozenset(
            model.geom(f"reach_obstacle_{i}").id
            for i in range(len(self.obstacle_centers))
        )

    def sample_goal(self, np_random) -> np.ndarray:
        centers = np.asarray(self.obstacle_centers)
        rho = np.random.normal(self.sample_radius, (self.obstacle_radius + self.sample_radius))
        theta = np.random.uniform(0, 2*np.pi)
        x = self.sample_radius * np.cos(theta) + centers[0][0]
        y = self.sample_radius * np.sin(theta) + centers[0][1]
        z = np.random.uniform() * self.obstacle_radius * 6
        point = np.array([x, y, z])
        return point

    def get_info(self, model, data, env) -> dict:
        is_collision = any(
            contact.dist <= 0.0
            and (
                contact.geom1 in self._obstacle_geom_ids
                or contact.geom2 in self._obstacle_geom_ids
            )
            for contact in data.contact
        )
        return {"is_collision": is_collision}

    def compute_reward(self, achieved_goal, desired_goal, info) -> float:
        reward = super().compute_reward(achieved_goal, desired_goal, info)
        if info.get("is_collision", False):
            reward -= self.collision_penalty
        return reward

    def get_obs(self, model, data, env) -> np.ndarray:
        if self._cdf is None:
            # Direct-pose CDFlow playback does not consume task observations.
            return np.concatenate((env.get_ee_pos(), np.zeros(8))).astype(np.float32)
        q = torch.as_tensor(
            data.qpos[:7],
            dtype=torch.float32,
            device=self._cdf.device,
        ).unsqueeze(0)
        distance, gradient = self._cdf.distance_and_gradient(self._obstacle_points, q)

        joint_scores = gradient.squeeze(0)
        norm = joint_scores.norm()
        if norm.item() > 1e-6:
            joint_scores = joint_scores / norm

        cdf_obs = torch.cat((distance, joint_scores)).cpu().numpy()
        return np.concatenate((env.get_ee_pos(), cdf_obs)).astype(np.float32)
