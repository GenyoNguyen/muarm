"""Reach-obstacles variant based on the CDF-MPPI cross benchmark."""
import numpy as np

from learning.envs.tasks.reach_obstacles import ReachObstaclesTask


# CDF-MPPI uses two chains of 15 overlapping spheres for this benchmark.
_CROSS_CENTER = (0.50, 0.00, 0.30)
_CROSS_OFFSETS = tuple((i - 7) * 0.02 for i in range(15))

OBSTACLE_CENTERS = tuple(
    (_CROSS_CENTER[0], _CROSS_CENTER[1] + offset, _CROSS_CENTER[2])
    for offset in _CROSS_OFFSETS
) + tuple(
    (_CROSS_CENTER[0], _CROSS_CENTER[1], _CROSS_CENTER[2] + offset)
    for offset in _CROSS_OFFSETS
)
OBSTACLE_RADIUS = 0.05


class ReachObstacles2Task(ReachObstaclesTask):
    """Reach around a cross made from overlapping spheres."""

    obstacle_centers = OBSTACLE_CENTERS
    obstacle_radius = OBSTACLE_RADIUS
    default_goal_center = (0.5, 0.0, 0.32)
    default_goal_range = (0.12, 0.18, 0.10)

    def sample_goal(self, np_random) -> np.ndarray:
        centers = np.asarray(self.obstacle_centers)
        for _ in range(1000):
            goal = super().sample_goal(np_random)
            clearance = np.linalg.norm(centers - goal, axis=1).min()
            if clearance > OBSTACLE_RADIUS + self.distance_threshold:
                return goal
        raise RuntimeError("Unable to sample a collision-free goal.")
