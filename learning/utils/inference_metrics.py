"""Metrics shared by policy and planner inference rollouts."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InferenceEpisodeMetrics:
    """Accumulate policy computation and end-effector travel for one episode.

    Computation time includes episode setup and policy decisions, but excludes
    simulation steps, viewer rendering, and real-time playback sleeps.
    """

    computation_time_s: float = 0.0
    trajectory_length_m: float = 0.0
    _last_ee: np.ndarray | None = field(default=None, repr=False)

    def add_computation(self, elapsed_s: float) -> None:
        self.computation_time_s += elapsed_s

    def observe_ee(self, position: np.ndarray) -> None:
        position = np.asarray(position, dtype=np.float64).copy()
        if self._last_ee is not None:
            self.trajectory_length_m += float(np.linalg.norm(position - self._last_ee))
        self._last_ee = position
