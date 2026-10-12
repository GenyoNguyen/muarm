"""Explicit sphere scenes for Panda CDFlow data and inference.

Coordinates and radii are in metres.  Rows are ordered by the numeric suffix
of the MuJoCo ``reach_obstacle_<index>`` geom names, not by a Python set.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import mujoco
import numpy as np


@dataclass(frozen=True)
class SphereScene:
    """A variable-size scene encoded as ``(x, y, z, radius)`` rows."""

    spheres: np.ndarray
    mask: np.ndarray | None = None
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        spheres = np.array(self.spheres, dtype=np.float64, copy=True)
        if spheres.ndim != 2 or spheres.shape[1] != 4:
            raise ValueError("spheres must have shape (N, 4)")
        if not np.all(np.isfinite(spheres)) or np.any(spheres[:, 3] <= 0):
            raise ValueError("sphere centers and radii must be finite; radii must be positive")
        mask = (np.ones(len(spheres), dtype=bool) if self.mask is None
                else np.array(self.mask, dtype=bool, copy=True))
        if mask.shape != (len(spheres),):
            raise ValueError("mask must have shape (N,)")
        if self.names and len(self.names) != len(spheres):
            raise ValueError("names must match the number of spheres")
        object.__setattr__(self, "spheres", spheres)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "names", tuple(self.names))

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        prefix: str = "reach_obstacle_",
    ) -> "SphereScene":
        """Read actual active MuJoCo sphere positions and radii.

        ``data.geom_xpos`` is used when a live state is supplied.  For these
        tasks the obstacles are fixed world geoms, so ``model.geom_pos`` also
        gives their world coordinates and permits offline generation.
        """
        matches: list[tuple[int, int, str]] = []
        pattern = re.compile(rf"{re.escape(prefix)}(\d+)\Z")
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            match = pattern.fullmatch(name or "")
            if match:
                if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_SPHERE:
                    raise ValueError(f"obstacle geom {name} is not a sphere")
                matches.append((int(match.group(1)), geom_id, name))
        matches.sort()
        if not matches:
            raise ValueError(f"model has no sphere geoms named {prefix}<index>")
        if [n for n, _, _ in matches] != list(range(len(matches))):
            raise ValueError("obstacle geom indices must be contiguous from zero")
        poses = data.geom_xpos if data is not None else model.geom_pos
        spheres = np.stack([
            np.r_[poses[geom_id], model.geom_size[geom_id, 0]]
            for _, geom_id, _ in matches
        ])
        return cls(spheres, names=tuple(name for _, _, name in matches))

    def padded(self, max_spheres: int) -> tuple[np.ndarray, np.ndarray]:
        """Return padded ``[max_spheres, 4]`` features and boolean mask."""
        if max_spheres < len(self.spheres):
            raise ValueError("max_spheres is smaller than this scene")
        features = np.zeros((max_spheres, 4), dtype=np.float32)
        mask = np.zeros(max_spheres, dtype=bool)
        features[:len(self.spheres)] = self.spheres
        mask[:len(self.spheres)] = self.mask
        return features, mask


def sample_scene_variants(
    base: SphereScene,
    count: int,
    *,
    seed: int = 0,
    center_jitter: float = 0.04,
    radius_jitter: float = 0.015,
) -> list[SphereScene]:
    """Keep the first scene exact; perturb the subsequent scenes reproducibly."""
    if count < 1 or center_jitter < 0 or radius_jitter < 0:
        raise ValueError("count must be positive and jitters nonnegative")
    rng = np.random.default_rng(seed)
    scenes = [base]
    for _ in range(1, count):
        spheres = base.spheres.copy()
        spheres[:, :3] += rng.uniform(-center_jitter, center_jitter, spheres[:, :3].shape)
        spheres[:, 3] = np.maximum(
            0.005, spheres[:, 3] + rng.uniform(-radius_jitter, radius_jitter, len(spheres))
        )
        scenes.append(SphereScene(spheres, base.mask, base.names))
    return scenes
