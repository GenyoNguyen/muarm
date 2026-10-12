"""Run the existing horizon-one Panda MPPI controller with a CDFlow field."""
from __future__ import annotations

import numpy as np
import torch

from learning.algos.cdf_mppi import CDFMPPIConfig, CDFMPPIPolicy
from learning.cdflow.field import CDFlowField
from learning.cdflow.scene import SphereScene


class CDFlowMPPIPolicy(CDFMPPIPolicy):
    """Use a learned scene-conditioned distance and direction in MPPI costs."""

    def __init__(
        self,
        env,
        field: CDFlowField,
        cfg: CDFMPPIConfig | None = None,
        *,
        mode: str = "regression",
        samples: int = 64,
    ) -> None:
        if mode not in {"regression", "mc"}:
            raise ValueError("CDFlow inference mode must be 'regression' or 'mc'.")
        if samples < 1:
            raise ValueError("CDFlow Monte Carlo sample count must be positive.")
        self.field = field
        self.mode = mode
        self.samples = samples
        super().__init__(env, cfg, require_cdf=False)
        trained_limits = np.asarray(self.field.model.config.joint_limits, dtype=np.float64)
        live_limits = np.asarray(self.env.model.jnt_range[:7], dtype=np.float64)
        if not np.allclose(trained_limits, live_limits, atol=1e-4, rtol=0):
            raise ValueError("CDFlow checkpoint joint limits do not match the live Panda model.")
        robot = self.field.metadata.get("robot")
        if robot is not None and not str(robot).startswith("Franka Panda"):
            raise ValueError("CDFlow checkpoint was not trained for the Franka Panda.")
        if self.field.device != self.device:
            self.device = self.field.device
            self.limits = self.limits.to(self.device)
            self.eye = self.eye.to(self.device)
            self.reset()
        self.scene: SphereScene | None = None

    def start_episode(self) -> None:
        super().start_episode()
        self.scene = SphereScene.from_model(self.env.model, self.env.data)
        if len(self.scene.spheres) > self.field.max_spheres:
            raise ValueError(
                f"CDFlow checkpoint supports {self.field.max_spheres} spheres; "
                f"scene has {len(self.scene.spheres)}."
            )

    def _distance_and_gradient(
        self, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scene is None:
            raise RuntimeError("Call start_episode() before CDFlow planning.")
        return self.field.distance_and_gradient(
            self.scene,
            q.unsqueeze(0),
            mode=self.mode,
            samples=self.samples,
        )


def load_cdflow_mppi(
    env,
    model_path: str,
    *,
    cfg: CDFMPPIConfig | None = None,
    device: str = "cpu",
    mode: str = "regression",
    samples: int = 64,
) -> CDFlowMPPIPolicy:
    """Load a trained CDFlow checkpoint as the MPPI field backend."""
    field = CDFlowField.load(model_path, device=device)
    return CDFlowMPPIPolicy(env, field, cfg, mode=mode, samples=samples)
