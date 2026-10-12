"""Inference-only configuration-space distance field for Franka Panda.

Adapted from https://github.com/yimingli1998/cdf (MIT license).
"""
from __future__ import annotations

from pathlib import Path

import torch

from learning.cdf.mlp import MLPRegression


_MODEL_PATH = Path(__file__).with_name("model_dict.pt")
_CHECKPOINT = 49_900


class CDF:
    """Load the pretrained CDF and evaluate distance and joint gradients."""

    def __init__(self, device: str = "cpu", model_path: str | Path = _MODEL_PATH):
        self.device = torch.device(device)
        self.model = MLPRegression().to(self.device)

        state = torch.load(model_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and all(isinstance(key, int) for key in state):
            state = state[_CHECKPOINT]
        self.model.load_state_dict(state)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def inference(self, points: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Return CDF distances for every obstacle-point/configuration pair."""
        points = points.to(self.device)
        q = q.to(self.device)
        points_batch = points.unsqueeze(1).expand(-1, len(q), -1).reshape(-1, 3)
        q_batch = q.unsqueeze(0).expand(len(points), -1, -1).reshape(-1, 7)
        inputs = torch.cat((points_batch, q_batch), dim=-1)
        return self.model(inputs).reshape(len(points), len(q))

    def distance_and_gradient(
        self,
        points: torch.Tensor,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nearest-point CDF distance and its gradient with respect to q."""
        with torch.enable_grad():
            q = q.detach().clone().to(self.device).requires_grad_(True)
            distance = self.inference(points, q).min(dim=0).values
            gradient = torch.autograd.grad(distance.sum(), q)[0]
        return distance.detach(), gradient.detach()
