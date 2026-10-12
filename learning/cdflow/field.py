"""Scene-conditioned CDFlow distance and boundary-direction inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from learning.cdflow.model import CDFlowConfig, CDFlowModel


class CDFlowField:
    """Physical-radian distance and outward direction for a sphere scene."""

    def __init__(
        self, model: CDFlowModel, device: str | torch.device = "cpu",
        metadata: dict[str, Any] | None = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.max_spheres = model.config.max_spheres
        self.scene_schema = {
            "representation": "sphere_xyz_radius",
            "max_spheres": self.max_spheres,
            "position_unit": "meter",
            "joint_unit": "radian",
        }
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "CDFlowField":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "muarm-cdflow-v1":
            raise ValueError("not a muarm CDFlow v1 checkpoint")
        config = CDFlowConfig(**checkpoint["model_config"])
        model = CDFlowModel(config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        return cls(model, device=device, metadata=checkpoint.get("dataset_metadata"))

    def _q_tensor(self, q: torch.Tensor | Any) -> torch.Tensor:
        result = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        if result.ndim == 1:
            result = result.unsqueeze(0)
        if result.ndim != 2 or result.shape[1] != 7:
            raise ValueError("q must be [7] or [batch,7]")
        return result

    def _scene_tensors(self, scene: Any, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(scene, "padded"):
            features, mask = scene.padded(self.max_spheres)
        elif isinstance(scene, tuple) and len(scene) == 2:
            features, mask = scene
        else:
            features = scene
            mask = None
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3 or features.shape[-1] != 4:
            raise ValueError("scene must contain sphere features [batch,spheres,4]")
        if mask is None:
            mask_tensor = torch.ones(features.shape[:2], dtype=torch.bool, device=self.device)
        else:
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            if mask_tensor.ndim == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            if mask_tensor.shape != features.shape[:2]:
                raise ValueError("scene mask must match sphere feature dimensions")
        if features.shape[1] > self.max_spheres:
            raise ValueError(
                f"checkpoint supports at most {self.max_spheres} spheres; "
                f"scene has {features.shape[1]}"
            )
        if features.shape[1] < self.max_spheres:
            pad_count = self.max_spheres - features.shape[1]
            features = torch.nn.functional.pad(features, (0, 0, 0, pad_count))
            mask_tensor = torch.nn.functional.pad(mask_tensor, (0, pad_count))
        if features.shape[0] == 1:
            features = features.expand(batch_size, -1, -1)
            mask_tensor = mask_tensor.expand(batch_size, -1)
        elif features.shape[0] != batch_size:
            raise ValueError("scene batch must be one or match q batch")
        if not torch.isfinite(features).all():
            raise ValueError("scene contains nonfinite sphere geometry")
        return features, mask_tensor

    def sample_boundary(
        self, scene: Any, q: torch.Tensor | Any, *, samples: int = 64,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        q_tensor = self._q_tensor(q)
        scene_tensor, scene_mask = self._scene_tensors(scene, len(q_tensor))
        with torch.no_grad():
            return self.model.sample_boundary(
                q_tensor, scene_tensor, scene_mask, samples, generator=generator,
            )

    def distance_and_gradient(
        self,
        scene: Any,
        q: torch.Tensor | Any,
        mode: str = "mc",
        samples: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return distance [N] and outward joint direction [N,7].

        ``mode='mc'`` computes the paper's expected distance and expected
        normalized displacement from CNF samples. It is not the full
        derivative of expected distance because the conditional distribution
        itself varies with q.
        """
        q_tensor = self._q_tensor(q)
        scene_tensor, scene_mask = self._scene_tensors(scene, len(q_tensor))
        if mode == "regression":
            with torch.enable_grad():
                q_for_grad = q_tensor.detach().clone().requires_grad_(True)
                distance = self.model.distance(q_for_grad, scene_tensor, scene_mask)
                gradient = torch.autograd.grad(distance.sum(), q_for_grad)[0]
            return distance.detach(), gradient.detach()
        if mode == "mc":
            with torch.no_grad():
                boundary = self.model.sample_boundary(
                    q_tensor, scene_tensor, scene_mask, samples,
                )
                displacement = q_tensor[:, None, :] - boundary
                norms = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1e-8)
                distance = norms.mean(dim=1)
                gradient = (displacement / norms.unsqueeze(-1)).mean(dim=1)
            return distance, gradient
        raise ValueError("mode must be 'regression' or 'mc'")
