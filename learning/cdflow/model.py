"""Unified scene-conditioned CDFlow CNF and joint-space distance regressor."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from learning.cdflow.flow import ConditionalCNF


@dataclass
class CDFlowConfig:
    """Recorded architecture and solver choices for a seven-joint Panda model."""

    joint_limits: list[list[float]]
    max_spheres: int
    context_dim: int = 128
    scene_width: int = 128
    flow_width: int = 1024
    flow_layers: int = 4
    distance_widths: tuple[int, ...] = (1024, 512, 256, 128, 128)
    positional_frequencies: int = 4
    ode_solver: str = "rk4"
    ode_steps: int = 4
    ode_rtol: float = 1e-4
    ode_atol: float = 1e-5
    limit_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if len(self.joint_limits) != 7 or any(len(pair) != 2 for pair in self.joint_limits):
            raise ValueError("joint_limits must have shape [7,2]")
        if self.max_spheres <= 0 or self.positional_frequencies < 0:
            raise ValueError("max_spheres must be positive and frequencies nonnegative")
        if not (0 < self.limit_epsilon < 0.1):
            raise ValueError("limit_epsilon must lie in (0, 0.1)")


class SphereEncoder(nn.Module):
    """Masked mean and max pooling make sphere order irrelevant."""

    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )

    def forward(self, scene: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if scene.ndim != 3 or scene.shape[-1] != 4 or mask.shape != scene.shape[:2]:
            raise ValueError("scene must be [batch,spheres,4] with matching mask")
        features = self.net(scene)
        valid = mask.to(torch.bool)
        count = valid.sum(dim=1, keepdim=True)
        mean = (features * valid.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1)
        maximum = features.masked_fill(~valid.unsqueeze(-1), -torch.inf).max(dim=1).values
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=-1)


class CDFlowModel(nn.Module):
    """Shared condition trunk, conditional CNF, and scalar distance head.

    Inputs and distance labels are physical joint radians. The CNF itself acts
    on ``atanh``-transformed joint coordinates, so samples respect joint limits.
    """

    def __init__(self, config: CDFlowConfig):
        super().__init__()
        self.config = config
        limits = torch.as_tensor(config.joint_limits, dtype=torch.float32)
        if not torch.isfinite(limits).all() or not torch.all(limits[:, 1] > limits[:, 0]):
            raise ValueError("joint_limits must be finite and strictly increasing")
        self.register_buffer("joint_center", limits.mean(dim=1))
        self.register_buffer("joint_half_range", (limits[:, 1] - limits[:, 0]) / 2)
        self.scene_encoder = SphereEncoder(config.scene_width)
        self.condition_trunk = nn.Sequential(
            nn.Linear(2 * config.scene_width + 7, config.context_dim), nn.SiLU(),
            nn.Linear(config.context_dim, config.context_dim), nn.SiLU(),
        )
        self.flow = ConditionalCNF(
            dim=7, context_dim=config.context_dim,
            width=config.flow_width, hidden_layers=config.flow_layers,
            solver=config.ode_solver, steps=config.ode_steps,
            rtol=config.ode_rtol, atol=config.ode_atol,
        )
        encoded_dim = 7 * (1 + 2 * config.positional_frequencies)
        widths = (encoded_dim + config.context_dim, *config.distance_widths, 1)
        layers: list[nn.Module] = []
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(widths[index], widths[index + 1]))
            if index < len(widths) - 2:
                layers.append(nn.SiLU())
        self.distance_head = nn.Sequential(*layers)

    def normalized_q(self, q: torch.Tensor) -> torch.Tensor:
        return (q - self.joint_center) / self.joint_half_range

    def condition(
        self, q: torch.Tensor, scene: torch.Tensor, scene_mask: torch.Tensor,
    ) -> torch.Tensor:
        if q.ndim != 2 or q.shape[1] != 7 or q.shape[0] != scene.shape[0]:
            raise ValueError("q must be [batch,7] with matching scene batch")
        if scene.shape[1] != self.config.max_spheres:
            raise ValueError(
                f"checkpoint expects {self.config.max_spheres} padded spheres, "
                f"received {scene.shape[1]}"
            )
        return self.condition_trunk(
            torch.cat((self.normalized_q(q), self.scene_encoder(scene, scene_mask)), dim=-1)
        )

    def _positional_q(self, q: torch.Tensor) -> torch.Tensor:
        normalized = self.normalized_q(q)
        terms = [normalized]
        for level in range(self.config.positional_frequencies):
            phase = normalized * (math.pi * 2**level)
            terms.extend((phase.sin(), phase.cos()))
        return torch.cat(terms, dim=-1)

    def distance(
        self, q: torch.Tensor, scene: torch.Tensor, scene_mask: torch.Tensor,
    ) -> torch.Tensor:
        context = self.condition(q, scene, scene_mask)
        raw = self.distance_head(torch.cat((context, self._positional_q(q)), dim=-1))
        return F.softplus(raw).squeeze(-1)

    def unconstrain(self, boundary_q: torch.Tensor) -> torch.Tensor:
        normalized = self.normalized_q(boundary_q)
        eps = self.config.limit_epsilon
        return torch.atanh(normalized.clamp(-1 + eps, 1 - eps))

    def constrain(self, unconstrained: torch.Tensor) -> torch.Tensor:
        return self.joint_center + self.joint_half_range * unconstrained.tanh()

    def log_abs_det_constrain(self, unconstrained: torch.Tensor) -> torch.Tensor:
        # log(sech(x)^2) = 2 * (log(2) - x - softplus(-2x)).
        log_sech2 = 2 * (
            math.log(2.0) - unconstrained - F.softplus(-2 * unconstrained)
        )
        return (self.joint_half_range.log() + log_sech2).sum(dim=-1)

    def log_prob(
        self,
        boundary_q: torch.Tensor,
        q: torch.Tensor,
        scene: torch.Tensor,
        scene_mask: torch.Tensor,
        *,
        owner: torch.Tensor | None = None,
        noise_std: float = 0.0,
        create_graph: bool | None = None,
    ) -> torch.Tensor:
        """Conditional physical-joint density of each boundary target.

        ``owner[k]`` selects the query/scene for target ``boundary_q[k]``.
        Gaussian training noise is added in unconstrained coordinates; the
        Jacobian is evaluated at the resulting bounded target.
        """
        if noise_std < 0:
            raise ValueError("noise_std must be nonnegative")
        context = self.condition(q, scene, scene_mask)
        if owner is not None:
            context = context.index_select(0, owner.to(device=q.device, dtype=torch.long))
        if context.shape[0] != boundary_q.shape[0]:
            raise ValueError("boundary targets need one owner per target")
        x = self.unconstrain(boundary_q)
        if noise_std:
            x = x + torch.randn_like(x) * noise_std
        latent, inverse_logdet = self.flow.to_latent(x, context, create_graph=create_graph)
        return (
            self.flow.standard_normal_log_prob(latent)
            + inverse_logdet
            - self.log_abs_det_constrain(x)
        )

    def sample_boundary(
        self,
        q: torch.Tensor,
        scene: torch.Tensor,
        scene_mask: torch.Tensor,
        samples: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return bounded collision-boundary candidates [batch,samples,7]."""
        if samples <= 0:
            raise ValueError("samples must be positive")
        context = self.condition(q, scene, scene_mask)
        context = context.repeat_interleave(samples, dim=0)
        latent = torch.randn(
            (q.shape[0] * samples, 7), dtype=q.dtype, device=q.device,
            generator=generator,
        )
        x = self.flow.to_data(latent, context)
        return self.constrain(x).reshape(q.shape[0], samples, 7)

    def checkpoint_config(self) -> dict:
        return asdict(self.config)
