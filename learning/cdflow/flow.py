"""Conditional continuous normalizing flow for Panda boundary configurations.

The ODE lives in unconstrained coordinates. ``CDFlowModel`` owns the invertible
joint-limit transform and its density Jacobian.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torchdiffeq import odeint


class FlowVelocity(nn.Module):
    """Time-, latent-, and query/scene-conditioned CNF velocity."""

    def __init__(self, dim: int, context_dim: int, width: int, hidden_layers: int):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        layers: list[nn.Module] = []
        channels = dim + context_dim + 1
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(channels, width), nn.ReLU()))
            channels = width
        layers.append(nn.Linear(channels, dim))
        self.net = nn.Sequential(*layers)

        # Start near the identity map; this also makes early likelihoods stable.
        nn.init.normal_(layers[-1].weight, std=1e-3)
        nn.init.zeros_(layers[-1].bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        time = torch.ones_like(x[:, :1]) * t
        return self.net(torch.cat((x, time, context), dim=-1))


class ConditionalCNF(nn.Module):
    """Integrate a conditional flow and, in reverse, its exact log-Jacobian.

    ``to_latent`` returns ``logdet`` = log |det(d x_data / d z_prior)|^-1.
    Thus ``log p(x_data) = log p(z_prior) + logdet``.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        width: int = 1024,
        hidden_layers: int = 4,
        solver: str = "rk4",
        steps: int = 4,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ):
        super().__init__()
        if dim <= 0 or steps <= 0:
            raise ValueError("dim and steps must be positive")
        self.dim = dim
        self.solver = solver
        self.steps = steps
        self.rtol = rtol
        self.atol = atol
        self.velocity = FlowVelocity(dim, context_dim, width, hidden_layers)

    def _integrate(self, func, state, times: torch.Tensor):
        options = {"step_size": 1.0 / self.steps} if self.solver in {"euler", "midpoint", "rk4"} else None
        return odeint(
            func, state, times, method=self.solver, options=options,
            rtol=self.rtol, atol=self.atol,
        )

    def to_data(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.dim or context.shape[0] != z.shape[0]:
            raise ValueError("z and context must be batches with the same length")
        times = z.new_tensor((0.0, 1.0))

        def dynamics(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.velocity(t, x, context)

        return self._integrate(dynamics, z, times)[-1]

    def to_latent(
        self, x: torch.Tensor, context: torch.Tensor, *, create_graph: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map data to Gaussian latent and integrate exact seven-dimensional trace."""
        if x.ndim != 2 or x.shape[1] != self.dim or context.shape[0] != x.shape[0]:
            raise ValueError("x and context must be batches with the same length")
        if create_graph is None:
            create_graph = torch.is_grad_enabled()
        times = x.new_tensor((1.0, 0.0))
        initial_logdet = x.new_zeros(x.shape[0])

        def augmented(t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
            position, _ = state
            # Exact trace needs autograd even when validating under no_grad.
            with torch.enable_grad():
                position = position.requires_grad_(True)
                velocity = self.velocity(t, position, context)
                diagonal = []
                for axis in range(self.dim):
                    derivative = torch.autograd.grad(
                        velocity[:, axis].sum(), position,
                        create_graph=create_graph, retain_graph=True,
                    )[0]
                    diagonal.append(derivative[:, axis])
                divergence = torch.stack(diagonal, dim=-1).sum(dim=-1)
            return velocity, divergence

        latent_path, logdet_path = self._integrate(
            augmented, (x, initial_logdet), times,
        )
        return latent_path[-1], logdet_path[-1]

    @staticmethod
    def standard_normal_log_prob(z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (z.square() + math.log(2.0 * math.pi)).sum(dim=-1)
