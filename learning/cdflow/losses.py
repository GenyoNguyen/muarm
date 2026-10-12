"""Five losses from CDFlow Equations (7) and (10)--(14)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from learning.cdflow.model import CDFlowModel


@dataclass
class LossWeights:
    nll: float = 1.0
    distance: float = 10.0
    gradient: float = 1.0
    eikonal: float = 0.1
    tension: float = 1e-4

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("loss weights must be nonnegative")


def cdflow_loss(
    model: CDFlowModel,
    q: torch.Tensor,
    scene: torch.Tensor,
    scene_mask: torch.Tensor,
    boundary_q: torch.Tensor,
    owner: torch.Tensor,
    y_d: torch.Tensor,
    y_g: torch.Tensor,
    *,
    weights: LossWeights,
    noise_std: float = 0.02,
    hessian_samples: int = 2,
    training: bool = True,
    cosine_epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compute a query-averaged objective while retaining every boundary mode.

    ``owner`` maps flattened boundary targets to their query row. The NLL is
    first averaged over each query's targets, then over queries; a query with
    many refined modes therefore does not dominate the batch.
    """
    if q.ndim != 2 or q.shape[-1] != 7 or boundary_q.ndim != 2 or boundary_q.shape[-1] != 7:
        raise ValueError("q and boundary_q must have shape [batch,7] and [targets,7]")
    if owner.ndim != 1 or len(owner) != len(boundary_q):
        raise ValueError("owner must have one index per boundary target")
    if not len(q) or not len(boundary_q):
        raise ValueError("the batch and its boundary targets must be nonempty")
    if owner.min() < 0 or owner.max() >= len(q):
        raise ValueError("owner index is outside the query batch")
    if hessian_samples < 0:
        raise ValueError("hessian_samples must be nonnegative")

    q_for_grad = q.detach().requires_grad_(True)
    predicted_distance = model.distance(q_for_grad, scene, scene_mask)
    predicted_gradient = torch.autograd.grad(
        predicted_distance.sum(), q_for_grad, create_graph=True,
    )[0]
    distance_loss = (predicted_distance - y_d).square().mean()

    gradient_norm = torch.linalg.vector_norm(predicted_gradient, dim=-1)
    label_norm = torch.linalg.vector_norm(y_g, dim=-1)
    valid_direction = label_norm > cosine_epsilon
    cosine = (predicted_gradient * y_g).sum(dim=-1) / (
        gradient_norm.clamp_min(cosine_epsilon)
        * label_norm.clamp_min(cosine_epsilon)
    )
    gradient_loss = torch.where(
        valid_direction, 1 - cosine.clamp(-1, 1), torch.zeros_like(cosine),
    ).sum() / valid_direction.sum().clamp_min(1)
    eikonal_loss = (gradient_norm - 1).abs().mean()

    tension_loss = predicted_distance.sum() * 0
    if weights.tension > 0 and hessian_samples > 0:
        chosen = min(hessian_samples, len(q))
        rows = []
        for axis in range(7):
            second = torch.autograd.grad(
                predicted_gradient[:chosen, axis].sum(), q_for_grad,
                create_graph=training, retain_graph=True,
            )[0][:chosen]
            rows.append(second)
        hessian = torch.stack(rows, dim=-2)
        tension_loss = hessian.square().sum(dim=(-1, -2)).mean()

    log_density = model.log_prob(
        boundary_q, q, scene, scene_mask,
        owner=owner, noise_std=noise_std if training else 0.0,
        create_graph=training,
    )
    owner = owner.to(device=q.device, dtype=torch.long)
    per_query_nll = q.new_zeros(len(q)).scatter_add(0, owner, -log_density)
    counts = q.new_zeros(len(q)).scatter_add(0, owner, torch.ones_like(log_density))
    if torch.any(counts == 0):
        raise ValueError("every query must have at least one boundary target")
    nll_loss = (per_query_nll / counts).mean()

    total = (
        weights.nll * nll_loss
        + weights.distance * distance_loss
        + weights.gradient * gradient_loss
        + weights.eikonal * eikonal_loss
        + weights.tension * tension_loss
    )
    return {
        "total": total,
        "nll": nll_loss,
        "distance": distance_loss,
        "gradient": gradient_loss,
        "eikonal": eikonal_loss,
        "tension": tension_loss,
        "gradient_valid_fraction": valid_direction.float().mean(),
    }
