"""Small, CPU-only checks for CDFlow density, geometry, and inference."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from learning.cdflow.field import CDFlowField
from learning.cdflow.losses import LossWeights, cdflow_loss
from learning.cdflow.model import CDFlowConfig, CDFlowModel


def _small_model() -> CDFlowModel:
    torch.manual_seed(7)
    return CDFlowModel(CDFlowConfig(
        joint_limits=[[-2.0, 2.0]] * 7, max_spheres=2,
        context_dim=12, scene_width=12, flow_width=16,
        flow_layers=2, distance_widths=(20, 12),
        positional_frequencies=2, ode_steps=2,
    ))


class CDFlowModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._torch_rng_state = torch.random.get_rng_state()
        self.model = _small_model()
        self.q = torch.tensor([
            [0.2, -0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
            [-0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
        ])
        self.scene = torch.tensor([
            [[0.4, 0.0, 0.2, 0.05], [0.5, 0.1, 0.2, 0.03]],
            [[0.4, 0.0, 0.2, 0.05], [0.5, 0.1, 0.2, 0.03]],
        ])
        self.mask = torch.ones((2, 2), dtype=torch.bool)

    def tearDown(self) -> None:
        torch.random.set_rng_state(self._torch_rng_state)

    def test_flow_reverse_and_bounded_samples(self) -> None:
        condition = self.model.condition(self.q, self.scene, self.mask)
        latent = torch.randn(2, 7)
        data = self.model.flow.to_data(latent, condition)
        recovered, reverse_logdet = self.model.flow.to_latent(
            data, condition, create_graph=False,
        )
        self.assertTrue(torch.allclose(recovered, latent, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.isfinite(reverse_logdet).all())
        physical = self.model.sample_boundary(self.q, self.scene, self.mask, 3)
        self.assertEqual(physical.shape, (2, 3, 7))
        self.assertTrue(torch.all(physical >= -2.0))
        self.assertTrue(torch.all(physical <= 2.0))

    def test_masked_permutation_and_joint_jacobian(self) -> None:
        original = self.model.distance(self.q, self.scene, self.mask)
        reordered = self.model.distance(
            self.q, self.scene.flip(1), self.mask.flip(1),
        )
        self.assertTrue(torch.allclose(original, reordered, atol=1e-6))
        x = torch.tensor([[0.0] * 7, [1.0] * 7], requires_grad=True)
        expected = (self.model.joint_half_range.log()
                    + torch.log1p(-x.tanh().square())).sum(dim=-1)
        self.assertTrue(torch.allclose(
            self.model.log_abs_det_constrain(x), expected, atol=1e-6,
        ))

    def test_all_five_losses_backpropagate_and_checkpoint_loads(self) -> None:
        boundary = torch.stack((
            self.q[0] - 0.1, self.q[0] + 0.1, self.q[1] - 0.1,
        ))
        owner = torch.tensor([0, 0, 1])
        displacement = self.q[owner] - boundary
        norms = displacement.norm(dim=-1)
        y_d = torch.stack((norms[:2].mean(), norms[2]))
        y_g = torch.stack((
            (displacement[:2] / norms[:2, None]).mean(dim=0),
            displacement[2] / norms[2],
        ))
        terms = cdflow_loss(
            self.model, self.q, self.scene, self.mask,
            boundary, owner, y_d, y_g,
            weights=LossWeights(), hessian_samples=1,
        )
        for value in terms.values():
            self.assertTrue(torch.isfinite(value).all())
        terms["total"].backward()
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in self.model.flow.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in self.model.distance_head.parameters()
        ))
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "small.pt"
            torch.save({
                "format": "muarm-cdflow-v1",
                "model_config": asdict(self.model.config),
                "model_state": self.model.state_dict(),
            }, checkpoint)
            field = CDFlowField.load(checkpoint)
            distance, gradient = field.distance_and_gradient(
                (self.scene[0], self.mask[0]), self.q,
                mode="regression",
            )
            mc_distance, mc_gradient = field.distance_and_gradient(
                (self.scene[0], self.mask[0]), self.q,
                mode="mc", samples=4,
            )
            self.assertEqual(distance.shape, (2,))
            self.assertEqual(gradient.shape, (2, 7))
            self.assertEqual(mc_distance.shape, (2,))
            self.assertEqual(mc_gradient.shape, (2, 7))
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue(torch.isfinite(mc_gradient).all())


if __name__ == "__main__":
    unittest.main()
