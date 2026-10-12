#!/usr/bin/env python3
"""Train a Panda CDFlow CNF and distance head from adaptive boundary sets.

Run from the repository root, for example::

    .venv/bin/python scripts/train_cdflow.py --data data/cdflow.npz \
        --out learning/runs/panda_cdflow.pt --device cuda

The saved checkpoint includes the model architecture, data metadata, optimizer
state, explicit solver/noise/loss settings, and held-out validation metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

# Direct `python scripts/train_cdflow.py` puts scripts/, not the repository, on
# sys.path. Keep the command usable without an editable package installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from learning.cdflow.losses import LossWeights, cdflow_loss
from learning.cdflow.model import CDFlowConfig, CDFlowModel


class BoundaryDataset:
    """Validated, scene-split NPZ with offset-indexed boundary target sets."""

    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as archive:
            required = (
                "q", "scene", "scene_mask", "boundary_q", "offsets",
                "y_d", "y_g", "joint_limits", "scene_id", "split", "metadata_json",
            )
            missing = [key for key in required if key not in archive]
            if missing:
                raise ValueError(f"CDFlow dataset lacks keys: {missing}")
            self.q = np.asarray(archive["q"], dtype=np.float32)
            self.scene = np.asarray(archive["scene"], dtype=np.float32)
            self.scene_mask = np.asarray(archive["scene_mask"], dtype=bool)
            self.boundary_q = np.asarray(archive["boundary_q"], dtype=np.float32)
            self.offsets = np.asarray(archive["offsets"], dtype=np.int64)
            self.y_d = np.asarray(archive["y_d"], dtype=np.float32)
            self.y_g = np.asarray(archive["y_g"], dtype=np.float32)
            self.joint_limits = np.asarray(archive["joint_limits"], dtype=np.float32)
            self.scene_id = np.asarray(archive["scene_id"], dtype=np.int64)
            self.split = np.asarray(archive["split"], dtype=np.int8)
            self.metadata = json.loads(str(archive["metadata_json"].item()))
        count = len(self.q)
        if (
            self.q.shape != (count, 7)
            or self.scene.ndim != 3 or self.scene.shape[0] != count
            or self.scene.shape[-1] != 4
            or self.scene_mask.shape != self.scene.shape[:2]
            or self.boundary_q.ndim != 2 or self.boundary_q.shape[-1] != 7
            or self.offsets.shape != (count + 1,)
            or self.y_d.shape != (count,) or self.y_g.shape != (count, 7)
            or self.joint_limits.shape != (7, 2)
            or self.scene_id.shape != (count,) or self.split.shape != (count,)
        ):
            raise ValueError("CDFlow dataset array shapes are inconsistent")
        if not count or self.offsets[0] != 0 or self.offsets[-1] != len(self.boundary_q):
            raise ValueError("CDFlow dataset is empty or has invalid target offsets")
        if np.any(np.diff(self.offsets) <= 0):
            raise ValueError("each query must have at least one boundary target")
        if not np.all(np.isin(self.split, (0, 1, 2))):
            raise ValueError("split labels must be 0=train, 1=validation, 2=test")
        if not all(np.all(np.isfinite(array)) for array in (
            self.q, self.scene, self.boundary_q, self.y_d, self.y_g, self.joint_limits,
        )):
            raise ValueError("CDFlow dataset contains nonfinite geometry or labels")
        for scene_id in np.unique(self.scene_id):
            if len(np.unique(self.split[self.scene_id == scene_id])) != 1:
                raise ValueError("scene_id appears in multiple splits")

    def rows(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def batch(
        self,
        rows: np.ndarray,
        *,
        device: torch.device,
        targets_per_query: int,
        rng: np.random.Generator,
    ) -> dict[str, torch.Tensor]:
        targets: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        for owner, row in enumerate(rows):
            target_indices = np.arange(self.offsets[row], self.offsets[row + 1])
            if targets_per_query > 0 and len(target_indices) > targets_per_query:
                target_indices = rng.choice(
                    target_indices, size=targets_per_query, replace=False,
                )
            targets.append(self.boundary_q[target_indices])
            owners.append(np.full(len(target_indices), owner, dtype=np.int64))

        def tensor(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(values, dtype=dtype, device=device)

        return {
            "q": tensor(self.q[rows], torch.float32),
            "scene": tensor(self.scene[rows], torch.float32),
            "scene_mask": tensor(self.scene_mask[rows], torch.bool),
            "boundary_q": tensor(np.concatenate(targets), torch.float32),
            "owner": tensor(np.concatenate(owners), torch.long),
            "y_d": tensor(self.y_d[rows], torch.float32),
            "y_g": tensor(self.y_g[rows], torch.float32),
        }


def _mean_metrics(records: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    total_rows = sum(rows for rows, _ in records)
    if not total_rows:
        return {}
    return {
        key: sum(rows * values[key] for rows, values in records) / total_rows
        for key in records[0][1]
    }


def _epoch(
    model: CDFlowModel,
    dataset: BoundaryDataset,
    rows: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    targets_per_query: int,
    weights: LossWeights,
    noise_std: float,
    hessian_samples: int,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        rows = rng.permutation(rows)
    records: list[tuple[int, dict[str, float]]] = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        batch = dataset.batch(
            selected, device=device,
            targets_per_query=targets_per_query if training else 0,
            rng=rng,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        # First and second joint derivatives are needed even in validation.
        with torch.enable_grad():
            terms = cdflow_loss(
                model, **batch, weights=weights, noise_std=noise_std,
                hessian_samples=hessian_samples, training=training,
            )
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError("CDFlow loss became nonfinite")
            if training:
                terms["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        records.append((len(selected), {
            name: float(value.detach().cpu()) for name, value in terms.items()
        }))
    return _mean_metrics(records)


def train_cdflow(
    data_path: str | Path,
    out_path: str | Path,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 0,
    learning_rate: float = 1e-3,
    lr_halving_interval: int = 25,
    patience: int = 20,
    validation_interval: int = 1,
    targets_per_query: int = 4,
    target_noise_std: float = 0.02,
    hessian_samples: int = 2,
    grad_clip: float = 10.0,
    context_dim: int = 128,
    scene_width: int = 128,
    flow_width: int = 1024,
    flow_layers: int = 4,
    distance_widths: tuple[int, ...] = (1024, 512, 256, 128, 128),
    positional_frequencies: int = 4,
    ode_solver: str = "rk4",
    ode_steps: int = 4,
    ode_rtol: float = 1e-4,
    ode_atol: float = 1e-5,
    weights: LossWeights | None = None,
) -> str:
    """Train and save the best held-out-scene checkpoint; return its path."""
    if min(epochs, batch_size, lr_halving_interval, validation_interval) <= 0:
        raise ValueError("epoch, batch, decay, and validation counts must be positive")
    if patience < 0 or targets_per_query < 0 or hessian_samples < 0:
        raise ValueError("patience, target count, and Hessian count must be nonnegative")
    if target_noise_std < 0 or grad_clip <= 0 or learning_rate <= 0:
        raise ValueError("noise, clipping, and learning rate settings are invalid")

    dataset = BoundaryDataset(data_path)
    train_rows = dataset.rows(0)
    validation_rows = dataset.rows(1)
    if not len(train_rows):
        raise ValueError("dataset has no training scenes")
    if not len(validation_rows):
        raise ValueError(
            "dataset has no held-out validation scenes; generate at least two scenes"
        )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    config = CDFlowConfig(
        joint_limits=dataset.joint_limits.tolist(),
        max_spheres=dataset.scene.shape[1],
        context_dim=context_dim, scene_width=scene_width,
        flow_width=flow_width, flow_layers=flow_layers,
        distance_widths=distance_widths,
        positional_frequencies=positional_frequencies,
        ode_solver=ode_solver, ode_steps=ode_steps,
        ode_rtol=ode_rtol, ode_atol=ode_atol,
    )
    model = CDFlowModel(config).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_halving_interval, gamma=0.5,
    )
    weights = weights or LossWeights()
    settings: dict[str, Any] = {
        "epochs": epochs, "batch_size": batch_size, "seed": seed,
        "learning_rate": learning_rate,
        "lr_halving_interval": lr_halving_interval,
        "patience": patience, "validation_interval": validation_interval,
        "targets_per_query": targets_per_query,
        "target_noise_std_unconstrained": target_noise_std,
        "hessian_samples": hessian_samples,
        "grad_clip": grad_clip, "loss_weights": asdict(weights),
    }
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    stagnant_validations = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(
            model, dataset, train_rows, batch_size=batch_size,
            device=target_device, rng=rng, targets_per_query=targets_per_query,
            weights=weights, noise_std=target_noise_std,
            hessian_samples=hessian_samples, optimizer=optimizer,
            grad_clip=grad_clip,
        )
        scheduler.step()
        should_validate = (
            epoch % validation_interval == 0 or epoch == epochs
        )
        if not should_validate:
            continue
        validation_metrics = _epoch(
            model, dataset, validation_rows, batch_size=batch_size,
            device=target_device, rng=rng, targets_per_query=0,
            weights=weights, noise_std=0.0,
            hessian_samples=hessian_samples, optimizer=None,
            grad_clip=grad_clip,
        )
        score = validation_metrics["total"]
        report = {
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics, "validation": validation_metrics,
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            stagnant_validations = 0
            checkpoint = {
                "format": "muarm-cdflow-v1",
                "model_config": model.checkpoint_config(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "training_settings": settings,
                "dataset_metadata": dataset.metadata,
                "metrics": report,
                "epoch": epoch,
            }
            temporary = out_path.with_suffix(out_path.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, out_path)
        else:
            stagnant_validations += 1
        if patience and stagnant_validations >= patience:
            print(f"Stopped after {epoch} epochs; best epoch was {best_epoch}", flush=True)
            break
    return str(out_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="generated CDFlow NPZ")
    parser.add_argument("--out", type=Path, required=True, help="checkpoint .pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-halving-interval", type=int, default=25)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--targets-per-query", type=int, default=4,
                        help="0 uses all boundary targets each update")
    parser.add_argument("--target-noise-std", type=float, default=0.02,
                        help="Gaussian standard deviation in atanh joint coordinates")
    parser.add_argument("--hessian-samples", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--scene-width", type=int, default=128)
    parser.add_argument("--flow-width", type=int, default=1024)
    parser.add_argument("--flow-layers", type=int, default=4)
    parser.add_argument("--distance-widths", default="1024,512,256,128,128",
                        help="comma-separated hidden widths")
    parser.add_argument("--positional-frequencies", type=int, default=4)
    parser.add_argument("--ode-solver", default="rk4")
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument("--ode-rtol", type=float, default=1e-4)
    parser.add_argument("--ode-atol", type=float, default=1e-5)
    parser.add_argument("--lambda-nll", type=float, default=1.0)
    parser.add_argument("--lambda-distance", type=float, default=10.0)
    parser.add_argument("--lambda-gradient", type=float, default=1.0)
    parser.add_argument("--lambda-eikonal", type=float, default=0.1)
    parser.add_argument("--lambda-tension", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    widths = tuple(int(value) for value in args.distance_widths.split(","))
    path = train_cdflow(
        args.data, args.out,
        epochs=args.epochs, batch_size=args.batch_size, device=args.device,
        seed=args.seed, learning_rate=args.learning_rate,
        lr_halving_interval=args.lr_halving_interval,
        patience=args.patience, validation_interval=args.validation_interval,
        targets_per_query=args.targets_per_query,
        target_noise_std=args.target_noise_std,
        hessian_samples=args.hessian_samples, grad_clip=args.grad_clip,
        context_dim=args.context_dim, scene_width=args.scene_width,
        flow_width=args.flow_width, flow_layers=args.flow_layers,
        distance_widths=widths,
        positional_frequencies=args.positional_frequencies,
        ode_solver=args.ode_solver, ode_steps=args.ode_steps,
        ode_rtol=args.ode_rtol, ode_atol=args.ode_atol,
        weights=LossWeights(
            nll=args.lambda_nll, distance=args.lambda_distance,
            gradient=args.lambda_gradient, eikonal=args.lambda_eikonal,
            tension=args.lambda_tension,
        ),
    )
    print(f"Saved CDFlow checkpoint: {path}")


if __name__ == "__main__":
    main()
