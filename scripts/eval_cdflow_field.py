#!/usr/bin/env python3
"""Evaluate a CDFlow field on scene-held-out Panda boundary queries.

The distance and direction labels are joint-space quantities (radians). The
boundary residuals below are independently recomputed MuJoCo sphere gaps in
metres. A one-epoch smoke checkpoint is useful for checking this command, but
its accuracy numbers are not evidence of a trained model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.cdflow.collision import PandaCollisionOracle
from learning.cdflow.field import CDFlowField
from learning.cdflow.scene import SphereScene
from learning.robots.panda import _build_model
from scripts.train_cdflow import BoundaryDataset


SPLITS = {"train": 0, "validation": 1, "test": 2}
SCHEMA_KEYS = (
    "version", "robot", "scene_schema", "mask_schema", "gap_units",
    "direction_schema", "split_schema",
)


def _check_metadata(field: CDFlowField, dataset: BoundaryDataset, task: str | None) -> tuple[str, bool]:
    checkpoint_meta = field.metadata
    dataset_meta = dataset.metadata
    if not isinstance(checkpoint_meta, dict) or not checkpoint_meta:
        raise ValueError("checkpoint has no dataset metadata to verify")
    if not isinstance(dataset_meta, dict):
        raise ValueError("dataset metadata is missing or malformed")
    for key in SCHEMA_KEYS:
        if key not in checkpoint_meta or key not in dataset_meta:
            raise ValueError(f"checkpoint or dataset lacks required {key!r} metadata")
        if checkpoint_meta[key] != dataset_meta[key]:
            raise ValueError(f"checkpoint and dataset {key!r} metadata differ")
    for key in ("task", "single_sphere_index"):
        if key in checkpoint_meta and key in dataset_meta:
            if checkpoint_meta[key] != dataset_meta[key]:
                raise ValueError(f"checkpoint and dataset {key!r} metadata differ")
    if dataset.scene.shape[1] != field.max_spheres:
        raise ValueError(
            f"checkpoint expects {field.max_spheres} scene slots, "
            f"dataset has {dataset.scene.shape[1]}"
        )
    checkpoint_limits = np.asarray(field.model.config.joint_limits, dtype=np.float64)
    if not np.allclose(checkpoint_limits, dataset.joint_limits, atol=1e-5, rtol=0):
        raise ValueError("checkpoint and dataset Panda joint limits differ")
    recorded_task = dataset_meta.get("task") or checkpoint_meta.get("task")
    if task is None:
        task = recorded_task
    if task is None:
        raise ValueError("dataset has no task metadata; supply --task for MuJoCo validation")
    if recorded_task is not None and task != recorded_task:
        raise ValueError(f"requested task {task!r} differs from dataset task {recorded_task!r}")
    if task not in {"reach_obstacles", "reach_obstacles_2"}:
        raise ValueError("field evaluation requires a Panda obstacle-reaching task")
    return task, checkpoint_meta == dataset_meta


def _direction_cosines(predicted: np.ndarray, reference: np.ndarray) -> tuple[float | None, int]:
    predicted_norm = np.linalg.norm(predicted, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    usable = (np.isfinite(predicted_norm) & np.isfinite(reference_norm)
              & (predicted_norm > 1e-8) & (reference_norm > 1e-8))
    if not np.any(usable):
        return None, 0
    cosine = np.sum(predicted[usable] * reference[usable], axis=1)
    cosine /= predicted_norm[usable] * reference_norm[usable]
    return float(np.mean(np.clip(cosine, -1, 1))), int(usable.sum())


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _field_pass(
    field: CDFlowField,
    dataset: BoundaryDataset,
    rows: np.ndarray,
    *,
    mode: str,
    batch_size: int,
    samples: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    distances: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    elapsed = 0.0
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        scene = (dataset.scene[selected], dataset.scene_mask[selected])
        _synchronize(field.device)
        beginning = time.perf_counter()
        distance, direction = field.distance_and_gradient(
            scene, dataset.q[selected], mode=mode, samples=samples,
        )
        _synchronize(field.device)
        elapsed += time.perf_counter() - beginning
        distances.append(distance.cpu().numpy())
        directions.append(direction.cpu().numpy())
    return np.concatenate(distances), np.concatenate(directions), elapsed


def _boundary_samples(
    field: CDFlowField,
    dataset: BoundaryDataset,
    rows: np.ndarray,
    *,
    batch_size: int,
    samples: int,
) -> tuple[np.ndarray, float]:
    chunks: list[np.ndarray] = []
    elapsed = 0.0
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        scene = (dataset.scene[selected], dataset.scene_mask[selected])
        _synchronize(field.device)
        beginning = time.perf_counter()
        drawn = field.sample_boundary(scene, dataset.q[selected], samples=samples)
        _synchronize(field.device)
        elapsed += time.perf_counter() - beginning
        chunks.append(drawn.cpu().numpy())
    return np.concatenate(chunks), elapsed


def _geometry_metrics(
    oracle: PandaCollisionOracle,
    dataset: BoundaryDataset,
    rows: np.ndarray,
    generated: np.ndarray,
    *,
    tolerance: float,
    scene_names: tuple[str, ...],
) -> dict[str, dict[str, float | int | None]]:
    gaps: dict[str, list[float]] = {"targets": [], "generated": []}
    valid: dict[str, int] = {"targets": 0, "generated": 0}
    in_limits: dict[str, int] = {"targets": 0, "generated": 0}
    counts: dict[str, int] = {"targets": 0, "generated": 0}
    for position, row in enumerate(rows):
        features = dataset.scene[row].astype(np.float64, copy=True)
        mask = dataset.scene_mask[row]
        # Inactive padding may have a zero radius. MuJoCo moves inactive geoms
        # out of the scene, but SphereScene still needs a positive placeholder.
        features[~mask, 3] = np.maximum(features[~mask, 3], 1e-6)
        oracle.set_scene(SphereScene(features, mask=mask, names=scene_names))
        targets = dataset.boundary_q[dataset.offsets[row]:dataset.offsets[row + 1]]
        for kind, poses in (("targets", targets), ("generated", generated[position])):
            for q in poses:
                counts[kind] += 1
                if not oracle.in_limits(q):
                    continue
                in_limits[kind] += 1
                gap = oracle.gap(q)
                if np.isfinite(gap):
                    gaps[kind].append(float(gap))
                    if oracle.boundary_valid(q, tolerance=tolerance):
                        valid[kind] += 1

    result: dict[str, dict[str, float | int | None]] = {}
    for kind in ("targets", "generated"):
        values = np.asarray(gaps[kind], dtype=np.float64)
        count = counts[kind]
        result[kind] = {
            "count": count,
            "in_joint_limits_fraction": in_limits[kind] / count if count else None,
            "finite_gap_count": len(values),
            "mean_abs_gap_m": float(np.mean(np.abs(values))) if len(values) else None,
            "median_abs_gap_m": float(np.median(np.abs(values))) if len(values) else None,
            "max_abs_gap_m": float(np.max(np.abs(values))) if len(values) else None,
            "within_gap_tolerance_fraction": (
                int(np.sum(np.abs(values) <= tolerance)) / count if count else None
            ),
            "boundary_valid_fraction": valid[kind] / count if count else None,
        }
    return result


def evaluate(
    model_path: str | Path,
    data_path: str | Path,
    *,
    split: str = "test",
    task: str | None = None,
    device: str = "auto",
    seed: int = 0,
    row_limit: int = 0,
    batch_size: int = 32,
    mc_samples: int = 64,
    boundary_samples: int = 16,
    boundary_tolerance: float | None = None,
) -> dict[str, Any]:
    """Return field errors and independent MuJoCo boundary checks as JSON data."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {tuple(SPLITS)}")
    if row_limit < 0 or min(batch_size, mc_samples, boundary_samples) <= 0:
        raise ValueError("row limit must be nonnegative and batch/sample counts positive")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    field = CDFlowField.load(model_path, device=device)
    dataset = BoundaryDataset(data_path)
    task, same_metadata = _check_metadata(field, dataset, task)
    tolerance = (float(dataset.metadata.get("config", {}).get("boundary_tolerance", 0.002))
                 if boundary_tolerance is None else boundary_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("boundary tolerance must be finite and positive")
    rows = dataset.rows(SPLITS[split])
    if not len(rows):
        raise ValueError(f"dataset has no {split} rows; generate scenes for that split")
    if row_limit and row_limit < len(rows):
        rows = np.sort(np.random.default_rng(seed).choice(rows, row_limit, replace=False))
    torch.manual_seed(seed)

    model = _build_model(task)
    oracle = PandaCollisionOracle(model)
    if not np.allclose(oracle.joint_limits, dataset.joint_limits, atol=1e-5, rtol=0):
        raise ValueError("dataset Panda joint limits differ from the selected MuJoCo task")
    if len(SphereScene.from_model(model).spheres) != dataset.scene.shape[1]:
        raise ValueError("dataset sphere slots differ from the selected MuJoCo task")
    scene_names = tuple(f"reach_obstacle_{i}" for i in range(dataset.scene.shape[1]))

    regression_d, regression_g, regression_time = _field_pass(
        field, dataset, rows, mode="regression", batch_size=batch_size,
        samples=mc_samples,
    )
    mc_d, mc_g, mc_time = _field_pass(
        field, dataset, rows, mode="mc", batch_size=batch_size,
        samples=mc_samples,
    )
    generated, sampling_time = _boundary_samples(
        field, dataset, rows, batch_size=batch_size, samples=boundary_samples,
    )
    if not all(np.all(np.isfinite(array)) for array in (regression_d, regression_g, mc_d, mc_g)):
        raise FloatingPointError("field produced nonfinite distance or direction")
    labels_d, labels_g = dataset.y_d[rows], dataset.y_g[rows]
    regression_cos, regression_count = _direction_cosines(regression_g, labels_g)
    mc_cos, mc_count = _direction_cosines(mc_g, labels_g)
    agreement_cos, agreement_count = _direction_cosines(regression_g, mc_g)
    geometry = _geometry_metrics(
        oracle, dataset, rows, generated, tolerance=tolerance, scene_names=scene_names,
    )
    return {
        "format": "muarm-cdflow-field-eval-v1",
        "model": str(Path(model_path).expanduser().resolve()),
        "dataset": str(Path(data_path).expanduser().resolve()),
        "task": task,
        "split": split,
        "held_out_scene_ids": np.unique(dataset.scene_id[rows]).astype(int).tolist(),
        "rows": len(rows),
        "metadata": {
            "schema_verified": True,
            "same_generation_metadata": same_metadata,
            "joint_limits_verified_against_checkpoint_and_mujoco": True,
        },
        "settings": {
            "device": str(field.device), "seed": seed, "row_limit": row_limit,
            "batch_size": batch_size, "mc_samples_per_query": mc_samples,
            "boundary_samples_per_query": boundary_samples,
            "boundary_tolerance_m": tolerance,
        },
        "regression": {
            "distance_mae_rad": float(np.mean(np.abs(regression_d - labels_d))),
            "direction_cosine_mean": regression_cos,
            "direction_cosine_rows": regression_count,
            "inference_total_ms": 1000 * regression_time,
            "inference_ms_per_query_amortized": 1000 * regression_time / len(rows),
        },
        "mc": {
            "distance_mae_rad": float(np.mean(np.abs(mc_d - labels_d))),
            "direction_cosine_mean": mc_cos,
            "direction_cosine_rows": mc_count,
            "inference_total_ms": 1000 * mc_time,
            "inference_ms_per_query_amortized": 1000 * mc_time / len(rows),
        },
        "agreement": {
            "distance_mae_rad": float(np.mean(np.abs(regression_d - mc_d))),
            "direction_cosine_mean": agreement_cos,
            "direction_cosine_rows": agreement_count,
        },
        "boundary": {
            "reference_targets": geometry["targets"],
            "model_samples": geometry["generated"],
            "sampling_total_ms": 1000 * sampling_time,
            "sampling_ms_per_query_amortized": 1000 * sampling_time / len(rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="CDFlow checkpoint .pt")
    parser.add_argument("--data", type=Path, required=True, help="boundary dataset .npz")
    parser.add_argument("--split", choices=tuple(SPLITS), default="test")
    parser.add_argument("--task", choices=("reach_obstacles", "reach_obstacles_2"),
                        help="needed when dataset metadata does not record its Panda task")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-limit", type=int, default=0,
                        help="random scene-split rows to evaluate; 0 uses all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mc-samples", type=int, default=64)
    parser.add_argument("--boundary-samples", type=int, default=16)
    parser.add_argument("--boundary-tolerance", type=float,
                        help="MuJoCo gap tolerance in metres (default: dataset setting)")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    report = evaluate(
        args.model, args.data, split=args.split, task=args.task, device=args.device,
        seed=args.seed, row_limit=args.row_limit, batch_size=args.batch_size,
        mc_samples=args.mc_samples, boundary_samples=args.boundary_samples,
        boundary_tolerance=args.boundary_tolerance,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
