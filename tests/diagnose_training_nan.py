from __future__ import annotations

"""Diagnose NaN loss issues for WPFormer training.

This script bundles the most useful checks for the current MM-Fi training
pipeline into one executable entrypoint. It helps distinguish between:

- non-finite values inside the packed HDF5 dataset
- invalid batches returned by the dataloader
- NaNs introduced during model forward
- NaNs introduced during backward or optimizer.step()

Run it from the project root, for example:

python tests/diagnose_training_nan.py --dataset-root /data/WiFiPose/dataset/mmfi_pose.h5
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import create_data_loaders
from models.wpformer import WPFormer
from training.config import build_default_optimizer
from training.objectives import calculate_mse_loss


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose NaN loss issues for WPFormer training")
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the packed HDF5 dataset file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used for batch/forward/backward diagnosis",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count used during diagnosis",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for forward/backward checks",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk size for HDF5 finite-value scanning",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def inspect_h5_metadata(dataset_path: Path) -> dict[str, Any]:
    print_section("H5 Metadata")

    with h5py.File(dataset_path, "r") as h5_file:
        metadata = {
            "keys": list(h5_file.keys()),
            "amplitude_normalization": h5_file.attrs.get("amplitude_normalization"),
            "amplitude_train_min": h5_file.attrs.get("amplitude_train_min"),
            "amplitude_train_max": h5_file.attrs.get("amplitude_train_max"),
            "train_frames": len(h5_file["train_indices"]),
            "val_frames": len(h5_file["val_indices"]),
            "test_frames": len(h5_file["test_indices"]),
        }

    for key, value in metadata.items():
        print(f"{key}: {value}")

    normalization_ok = metadata["amplitude_normalization"] == "train_global_minmax"
    min_ok = is_finite_number(metadata["amplitude_train_min"])
    max_ok = is_finite_number(metadata["amplitude_train_max"])
    metadata["metadata_ok"] = normalization_ok and min_ok and max_ok
    print(f"metadata_ok: {metadata['metadata_ok']}")
    return metadata


def scan_dataset_finite(
    dataset_path: Path,
    dataset_name: str,
    chunk_size: int,
) -> dict[str, Any]:
    print_section(f"Finite Scan: {dataset_name}")

    total_non_finite = 0
    finite_min = float("inf")
    finite_max = float("-inf")
    bad_slices: list[tuple[int, int, int]] = []

    with h5py.File(dataset_path, "r") as h5_file:
        dataset = h5_file[dataset_name]
        total_records = len(dataset)

        for start in range(0, total_records, chunk_size):
            end = min(start + chunk_size, total_records)
            values = dataset[start:end]
            finite_mask = np.isfinite(values)

            if not finite_mask.all():
                batch_bad = int(values.size - int(finite_mask.sum()))
                total_non_finite += batch_bad
                bad_slices.append((start, end, batch_bad))

            finite_values = values[finite_mask]
            if finite_values.size > 0:
                finite_min = min(finite_min, float(finite_values.min()))
                finite_max = max(finite_max, float(finite_values.max()))

    result = {
        "dataset": dataset_name,
        "total_non_finite": total_non_finite,
        "finite_min": finite_min,
        "finite_max": finite_max,
        "bad_slices": bad_slices[:10],
        "all_finite": total_non_finite == 0,
    }

    for key, value in result.items():
        print(f"{key}: {value}")

    return result


def inspect_first_batch(
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    print_section("First Batch")

    loaders = create_data_loaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    batch = next(iter(loaders["train"]))
    inputs = batch["csi_amplitude"].float()
    targets = batch["keypoints"].float()

    result = {
        "x_shape": tuple(inputs.shape),
        "y_shape": tuple(targets.shape),
        "x_all_finite": bool(torch.isfinite(inputs).all().item()),
        "y_all_finite": bool(torch.isfinite(targets).all().item()),
        "x_min": float(inputs.min().item()),
        "x_max": float(inputs.max().item()),
        "y_min": float(targets.min().item()),
        "y_max": float(targets.max().item()),
    }

    for key, value in result.items():
        print(f"{key}: {value}")

    return result, {"inputs": inputs, "targets": targets}


def inspect_single_forward(
    device: torch.device,
    inputs: torch.Tensor,
) -> tuple[dict[str, Any], WPFormer]:
    print_section("Single Forward")

    model = WPFormer().to(device)
    model.eval()

    with torch.no_grad():
        predictions = model(inputs.to(device))

    result = {
        "pred_shape": tuple(predictions.shape),
        "pred_all_finite": bool(torch.isfinite(predictions).all().item()),
        "pred_min": float(predictions.min().item()),
        "pred_max": float(predictions.max().item()),
    }

    for key, value in result.items():
        print(f"{key}: {value}")

    return result, model


def inspect_single_backward(
    device: torch.device,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[dict[str, Any], WPFormer]:
    print_section("Single Backward")

    model = WPFormer().to(device)
    optimizer = build_default_optimizer(model)

    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    predictions = model(inputs)
    loss = calculate_mse_loss(predictions, targets)

    loss_value = float(loss.item())
    loss_is_finite = bool(torch.isfinite(loss).item())

    if loss_is_finite:
        loss.backward()

    bad_grad_names: list[str] = []
    max_grad_abs = 0.0

    if loss_is_finite:
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                bad_grad_names.append(name)
            else:
                max_grad_abs = max(max_grad_abs, float(parameter.grad.abs().max().item()))

    result = {
        "pred_all_finite": bool(torch.isfinite(predictions).all().item()),
        "loss": loss_value,
        "loss_is_finite": loss_is_finite,
        "bad_grad_count": len(bad_grad_names),
        "bad_grad_names_first10": bad_grad_names[:10],
        "max_grad_abs": max_grad_abs,
    }

    for key, value in result.items():
        print(f"{key}: {value}")

    return result, model


def inspect_optimizer_step(
    device: torch.device,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    print_section("Optimizer Step")

    model = WPFormer().to(device)
    optimizer = build_default_optimizer(model)

    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    predictions = model(inputs)
    loss = calculate_mse_loss(predictions, targets)
    loss.backward()
    optimizer.step()

    bad_param_names: list[str] = []
    max_param_abs = 0.0
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            bad_param_names.append(name)
        else:
            max_param_abs = max(max_param_abs, float(parameter.abs().max().item()))

    result = {
        "loss": float(loss.item()),
        "bad_param_count": len(bad_param_names),
        "bad_param_names_first10": bad_param_names[:10],
        "max_param_abs": max_param_abs,
    }

    for key, value in result.items():
        print(f"{key}: {value}")

    return result


def print_diagnosis_summary(
    metadata: dict[str, Any],
    amplitude_scan: dict[str, Any],
    keypoint_scan: dict[str, Any],
    batch_result: dict[str, Any],
    forward_result: dict[str, Any],
    backward_result: dict[str, Any],
    step_result: dict[str, Any],
) -> None:
    print_section("Diagnosis Summary")

    if not metadata["metadata_ok"]:
        print("Likely cause: the H5 file was not generated by the latest cleaning/normalization pipeline.")
        return

    if not amplitude_scan["all_finite"] or not keypoint_scan["all_finite"]:
        print("Likely cause: the H5 file still contains non-finite values.")
        return

    if not batch_result["x_all_finite"] or not batch_result["y_all_finite"]:
        print("Likely cause: the dataloader returned a non-finite batch.")
        return

    if not forward_result["pred_all_finite"]:
        print("Likely cause: the model forward pass already produces NaN/Inf.")
        return

    if not backward_result["loss_is_finite"]:
        print("Likely cause: the loss becomes non-finite before backward.")
        return

    if backward_result["bad_grad_count"] > 0:
        print("Likely cause: backward introduces non-finite gradients.")
        return

    if step_result["bad_param_count"] > 0:
        print("Likely cause: optimizer.step() corrupts parameters on the first update.")
        return

    print("No NaN source was reproduced in the first-step diagnosis.")
    print("Next suspects: large batch size, later-batch outliers, or longer-horizon optimization instability.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = Path(args.dataset_root)
    device = torch.device(args.device)

    metadata = inspect_h5_metadata(dataset_path)
    amplitude_scan = scan_dataset_finite(dataset_path, "csi_amplitude", args.chunk_size)
    keypoint_scan = scan_dataset_finite(dataset_path, "keypoints", args.chunk_size)
    batch_result, batch_tensors = inspect_first_batch(
        dataset_root=dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    forward_result, _ = inspect_single_forward(device, batch_tensors["inputs"])
    backward_result, _ = inspect_single_backward(
        device,
        batch_tensors["inputs"],
        batch_tensors["targets"],
    )
    step_result = inspect_optimizer_step(
        device,
        batch_tensors["inputs"],
        batch_tensors["targets"],
    )

    print_diagnosis_summary(
        metadata=metadata,
        amplitude_scan=amplitude_scan,
        keypoint_scan=keypoint_scan,
        batch_result=batch_result,
        forward_result=forward_result,
        backward_result=backward_result,
        step_result=step_result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
