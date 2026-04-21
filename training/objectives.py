from __future__ import annotations

"""Loss and evaluation metrics for WPFormer training and validation."""

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor


RIGHT_SHOULDER_INDEX = 6    # Keypoint index of the right shoulder
LEFT_HIP_INDEX = 11         # Keypoint index of the left hip
EPSILON = 1e-8              # constant to avoid division by zero


def calculate_mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Compute plain coordinate-space MSE loss on [B, 17, 2] tensors."""

    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions and targets must have the same shape, "
            f"got {tuple(predictions.shape)} and {tuple(targets.shape)}"
        )
    if predictions.ndim != 3 or predictions.shape[1:] != (17, 2):
        raise ValueError(
            f"Expected predictions and targets with shape [B, 17, 2], got {tuple(predictions.shape)}"
        )

    return F.mse_loss(predictions, targets)


# human body constrain loss to avoid unreasonable pose
def _calculate_torso_length(targets: Tensor) -> Tensor:
    right_shoulder = targets[:, RIGHT_SHOULDER_INDEX, :]
    left_hip = targets[:, LEFT_HIP_INDEX, :]
    return torch.linalg.norm(right_shoulder - left_hip, dim=-1)     # distance between right shoulder and left hip


def calculate_pck(predictions: Tensor, targets: Tensor, threshold: float) -> Tensor:
    """Compute normalized PCK using torso length as the reference distance."""

    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions and targets must have the same shape, "
            f"got {tuple(predictions.shape)} and {tuple(targets.shape)}"
        )
    if predictions.ndim != 3 or predictions.shape[1:] != (17, 2):
        raise ValueError(
            f"Expected predictions and targets with shape [B, 17, 2], got {tuple(predictions.shape)}"
        )

    errors = torch.linalg.norm(predictions - targets, dim=-1)       # B, 17 pairs of keypoints in total
    torso_length = _calculate_torso_length(targets).unsqueeze(-1)
    normalized_errors = errors / (torso_length + EPSILON)           # normalize
    correct_keypoints = normalized_errors < threshold               # calculate the correct pairs' num
    return correct_keypoints.float().mean()                         # average over all keypoints and batch to get the PCK score


def calculate_pck_scores(
    predictions: Tensor,
    targets: Tensor,
    thresholds: Iterable[float] = (0.10, 0.20, 0.30, 0.40, 0.50),
) -> dict[str, Tensor]:
    """Compute PCK scores for a collection of normalized thresholds."""

    scores: dict[str, Tensor] = {}
    for threshold in thresholds:
        key = f"pck@{int(threshold * 100)}"
        scores[key] = calculate_pck(predictions, targets, threshold)
    return scores
