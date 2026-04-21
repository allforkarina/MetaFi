from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from models.wpformer import WPFormer
from training.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_MOMENTUM,
    DEFAULT_PCK_THRESHOLDS,
    build_default_optimizer,
    build_lambda_scheduler,
)
from training.objectives import (
    LEFT_HIP_INDEX,
    RIGHT_SHOULDER_INDEX,
    calculate_mse_loss,
    calculate_pck,
    calculate_pck_scores,
)


def test_calculate_mse_loss_returns_zero_for_identical_inputs() -> None:
    targets = torch.zeros(2, 17, 2)

    loss = calculate_mse_loss(targets, targets)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_calculate_mse_loss_matches_manual_value() -> None:
    predictions = torch.zeros(1, 17, 2)
    targets = torch.zeros(1, 17, 2)
    predictions[0, 0, 0] = 2.0

    loss = calculate_mse_loss(predictions, targets)

    assert torch.isclose(loss, torch.tensor(4.0 / (17 * 2)))


def test_calculate_pck_returns_one_for_identical_inputs() -> None:
    targets = torch.zeros(2, 17, 2)
    targets[:, RIGHT_SHOULDER_INDEX, :] = torch.tensor([1.0, 0.0])
    targets[:, LEFT_HIP_INDEX, :] = torch.tensor([0.0, 0.0])

    pck = calculate_pck(targets, targets, threshold=0.10)

    assert torch.isclose(pck, torch.tensor(1.0))


def test_calculate_pck_matches_hand_computed_ratio() -> None:
    targets = torch.zeros(1, 17, 2)
    targets[0, RIGHT_SHOULDER_INDEX, :] = torch.tensor([2.0, 0.0])
    targets[0, LEFT_HIP_INDEX, :] = torch.tensor([0.0, 0.0])

    predictions = targets.clone()
    predictions[0, 0, :] = torch.tensor([0.1, 0.0])
    predictions[0, 1, :] = torch.tensor([0.4, 0.0])

    pck = calculate_pck(predictions, targets, threshold=0.10)

    assert torch.isclose(pck, torch.tensor(16.0 / 17.0))


def test_calculate_pck_handles_zero_torso_length_without_nan() -> None:
    predictions = torch.zeros(1, 17, 2)
    targets = torch.zeros(1, 17, 2)

    pck = calculate_pck(predictions, targets, threshold=0.10)

    assert torch.isfinite(pck)
    assert torch.isclose(pck, torch.tensor(1.0))


def test_calculate_pck_scores_returns_default_threshold_keys() -> None:
    targets = torch.zeros(1, 17, 2)
    targets[0, RIGHT_SHOULDER_INDEX, :] = torch.tensor([1.0, 0.0])
    targets[0, LEFT_HIP_INDEX, :] = torch.tensor([0.0, 0.0])

    scores = calculate_pck_scores(targets, targets)

    assert list(scores.keys()) == ["pck@10", "pck@20", "pck@30", "pck@40", "pck@50"]


def test_default_training_constants_match_plan() -> None:
    assert DEFAULT_BATCH_SIZE == 32
    assert DEFAULT_LR == 1e-3
    assert DEFAULT_MOMENTUM == 0.9
    assert DEFAULT_PCK_THRESHOLDS == (0.10, 0.20, 0.30, 0.40, 0.50)


def test_build_default_optimizer_returns_sgd_with_expected_params() -> None:
    model = WPFormer()

    optimizer = build_default_optimizer(model)

    assert isinstance(optimizer, SGD)
    assert optimizer.defaults["lr"] == DEFAULT_LR
    assert optimizer.defaults["momentum"] == DEFAULT_MOMENTUM


def test_build_lambda_scheduler_returns_linear_decay_scheduler() -> None:
    model = WPFormer()
    optimizer = build_default_optimizer(model)

    scheduler = build_lambda_scheduler(optimizer, total_epochs=10)

    assert isinstance(scheduler, LambdaLR)
    assert scheduler.lr_lambdas[0](0) == 1.0
    assert scheduler.lr_lambdas[0](10) == 0.0


def test_build_lambda_scheduler_rejects_non_positive_total_epochs() -> None:
    model = WPFormer()
    optimizer = build_default_optimizer(model)

    with pytest.raises(ValueError, match="total_epochs must be greater than 0"):
        build_lambda_scheduler(optimizer, total_epochs=0)
