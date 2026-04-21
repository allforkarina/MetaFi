from __future__ import annotations

"""Default optimization settings for the future trainer stage."""

from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LambdaLR


DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_MOMENTUM = 0.9
DEFAULT_PCK_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50)


def build_default_optimizer(model: nn.Module) -> Optimizer:
    """Build the default SGDM optimizer described in the paper."""

    return SGD(model.parameters(), lr=DEFAULT_LR, momentum=DEFAULT_MOMENTUM)


def build_lambda_scheduler(optimizer: Optimizer, total_epochs: int) -> LambdaLR:
    """Linearly decay the learning rate to zero across the training horizon."""

    if total_epochs <= 0:
        raise ValueError("total_epochs must be greater than 0")

    return LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: max(0.0, 1.0 - (epoch / total_epochs)),
    )
