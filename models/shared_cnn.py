from __future__ import annotations

"""Shared CNN encoder for per-antenna CSI amplitude features.

The module receives one CSI frame with shape [B, 3, 114, 10], splits the input
into three single-antenna branches, applies the same CNN backbone to every
branch, and concatenates the branch features along the last spatial dimension.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BasicBlock(nn.Module):
    """Standard ResNet basic block with an optional downsample shortcut."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class SharedCNNBackbone(nn.Module):
    """Shared backbone that maps one antenna input from [B, 1, 136, 32] to [B, 512, 17, 4]."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(in_channels=64, out_channels=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(in_channels=64, out_channels=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(in_channels=128, out_channels=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(in_channels=256, out_channels=512, blocks=3, stride=2)

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SharedCNN(nn.Module):
    """Apply one shared CNN backbone to three antenna branches and concatenate outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = SharedCNNBackbone()

    @staticmethod
    def _upsample_branch(x: Tensor) -> Tensor:
        return F.interpolate(x, size=(136, 32), mode="bilinear", align_corners=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape [B, 3, 114, 10], got {tuple(x.shape)}")
        if x.shape[1:] != (3, 114, 10):
            raise ValueError(f"Expected input with shape [B, 3, 114, 10], got {tuple(x.shape)}")

        branch_inputs = torch.chunk(x, chunks=3, dim=1)
        branch_outputs = []

        # All three branches use the same backbone instance, so parameters are shared.
        for branch in branch_inputs:
            branch = self._upsample_branch(branch)
            branch_outputs.append(self.backbone(branch))

        return torch.cat(branch_outputs, dim=-1)
