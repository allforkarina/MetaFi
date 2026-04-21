from __future__ import annotations

"""Shared CNN encoder for per-antenna CSI amplitude features.

The module receives one CSI frame with shape [B, 3, 114, 10], splits the input
into three single-antenna branches, applies the same CNN backbone to every
branch, and concatenates the branch features along the last spatial dimension.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# Basic ResNet connection CNN block.
class BasicBlock(nn.Module):
    """Standard ResNet basic block with an optional downsample shortcut."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(                     # first cnn layer: in -> out
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)     # BatchNorm
        self.relu = nn.ReLU(inplace=True)           # activation function: Relu
        self.conv2 = nn.Conv2d(                     # second cnn layer: out -> out 
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,                              # out size = (in size + 2 * padding - kernel size) / stride + 1 = same
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)     # BatchNorm

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x                            # input: x

        out = self.conv1(x)                     # cnn layer: out - cnn(x)
        out = self.bn1(out)                     # batchnorm
        out = self.relu(out)                    # activation

        out = self.conv2(out)                   # cnn layer: out - cnn(out)
        out = self.bn2(out)                     # batchnorm 

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity                    # resnet connection
        out = self.relu(out)                    # activation
        return out


# The Shared CNN backbone, containing multiple layers of BasicBlock.
class SharedCNNBackbone(nn.Module):
    """Shared backbone that maps one antenna input from [B, 1, 136, 32] to [B, 512, 17, 4]."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),   # input 1 channel, out 64 channels, same size
            nn.BatchNorm2d(64),                                                 # batchnorm
            nn.ReLU(inplace=True),                                              # activation function: Relu 
        )

        # multiple blocks as a layer, each block is a resnet cnn
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
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]         # the first block in each layer: in -> out
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))     # the rest blocks in each layer: out -> out
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
    # upsample from [B, 1, 114, 10] to [B, 1, 136, 32]
    def _upsample_branch(x: Tensor) -> Tensor:
        return F.interpolate(x, size=(136, 32), mode="bilinear", align_corners=False)               

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape [B, 3, 114, 10], got {tuple(x.shape)}")
        if x.shape[1:] != (3, 114, 10):
            raise ValueError(f"Expected input with shape [B, 3, 114, 10], got {tuple(x.shape)}")

        # split the csi input into three branches along the channel of antenna dimension.
        branch_inputs = torch.chunk(x, chunks=3, dim=1)
        branch_outputs = []

        # All three branches use the same backbone instance, so parameters are shared.
        for branch in branch_inputs:
            branch = self._upsample_branch(branch)
            branch_outputs.append(self.backbone(branch))

        return torch.cat(branch_outputs, dim=-1)
