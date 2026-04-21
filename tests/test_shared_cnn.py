from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.shared_cnn import BasicBlock, SharedCNN, SharedCNNBackbone


def test_basic_block_downsamples_when_stride_or_channels_change() -> None:
    block = BasicBlock(in_channels=64, out_channels=128, stride=2)
    x = torch.randn(2, 64, 136, 32)

    y = block(x)

    assert y.shape == (2, 128, 68, 16)


def test_shared_backbone_output_shape() -> None:
    backbone = SharedCNNBackbone()
    x = torch.randn(2, 1, 136, 32)

    y = backbone(x)

    assert y.shape == (2, 512, 17, 4)


def test_shared_cnn_output_shape() -> None:
    model = SharedCNN()
    x = torch.randn(2, 3, 114, 10)

    y = model(x)

    assert y.shape == (2, 512, 17, 12)


def test_shared_cnn_uses_single_backbone_instance() -> None:
    model = SharedCNN()

    assert isinstance(model.backbone, SharedCNNBackbone)
    assert sum(1 for module in model.children() if isinstance(module, SharedCNNBackbone)) == 1


def test_shared_cnn_rejects_wrong_input_shape() -> None:
    model = SharedCNN()
    x = torch.randn(2, 6, 114, 10)

    with pytest.raises(ValueError, match=r"\[B, 3, 114, 10\]"):
        model(x)
