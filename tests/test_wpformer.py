from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.wpformer import WPFormer


def test_wpformer_output_shape() -> None:
    model = WPFormer()
    x = torch.randn(2, 3, 114, 10)

    y = model(x)

    assert y.shape == (2, 17, 2)


def test_wpformer_rejects_wrong_input_shape() -> None:
    model = WPFormer()
    x = torch.randn(2, 6, 114, 10)

    with pytest.raises(ValueError, match=r"\[B, 3, 114, 10\]"):
        model(x)
