from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.wpformer import WPFormer
from models.wpformer_amp_phase import WPFormerAmpPhase


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


def test_wpformer_amp_phase_output_shape() -> None:
    model = WPFormerAmpPhase()
    amplitude = torch.randn(2, 3, 114, 10)
    phase_cos = torch.randn(2, 3, 114, 10)

    y = model(amplitude, phase_cos)

    assert y.shape == (2, 17, 2)


def test_wpformer_amp_phase_rejects_wrong_phase_shape() -> None:
    model = WPFormerAmpPhase()
    amplitude = torch.randn(2, 3, 114, 10)
    phase_cos = torch.randn(2, 1, 114, 10)

    with pytest.raises(ValueError, match=r"\[B, 3, 114, 10\]"):
        model(amplitude, phase_cos)


def test_wpformer_amp_phase_uses_independent_encoders() -> None:
    model = WPFormerAmpPhase()
    amp_parameter = next(model.amp_encoder.parameters())
    phase_parameter = next(model.phase_encoder.parameters())

    assert model.amp_encoder is not model.phase_encoder
    assert amp_parameter.data_ptr() != phase_parameter.data_ptr()
