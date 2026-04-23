from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import train


def test_parse_args_supports_expected_training_arguments() -> None:
    args = train.parse_args(
        [
            "--dataset-root",
            "dataset",
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--num-epochs",
            "1",
            "--input-mode",
            "amp_phase",
            "--checkpoint-path",
            "best.pt",
            "--output-dir",
            "logs",
            "--num-workers",
            "2",
        ]
    )

    assert args.dataset_root == "dataset"
    assert args.device == "cpu"
    assert args.batch_size == 4
    assert args.num_epochs == 1
    assert args.input_mode == "amp_phase"
    assert args.checkpoint_path == "best.pt"
    assert args.output_dir == "logs"
    assert args.num_workers == 2


def test_main_builds_model_loaders_and_trainer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyTrainer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def fit(self) -> list[dict[str, float]]:
            return [{"epoch": 1, "train_loss": 1.0, "val_loss": 1.0, "pck@50": 0.5, "lr": 0.001}]

    monkeypatch.setattr(
        train,
        "create_data_loaders",
        lambda dataset_root, batch_size, num_workers: {
            "train": ["train_loader"],
            "val": ["val_loader"],
            "test": ["test_loader"],
        },
    )
    monkeypatch.setattr(train, "Trainer", DummyTrainer)

    history = train.main(
        [
            "--dataset-root",
            "dataset",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-epochs",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert history[0]["epoch"] == 1
    assert captured["train_loader"] == ["train_loader"]
    assert captured["val_loader"] == ["val_loader"]
    assert captured["device"] == "cpu"
    assert captured["num_epochs"] == 1
    assert captured["input_mode"] == "amp"


def test_main_supports_amp_phase_input_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyTrainer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def fit(self) -> list[dict[str, float]]:
            return [{"epoch": 1, "train_loss": 1.0, "val_loss": 1.0, "pck@50": 0.5, "lr": 0.001}]

    monkeypatch.setattr(
        train,
        "create_data_loaders",
        lambda dataset_root, batch_size, num_workers: {
            "train": ["train_loader"],
            "val": ["val_loader"],
            "test": ["test_loader"],
        },
    )
    monkeypatch.setattr(train, "Trainer", DummyTrainer)

    train.main(
        [
            "--dataset-root",
            "dataset",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-epochs",
            "1",
            "--input-mode",
            "amp_phase",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert captured["input_mode"] == "amp_phase"
    assert captured["model"].__class__.__name__ == "WPFormerAmpPhase"
