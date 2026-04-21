from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from models.wpformer import WPFormer
from training.trainer import Trainer


def _make_dummy_loader(num_batches: int = 1, batch_size: int = 2) -> list[dict[str, torch.Tensor]]:
    loader: list[dict[str, torch.Tensor]] = []
    for _ in range(num_batches):
        loader.append(
            {
                "csi_amplitude": torch.randn(batch_size, 3, 114, 10),
                "keypoints": torch.randn(batch_size, 17, 2),
            }
        )
    return loader


def test_train_epoch_returns_train_loss(tmp_path: Path) -> None:
    trainer = Trainer(
        model=WPFormer(),
        train_loader=_make_dummy_loader(),
        val_loader=_make_dummy_loader(),
        device="cpu",
        num_epochs=1,
        output_dir=tmp_path,
    )

    metrics = trainer.train_epoch()

    assert "train_loss" in metrics
    assert metrics["train_loss"] >= 0.0


def test_validate_epoch_returns_loss_and_pck_metrics(tmp_path: Path) -> None:
    trainer = Trainer(
        model=WPFormer(),
        train_loader=_make_dummy_loader(),
        val_loader=_make_dummy_loader(),
        device="cpu",
        num_epochs=1,
        output_dir=tmp_path,
    )

    metrics = trainer.validate_epoch()

    assert set(metrics.keys()) == {"val_loss", "pck@10", "pck@20", "pck@30", "pck@40", "pck@50"}


def test_fit_runs_one_epoch_and_saves_checkpoint_and_loss_curve(tmp_path: Path) -> None:
    trainer = Trainer(
        model=WPFormer(),
        train_loader=_make_dummy_loader(),
        val_loader=_make_dummy_loader(),
        device="cpu",
        num_epochs=1,
        output_dir=tmp_path,
    )

    history = trainer.fit()

    assert len(history) == 1
    assert history[0]["epoch"] == 1
    assert (tmp_path / "checkpoints" / "best_wpformer.pt").exists()
    assert (tmp_path / "plots" / "loss_curve.png").exists()
    assert (tmp_path / "plots" / "loss_curve.png").stat().st_size > 0


def test_saved_checkpoint_contains_expected_keys(tmp_path: Path) -> None:
    trainer = Trainer(
        model=WPFormer(),
        train_loader=_make_dummy_loader(),
        val_loader=_make_dummy_loader(),
        device="cpu",
        num_epochs=1,
        output_dir=tmp_path,
    )
    trainer.fit()

    checkpoint = torch.load(tmp_path / "checkpoints" / "best_wpformer.pt", map_location="cpu")

    assert {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "best_val_pck50",
        "val_metrics",
    }.issubset(checkpoint.keys())


def test_scheduler_changes_learning_rate_after_fit(tmp_path: Path) -> None:
    trainer = Trainer(
        model=WPFormer(),
        train_loader=_make_dummy_loader(),
        val_loader=_make_dummy_loader(),
        device="cpu",
        num_epochs=2,
        output_dir=tmp_path,
    )
    initial_lr = trainer.optimizer.param_groups[0]["lr"]

    trainer.fit()

    assert trainer.optimizer.param_groups[0]["lr"] < initial_lr


def test_fit_runs_without_progress_bar_in_non_interactive_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)

    trainer = Trainer(
        model=WPFormer(),
        train_loader=_make_dummy_loader(),
        val_loader=_make_dummy_loader(),
        device="cpu",
        num_epochs=1,
        output_dir=tmp_path,
    )

    history = trainer.fit()

    assert len(history) == 1
