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


class _LoaderWithDataset:
    def __init__(self, batches: list[dict[str, torch.Tensor]], dataset) -> None:
        self._batches = batches
        self.dataset = dataset

    def __iter__(self):
        return iter(self._batches)


class _NormalizedDataset:
    keypoint_normalization = "train_axis_max"
    keypoint_x_scale = 100.0
    keypoint_y_scale = 200.0


class _ConstantModel(torch.nn.Module):
    def __init__(self, prediction: torch.Tensor) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("_prediction", prediction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._prediction.expand(x.shape[0], -1, -1) + (0.0 * self.dummy)


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


def test_validate_epoch_denormalizes_predictions_before_pck(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, torch.Tensor] = {}

    def fake_calculate_pck_scores(predictions: torch.Tensor, targets: torch.Tensor):
        captured["predictions"] = predictions
        captured["targets"] = targets
        return {
            "pck@10": torch.tensor(1.0),
            "pck@20": torch.tensor(1.0),
            "pck@30": torch.tensor(1.0),
            "pck@40": torch.tensor(1.0),
            "pck@50": torch.tensor(1.0),
        }

    monkeypatch.setattr("training.trainer.calculate_pck_scores", fake_calculate_pck_scores)

    normalized_target = torch.full((1, 17, 2), 0.5)
    val_loader = _LoaderWithDataset(
        [
            {
                "csi_amplitude": torch.zeros(1, 3, 114, 10),
                "keypoints": normalized_target,
            }
        ],
        dataset=_NormalizedDataset(),
    )
    trainer = Trainer(
        model=_ConstantModel(prediction=normalized_target),
        train_loader=val_loader,
        val_loader=val_loader,
        device="cpu",
        num_epochs=1,
        output_dir=tmp_path,
    )

    metrics = trainer.validate_epoch()

    assert metrics["pck@50"] == 1.0
    assert torch.allclose(captured["predictions"][0, 0], torch.tensor([50.0, 100.0]))
    assert torch.allclose(captured["targets"][0, 0], torch.tensor([50.0, 100.0]))
