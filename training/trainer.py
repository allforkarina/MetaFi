from __future__ import annotations

"""Training loop and checkpoint management for WPFormer."""

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .config import DEFAULT_NUM_EPOCHS, build_default_optimizer, build_lambda_scheduler
from .objectives import calculate_mse_loss, calculate_pck_scores


class Trainer:
    """Coordinate training, validation, checkpointing, and loss visualization."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str | torch.device,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        checkpoint_path: str | Path | None = None,
        output_dir: str | Path = "outputs",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.plot_dir = self.output_dir / "plots"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else self.checkpoint_dir / "best_wpformer.pt"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.optimizer = build_default_optimizer(self.model)
        self.scheduler = build_lambda_scheduler(self.optimizer, total_epochs=self.num_epochs)
        self.best_val_pck50 = 0.0
        self.history: list[dict[str, float]] = []

    def _prepare_batch(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        inputs = batch["csi_amplitude"].float().to(self.device)
        targets = batch["keypoints"].float().to(self.device)
        return inputs, targets

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            inputs, targets = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            loss = calculate_mse_loss(predictions, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return {"train_loss": total_loss / total_samples}

    def validate_epoch(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions: list[Tensor] = []
        all_targets: list[Tensor] = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = self._prepare_batch(batch)
                predictions = self.model(inputs)
                loss = calculate_mse_loss(predictions, targets)

                batch_size = inputs.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                all_predictions.append(predictions.detach().cpu())
                all_targets.append(targets.detach().cpu())

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics = calculate_pck_scores(predictions, targets)

        return {
            "val_loss": total_loss / total_samples,
            **{name: score.item() for name, score in metrics.items()},
        }

    def _save_best_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        current_pck50 = metrics["pck@50"]
        if self.checkpoint_path.exists() and current_pck50 <= self.best_val_pck50:
            return

        self.best_val_pck50 = current_pck50
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "best_val_pck50": self.best_val_pck50,
                "val_metrics": metrics,
            },
            self.checkpoint_path,
        )

    def _save_loss_curve(self) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [entry["epoch"] for entry in self.history]
        train_losses = [entry["train_loss"] for entry in self.history]
        val_losses = [entry["val_loss"] for entry in self.history]

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, train_losses, label="Train Loss", linewidth=2)
        axis.plot(epochs, val_losses, label="Val Loss", linewidth=2)
        axis.set_title("WPFormer Loss Curve")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        axis.legend()
        figure.tight_layout()

        loss_curve_path = self.plot_dir / "loss_curve.png"
        figure.savefig(loss_curve_path)
        plt.close(figure)
        return loss_curve_path

    def fit(self) -> list[dict[str, float]]:
        for epoch_index in range(self.num_epochs):
            epoch = epoch_index + 1
            current_lr = self.optimizer.param_groups[0]["lr"]

            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_metrics["val_loss"],
                "pck@10": val_metrics["pck@10"],
                "pck@20": val_metrics["pck@20"],
                "pck@30": val_metrics["pck@30"],
                "pck@40": val_metrics["pck@40"],
                "pck@50": val_metrics["pck@50"],
                "lr": current_lr,
            }
            self.history.append(epoch_record)

            self._save_best_checkpoint(epoch=epoch, metrics=val_metrics)
            self._save_loss_curve()
            self.scheduler.step()

            print(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"train_loss={epoch_record['train_loss']:.6f} | "
                f"val_loss={epoch_record['val_loss']:.6f} | "
                f"pck@50={epoch_record['pck@50']:.4f} | "
                f"lr={epoch_record['lr']:.6f}"
            )

        self._save_loss_curve()
        return self.history
