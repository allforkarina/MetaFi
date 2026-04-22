from __future__ import annotations

"""Training loop and checkpoint management for WPFormer."""

from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

from .config import DEFAULT_NUM_EPOCHS, build_default_optimizer, build_lambda_scheduler
from .objectives import calculate_mse_loss, calculate_pck_scores


class Trainer:
    """Coordinate training, validation, checkpointing, and loss visualization."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,                                   # train_dataset dataloader
        val_loader,                                     # val_dataset dataloader
        device: str | torch.device,                     # default "cuda" if available, else "cpu"
        num_epochs: int = DEFAULT_NUM_EPOCHS,           # 50
        checkpoint_path: str | Path | None = None,
        output_dir: str | Path = "outputs",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.plot_dir = self.output_dir / "plots"               # store the loss curve plot
        self.checkpoint_dir = self.output_dir / "checkpoints"   # store the best checkpoint of pck@50
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else self.checkpoint_dir / "best_wpformer.pt"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.optimizer = build_default_optimizer(self.model)    # AdamW optimizer
        self.scheduler = build_lambda_scheduler(self.optimizer, total_epochs=self.num_epochs)
        self.best_val_pck50 = 0.0                               # best pck@50
        self.history: list[dict[str, float]] = []               # trainning parameters history
        self.keypoint_x_scale, self.keypoint_y_scale = self._resolve_keypoint_scales()

    def _prepare_batch(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        inputs = batch["csi_amplitude"].float().to(self.device) # [B, 3, 114, 10]
        targets = batch["keypoints"].float().to(self.device)    # [B, 17, 2]
        return inputs, targets

    # get the scaling factors for keypoints
    def _resolve_keypoint_scales(self) -> tuple[float, float]:
        dataset = getattr(self.train_loader, "dataset", None)
        normalization = getattr(dataset, "keypoint_normalization", "")
        if normalization != "train_axis_max":
            return 1.0, 1.0

        x_scale = float(getattr(dataset, "keypoint_x_scale", 1.0))
        y_scale = float(getattr(dataset, "keypoint_y_scale", 1.0))
        return x_scale, y_scale

    # using the scaling factors to de-normalize the keypoints
    def _denormalize_keypoints_for_metrics(self, keypoints: Tensor) -> Tensor:
        restored = keypoints.clone()
        restored[..., 0] = restored[..., 0] * self.keypoint_x_scale
        restored[..., 1] = restored[..., 1] * self.keypoint_y_scale
        return restored

    def _create_progress_bar(self, loader, description: str):
        return tqdm(
            loader,
            desc=description,
            leave=False,
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )

    def train_epoch(self, epoch: int | None = None) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        description = f"Train {epoch}/{self.num_epochs}" if epoch is not None else "Train"

        with self._create_progress_bar(self.train_loader, description) as progress_bar:
            for batch in progress_bar:
                inputs, targets = self._prepare_batch(batch)

                self.optimizer.zero_grad()                          # zero the parameter gradients
                predictions = self.model(inputs)                    # predictions
                loss = calculate_mse_loss(predictions, targets)     # calculate the MSE loss
                loss.backward()                                     # calculate the gradients of parameters
                self.optimizer.step()                               # optimize the parameters

                batch_size = inputs.shape[0]
                total_loss += loss.item() * batch_size              # total loss for epoch
                total_samples += batch_size                         # total samples for epoch
                progress_bar.set_postfix(train_loss=total_loss / total_samples)

        return {"train_loss": total_loss / total_samples}       # average loss

    def validate_epoch(self, epoch: int | None = None) -> dict[str, float]:
        self.model.eval()                                       # validation mode (no model update)
        total_loss = 0.0
        total_samples = 0
        all_predictions: list[Tensor] = []
        all_targets: list[Tensor] = []
        description = f"Val {epoch}/{self.num_epochs}" if epoch is not None else "Val"

        with torch.no_grad():
            with self._create_progress_bar(self.val_loader, description) as progress_bar:
                for batch in progress_bar:
                    inputs, targets = self._prepare_batch(batch)
                    predictions = self.model(inputs)
                    loss = calculate_mse_loss(predictions, targets)

                    batch_size = inputs.shape[0]
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    all_predictions.append(predictions.detach().cpu())
                    all_targets.append(targets.detach().cpu())
                    progress_bar.set_postfix(val_loss=total_loss / total_samples)

        predictions = torch.cat(all_predictions, dim=0)                         # B, 17, 2
        targets = torch.cat(all_targets, dim=0)                                 # B, 17, 2
        predictions = self._denormalize_keypoints_for_metrics(predictions)      # de-normalize the prediction keypoints
        targets = self._denormalize_keypoints_for_metrics(targets)              # de-normalize the target keypoints
        metrics = calculate_pck_scores(predictions, targets)                    # calculate pck@10, 20, 30, 40, 50 in raw scale

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
        # epoch training.
        for epoch_index in range(self.num_epochs):
            epoch = epoch_index + 1                                 # current epoch
            current_lr = self.optimizer.param_groups[0]["lr"]       # current learning rate

            train_metrics = self.train_epoch(epoch=epoch)           # training for one epoch
            val_metrics = self.validate_epoch(epoch=epoch)          # validation for one epoch

            epoch_record = {                                        # logs
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

            self._save_best_checkpoint(epoch=epoch, metrics=val_metrics)    # find the best checkpoints.
            self._save_loss_curve()                                         # re-cover update for each epoch
            self.scheduler.step()                                           # update learning rate

            print(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"train_loss={epoch_record['train_loss']:.6f} | "
                f"val_loss={epoch_record['val_loss']:.6f} | "
                f"pck@50={epoch_record['pck@50']:.4f} | "
                f"lr={epoch_record['lr']:.6f}"
            )

        self._save_loss_curve()                                             # final loss curve
        return self.history
