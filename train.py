from __future__ import annotations

"""Minimal training entrypoint for WPFormer."""

import argparse

import torch

from dataloader import create_data_loaders
from models.wpformer import WPFormer
from training import DEFAULT_BATCH_SIZE, DEFAULT_NUM_EPOCHS, Trainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train WPFormer on the MM-Fi dataset")
    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset root path")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Training epochs")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path for the best checkpoint file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for logs, plots, and default checkpoints",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str, float]]:
    args = parse_args(argv)
    loaders = create_data_loaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = WPFormer()
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=args.device,
        num_epochs=args.num_epochs,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
    )
    return trainer.fit()


if __name__ == "__main__":
    main()
