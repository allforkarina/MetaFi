from __future__ import annotations

"""Build a single HDF5 dataset file from the raw MM-Fi directory structure."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import build_h5_dataset, resolve_dataset_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack the raw MM-Fi dataset into one HDF5 file")
    parser.add_argument("--dataset-root", type=str, default=None, help="Raw MM-Fi dataset root directory")
    parser.add_argument("--output-path", type=str, required=True, help="Target HDF5 dataset path")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = parse_args(argv)
    dataset_root = resolve_dataset_root(args.dataset_root)
    summary = build_h5_dataset(
        dataset_root=dataset_root,
        output_path=args.output_path,
        seed=args.seed,
    )

    print(f"dataset_root: {dataset_root}")
    print(f"output_path: {Path(args.output_path)}")
    print(
        f"frames: total={summary['num_records']}, "
        f"train={summary['num_train_frames']}, "
        f"val={summary['num_val_frames']}, "
        f"test={summary['num_test_frames']}"
    )
    return summary


if __name__ == "__main__":
    main()
