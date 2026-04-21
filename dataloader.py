from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat

try:
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - handled at runtime when torch is unavailable.
    DataLoader = None


DEFAULT_LOCAL_DATASET_ROOT = Path(r"D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset")
DEFAULT_LINUX_DATASET_ROOT = Path("/data/WiFiPose/dataset/dataset")
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = {"train": 6, "val": 2, "test": 2}
FRAMES_PER_SAMPLE = 297


@dataclass(frozen=True)
class SampleSequence:
    action: str
    sample: str
    environment: str
    rgb_dir: Path
    csi_dir: Path


@dataclass(frozen=True)
class FrameRecord:
    action: str
    sample: str
    environment: str
    frame_stem: str
    keypoint_path: Path
    csi_path: Path


def resolve_dataset_root(dataset_root: Optional[str | Path] = None) -> Path:
    if dataset_root is not None:
        root = Path(dataset_root)
    elif DEFAULT_LOCAL_DATASET_ROOT.exists():
        root = DEFAULT_LOCAL_DATASET_ROOT
    else:
        root = DEFAULT_LINUX_DATASET_ROOT

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def sample_to_environment(sample_name: str) -> str:
    sample_index = int(sample_name[1:])
    environment_index = (sample_index - 1) // 10 + 1
    return f"env{environment_index}"


def _sorted_dirs(root: Path, prefix: str) -> List[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name,
    )


def discover_sample_sequences(dataset_root: str | Path) -> List[SampleSequence]:
    root = resolve_dataset_root(dataset_root)
    sequences: List[SampleSequence] = []

    for action_dir in _sorted_dirs(root, "A"):
        for sample_dir in _sorted_dirs(action_dir, "S"):
            rgb_dir = sample_dir / "rgb"
            csi_dir = sample_dir / "wifi-csi"
            if not rgb_dir.is_dir() or not csi_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected aligned rgb and wifi-csi directories under {sample_dir}"
                )

            sequences.append(
                SampleSequence(
                    action=action_dir.name,
                    sample=sample_dir.name,
                    environment=sample_to_environment(sample_dir.name),
                    rgb_dir=rgb_dir,
                    csi_dir=csi_dir,
                )
            )

    if not sequences:
        raise ValueError(f"No sample sequences found under {root}")

    return sequences


def build_sample_splits(
    dataset_root: str | Path,
    seed: int = 42,
    split_ratios: Optional[Dict[str, int]] = None,
) -> Dict[str, List[SampleSequence]]:
    ratios = split_ratios or SPLIT_RATIOS
    if tuple(ratios.keys()) != SPLIT_NAMES:
        raise ValueError(f"Split keys must be exactly {SPLIT_NAMES}, got {tuple(ratios.keys())}")
    if sum(ratios.values()) != 10:
        raise ValueError("Per-environment split ratios must sum to 10 samples")

    grouped_sequences: Dict[Tuple[str, str], List[SampleSequence]] = {}
    for sequence in discover_sample_sequences(dataset_root):
        grouped_sequences.setdefault((sequence.action, sequence.environment), []).append(sequence)

    splits: Dict[str, List[SampleSequence]] = {name: [] for name in SPLIT_NAMES}
    for (action, environment), sequences in sorted(grouped_sequences.items()):
        ordered_sequences = sorted(sequences, key=lambda item: item.sample)
        if len(ordered_sequences) != 10:
            raise ValueError(
                f"Expected 10 samples for {action}/{environment}, found {len(ordered_sequences)}"
            )

        group_rng = random.Random(f"{seed}:{action}:{environment}")
        shuffled_sequences = ordered_sequences[:]
        group_rng.shuffle(shuffled_sequences)

        train_end = ratios["train"]
        val_end = train_end + ratios["val"]
        splits["train"].extend(shuffled_sequences[:train_end])
        splits["val"].extend(shuffled_sequences[train_end:val_end])
        splits["test"].extend(shuffled_sequences[val_end:])

    return splits


def _sorted_files(directory: Path, pattern: str) -> List[Path]:
    return sorted(directory.glob(pattern), key=lambda path: path.name)


def expand_frame_records(sequences: Sequence[SampleSequence]) -> List[FrameRecord]:
    records: List[FrameRecord] = []
    for sequence in sequences:
        keypoint_files = _sorted_files(sequence.rgb_dir, "*.npy")
        csi_files = _sorted_files(sequence.csi_dir, "*.mat")

        if len(keypoint_files) != len(csi_files):
            raise ValueError(
                f"Mismatched frame count for {sequence.action}/{sequence.sample}: "
                f"{len(keypoint_files)} labels vs {len(csi_files)} CSI files"
            )

        if len(keypoint_files) != FRAMES_PER_SAMPLE:
            raise ValueError(
                f"Expected {FRAMES_PER_SAMPLE} frames for {sequence.action}/{sequence.sample}, "
                f"found {len(keypoint_files)}"
            )

        for keypoint_path, csi_path in zip(keypoint_files, csi_files):
            if keypoint_path.stem != csi_path.stem:
                raise ValueError(
                    f"Frame mismatch for {sequence.action}/{sequence.sample}: "
                    f"{keypoint_path.name} vs {csi_path.name}"
                )

            records.append(
                FrameRecord(
                    action=sequence.action,
                    sample=sequence.sample,
                    environment=sequence.environment,
                    frame_stem=keypoint_path.stem,
                    keypoint_path=keypoint_path,
                    csi_path=csi_path,
                )
            )

    return records


class MMFiPoseDataset:
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        seed: int = 42,
        split_ratios: Optional[Dict[str, int]] = None,
    ) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split}")

        self.dataset_root = resolve_dataset_root(dataset_root)
        self.split = split
        self.seed = seed
        self.split_ratios = split_ratios or SPLIT_RATIOS
        self.sample_splits = build_sample_splits(
            self.dataset_root, seed=self.seed, split_ratios=self.split_ratios
        )
        self.sample_sequences = self.sample_splits[self.split]
        self.records = expand_frame_records(self.sample_sequences)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray | str]:
        record = self.records[index]
        keypoints = np.load(record.keypoint_path).astype(np.float32)
        csi_data = loadmat(record.csi_path)
        csi_amplitude = np.asarray(csi_data["CSIamp"], dtype=np.float32)
        csi_phase = np.asarray(csi_data["CSIphase"], dtype=np.float32)

        return {
            "action": record.action,
            "sample": record.sample,
            "environment": record.environment,
            "frame_id": record.frame_stem,
            "keypoints": keypoints,
            "csi_amplitude": csi_amplitude,
            "csi_phase": csi_phase,
        }


def create_data_loader(
    dataset_root: str | Path,
    split: str,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    split_ratios: Optional[Dict[str, int]] = None,
):
    if DataLoader is None:
        raise ImportError(
            "PyTorch is not installed in the current environment. "
            "Install torch to create DataLoader instances."
        )

    dataset = MMFiPoseDataset(
        dataset_root=dataset_root,
        split=split,
        seed=seed,
        split_ratios=split_ratios,
    )
    should_shuffle = shuffle if shuffle is not None else split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
    )


def create_data_loaders(
    dataset_root: str | Path,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    split_ratios: Optional[Dict[str, int]] = None,
):
    return {
        split: create_data_loader(
            dataset_root=dataset_root,
            split=split,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            split_ratios=split_ratios,
        )
        for split in SPLIT_NAMES
    }


def summarize_splits(
    dataset_root: str | Path,
    seed: int = 42,
    split_ratios: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, int]]:
    sample_splits = build_sample_splits(dataset_root, seed=seed, split_ratios=split_ratios)
    summary: Dict[str, Dict[str, int]] = {}

    for split_name, sequences in sample_splits.items():
        environments = {sequence.environment for sequence in sequences}
        actions = {sequence.action for sequence in sequences}
        summary[split_name] = {
            "num_sequences": len(sequences),
            "num_frames": len(sequences) * FRAMES_PER_SAMPLE,
            "num_actions": len(actions),
            "num_environments": len(environments),
        }

    return summary


def _preview_sample(dataset: MMFiPoseDataset) -> Dict[str, Tuple[int, ...] | str]:
    sample = dataset[0]
    return {
        "action": sample["action"],
        "sample": sample["sample"],
        "environment": sample["environment"],
        "frame_id": sample["frame_id"],
        "keypoints_shape": tuple(sample["keypoints"].shape),
        "csi_amplitude_shape": tuple(sample["csi_amplitude"].shape),
        "csi_phase_shape": tuple(sample["csi_phase"].shape),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MM-Fi WiFi pose dataloader preview")
    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset root path")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Load one sample from each split and print its shapes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    summary = summarize_splits(dataset_root, seed=args.seed)

    print(f"dataset_root: {dataset_root}")
    for split_name in SPLIT_NAMES:
        split_info = summary[split_name]
        print(
            f"{split_name}: sequences={split_info['num_sequences']}, "
            f"frames={split_info['num_frames']}, "
            f"actions={split_info['num_actions']}, "
            f"environments={split_info['num_environments']}"
        )

    if args.preview:
        for split_name in SPLIT_NAMES:
            dataset = MMFiPoseDataset(dataset_root=dataset_root, split=split_name, seed=args.seed)
            print(f"{split_name}_preview: {_preview_sample(dataset)}")


if __name__ == "__main__":
    main()
