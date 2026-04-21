from __future__ import annotations

"""MM-Fi dataloader with per-(action, environment) mixed splitting.

This module keeps the split at the sample-sequence level first and then expands
each selected sequence into aligned frame pairs. That prevents frame-level
leakage across train/validation/test sets.
"""

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


# 定义数据集路径、数据集划分
DEFAULT_LOCAL_DATASET_ROOT = Path(r"D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset")
DEFAULT_LINUX_DATASET_ROOT = Path("/data/WiFiPose/dataset/dataset")
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = {"train": 6, "val": 2, "test": 2}
FRAMES_PER_SAMPLE = 297


@dataclass(frozen=True)
class SampleSequence:
    """One sample sequence under Axx/Syy, before expanding it into 297 frames."""

    action: str         # Axx - action
    sample: str         # Syy - sample
    environment: str    # env1 to env4 - define by the id of sample, 10 sample per env
    rgb_dir: Path       # Label directory
    csi_dir: Path       # raw csi directory


@dataclass(frozen=True)
class FrameRecord:
    """One aligned frame pair consisting of pose labels and CSI measurements."""

    action: str             # Axx - action
    sample: str             # Syy - sample
    environment: str        # env1 to env4 - define by the id of sample, 10 sample per env
    frame_stem: str         # Frame identifier - from 001 to 297
    keypoint_path: Path     # Path to pose labels
    csi_path: Path          # Path to CSI measurements


def resolve_dataset_root(dataset_root: Optional[str | Path] = None) -> Path:
    """Resolve the dataset root for the current machine or an explicit override."""

    # Judging the dataset path through the following order
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
    """Map sample ids S01-S40 to env1-env4 in blocks of ten samples."""

    sample_index = int(sample_name[1:])                 # Syy -> yy as id
    environment_index = (sample_index - 1) // 10 + 1    # yy / 10 -> env id
    return f"env{environment_index}"


def _sorted_dirs(root: Path, prefix: str) -> List[Path]:
    """List prefixed directories in lexicographic order for deterministic traversal."""

    return sorted(
        # If path exist and prefix match, then sorted by name
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name,
    )


def discover_sample_sequences(dataset_root: str | Path) -> List[SampleSequence]:
    """Scan the dataset root and collect all available Axx/Syy sample sequences."""

    # root path
    root = resolve_dataset_root(dataset_root)
    sequences: List[SampleSequence] = []

    for action_dir in _sorted_dirs(root, "A"):
        for sample_dir in _sorted_dirs(action_dir, "S"):
            rgb_dir = sample_dir / "rgb"
            csi_dir = sample_dir / "wifi-csi"

            # keypoints or csi directory missing -> Error
            if not rgb_dir.is_dir() or not csi_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected aligned rgb and wifi-csi directories under {sample_dir}"
                )

            # Axx/Syy/envz/rgb & Axx/Syy/envz/wifi-csi, each Sequence is 297 frames
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
    """Split each (action, environment) group with a fixed 6:2:2 sample ratio."""

    ratios = split_ratios or SPLIT_RATIOS       # 数据集划分比例

    if tuple(ratios.keys()) != SPLIT_NAMES:
        raise ValueError(f"Split keys must be exactly {SPLIT_NAMES}, got {tuple(ratios.keys())}")
    if sum(ratios.values()) != 10:
        raise ValueError("Per-environment split ratios must sum to 10 samples")

    # unordered grouping, key is (Axx, envz), each Axx has 4 envz, each envz has 10 samples
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

        # Shuffle inside each (action, environment) group so every split mixes all environments.
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
    """List files in lexicographic order so frame alignment stays deterministic."""

    return sorted(directory.glob(pattern), key=lambda path: path.name)


def expand_frame_records(sequences: Sequence[SampleSequence]) -> List[FrameRecord]:
    """Expand selected sample sequences into frame-level aligned label/CSI records."""

    records: List[FrameRecord] = []
    for sequence in sequences:
        keypoint_files = _sorted_files(sequence.rgb_dir, "*.npy")   # keypoints end
        csi_files = _sorted_files(sequence.csi_dir, "*.mat")        # csi end

        if len(keypoint_files) != len(csi_files):                   # It must be aligned
            raise ValueError(
                f"Mismatched frame count for {sequence.action}/{sequence.sample}: "
                f"{len(keypoint_files)} labels vs {len(csi_files)} CSI files"
            )

        if len(keypoint_files) != FRAMES_PER_SAMPLE:
            raise ValueError(
                f"Expected {FRAMES_PER_SAMPLE} frames for {sequence.action}/{sequence.sample}, "
                f"found {len(keypoint_files)}"
            )

        # Frame names must stay aligned because one pose frame matches one CSI frame.
        for keypoint_path, csi_path in zip(keypoint_files, csi_files):
            if keypoint_path.stem != csi_path.stem:
                raise ValueError(
                    f"Frame mismatch for {sequence.action}/{sequence.sample}: "
                    f"{keypoint_path.name} vs {csi_path.name}"
                )

            # pair csi and keypoints as one record
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
    """Frame-level dataset that returns aligned pose labels and CSI tensors."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        seed: int = 42,
        split_ratios: Optional[Dict[str, int]] = None,
    ) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split}")

        self.dataset_root = resolve_dataset_root(dataset_root)  # dataset root path
        self.split = split                                      # split as train/val/test
        self.seed = seed                                        # random seed 
        self.split_ratios = split_ratios or SPLIT_RATIOS        # split ratios, default 6:2:2

        # First split by sample sequence, then expand the selected sequences into frames.
        self.sample_splits = build_sample_splits(
            self.dataset_root, seed=self.seed, split_ratios=self.split_ratios
        )
        self.sample_sequences = self.sample_splits[self.split]
        self.records = expand_frame_records(self.sample_sequences)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray | str]:
        """Load one frame's keypoints and CSI amplitude/phase arrays."""

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
    """Create one PyTorch DataLoader for the requested split."""

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
    """Create train/val/test DataLoaders with the same split configuration."""

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


# ================ Below are utility functions for CLI preview and sanity checks ================

def summarize_splits(
    dataset_root: str | Path,
    seed: int = 42,
    split_ratios: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, int]]:
    """Return dataset statistics after applying the sample-level split."""

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
    """Load the first sample of a split and expose only shape-level information."""

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
    """Parse CLI arguments for split summary and optional sample preview."""

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
    """Print split statistics and, optionally, one loaded sample per split."""

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
