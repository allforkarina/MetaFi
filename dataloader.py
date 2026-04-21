from __future__ import annotations

"""HDF5-backed dataloader and raw-dataset packing utilities for MM-Fi pose data."""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

try:
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - handled at runtime when torch is unavailable.
    DataLoader = None


DEFAULT_LOCAL_DATASET_ROOT = Path(r"D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset")
DEFAULT_LINUX_DATASET_ROOT = Path("/data/WiFiPose/dataset/dataset")
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = {"train": 6, "val": 2, "test": 2}
FRAMES_PER_SAMPLE = 297
KEYPOINT_SHAPE = (17, 2)
CSI_SHAPE = (3, 114, 10)


@dataclass(frozen=True)
class SampleSequence:
    """One sample sequence under Axx/Syy before expanding it into aligned frames."""

    action: str             # Axx  : A01 - A27
    sample: str             # Syy  : S01 - S40
    environment: str        # envz : env1 - env4, mapped from Syy in blocks of ten samples
    rgb_dir: Path
    csi_dir: Path


@dataclass(frozen=True)
class FrameRecord:
    """One aligned frame pair consisting of pose labels and CSI measurements."""

    action: str             # Axx  : A01 - A27
    sample: str             # Syy  : S01 - S40
    environment: str        # envz : env1 - env4, mapped from Syy in blocks of ten samples
    frame_stem: str         # frame001 - frame297
    keypoint_path: Path
    csi_path: Path


def resolve_dataset_root(dataset_root: Optional[str | Path] = None) -> Path:
    """Resolve the raw MM-Fi dataset root from an override or machine-specific defaults."""

    if dataset_root is not None:
        root = Path(dataset_root)
    elif DEFAULT_LOCAL_DATASET_ROOT.exists():
        root = DEFAULT_LOCAL_DATASET_ROOT
    else:
        root = DEFAULT_LINUX_DATASET_ROOT

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    
    return root


def resolve_h5_dataset_path(dataset_root: str | Path) -> Path:
    """Resolve the prepacked HDF5 dataset path used for training."""

    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"HDF5 dataset does not exist: {dataset_path}")
    if dataset_path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"Expected an HDF5 dataset path, got: {dataset_path}")
    return dataset_path


def sample_to_environment(sample_name: str) -> str:
    """Map sample ids S01-S40 to env1-env4 in blocks of ten samples."""

    sample_index = int(sample_name[1:])
    environment_index = (sample_index - 1) // 10 + 1
    return f"env{environment_index}"


def _sorted_dirs(root: Path, prefix: str) -> List[Path]:
    """List prefixed directories in lexicographic order for deterministic traversal."""

    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name,
    )


def _sorted_files(directory: Path, pattern: str) -> List[Path]:
    """List files in lexicographic order so frame alignment stays deterministic."""

    return sorted(directory.glob(pattern), key=lambda path: path.name)


def discover_sample_sequences(dataset_root: str | Path) -> List[SampleSequence]:
    """Scan the raw dataset root and collect all available Axx/Syy sample sequences."""

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
    """Split each (action, environment) group with a fixed 6:2:2 sample ratio."""

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


def expand_frame_records(sequences: Sequence[SampleSequence]) -> List[FrameRecord]:
    """Expand selected sample sequences into frame-level aligned label/CSI records."""

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


def _decode_string(value: str | bytes) -> str:
    """Normalize HDF5 string values to plain Python strings."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_raw_frame(record: FrameRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one aligned raw frame and validate its expected shapes."""

    keypoints = np.load(record.keypoint_path).astype(np.float32)
    csi_data = loadmat(record.csi_path)
    csi_amplitude = np.asarray(csi_data["CSIamp"], dtype=np.float32)
    csi_phase = np.asarray(csi_data["CSIphase"], dtype=np.float32)

    if keypoints.shape != KEYPOINT_SHAPE:
        raise ValueError(f"Unexpected keypoint shape for {record.keypoint_path}: {keypoints.shape}")
    if csi_amplitude.shape != CSI_SHAPE:
        raise ValueError(f"Unexpected CSI amplitude shape for {record.csi_path}: {csi_amplitude.shape}")
    if csi_phase.shape != CSI_SHAPE:
        raise ValueError(f"Unexpected CSI phase shape for {record.csi_path}: {csi_phase.shape}")

    return keypoints, csi_amplitude, csi_phase


def build_h5_dataset(
    dataset_root: str | Path,
    output_path: str | Path,
    seed: int = 42,
    split_ratios: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Pack the raw MM-Fi dataset into a single HDF5 file with materialized split indices."""

    source_root = resolve_dataset_root(dataset_root)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    sample_splits = build_sample_splits(source_root, seed=seed, split_ratios=split_ratios)
    split_records = {
        split_name: expand_frame_records(sample_splits[split_name]) for split_name in SPLIT_NAMES
    }
    total_records = sum(len(records) for records in split_records.values())
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(target_path, "w") as h5_file:
        keypoints_dataset = h5_file.create_dataset(
            "keypoints", shape=(total_records, *KEYPOINT_SHAPE), dtype=np.float32
        )
        amplitude_dataset = h5_file.create_dataset(
            "csi_amplitude", shape=(total_records, *CSI_SHAPE), dtype=np.float32
        )
        phase_dataset = h5_file.create_dataset(
            "csi_phase", shape=(total_records, *CSI_SHAPE), dtype=np.float32
        )
        action_dataset = h5_file.create_dataset("action", shape=(total_records,), dtype=string_dtype)
        sample_dataset = h5_file.create_dataset("sample", shape=(total_records,), dtype=string_dtype)
        environment_dataset = h5_file.create_dataset(
            "environment", shape=(total_records,), dtype=string_dtype
        )
        frame_dataset = h5_file.create_dataset("frame_id", shape=(total_records,), dtype=string_dtype)

        h5_file.attrs["source_root"] = str(source_root)
        h5_file.attrs["seed"] = seed
        h5_file.attrs["frames_per_sample"] = FRAMES_PER_SAMPLE

        offset = 0
        with tqdm(total=total_records, desc="Packing HDF5", dynamic_ncols=True) as progress_bar:
            for split_name in SPLIT_NAMES:
                records = split_records[split_name]
                indices = np.arange(offset, offset + len(records), dtype=np.int64)
                h5_file.create_dataset(f"{split_name}_indices", data=indices)

                for local_index, record in enumerate(records):
                    dataset_index = offset + local_index
                    keypoints, csi_amplitude, csi_phase = _load_raw_frame(record)

                    keypoints_dataset[dataset_index] = keypoints
                    amplitude_dataset[dataset_index] = csi_amplitude
                    phase_dataset[dataset_index] = csi_phase
                    action_dataset[dataset_index] = record.action
                    sample_dataset[dataset_index] = record.sample
                    environment_dataset[dataset_index] = record.environment
                    frame_dataset[dataset_index] = record.frame_stem
                    progress_bar.update(1)

                offset += len(records)

    return {
        "num_records": total_records,
        "num_train_frames": len(split_records["train"]),
        "num_val_frames": len(split_records["val"]),
        "num_test_frames": len(split_records["test"]),
    }


class MMFiPoseDataset:
    """Frame-level HDF5 dataset that returns aligned pose labels and CSI tensors."""

    def __init__(self, dataset_root: str | Path, split: str) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split}")

        self.dataset_root = resolve_h5_dataset_path(dataset_root)
        self.split = split
        self._h5_file: h5py.File | None = None

        with h5py.File(self.dataset_root, "r") as h5_file:
            self.indices = np.asarray(h5_file[f"{split}_indices"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def _get_h5_file(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.dataset_root, "r")
        return self._h5_file

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup.
        self.close()

    def __getitem__(self, index: int) -> Dict[str, np.ndarray | str]:
        """Load one frame's keypoints and CSI amplitude/phase arrays from HDF5."""

        h5_file = self._get_h5_file()
        frame_index = int(self.indices[index])

        return {
            "action": _decode_string(h5_file["action"][frame_index]),
            "sample": _decode_string(h5_file["sample"][frame_index]),
            "environment": _decode_string(h5_file["environment"][frame_index]),
            "frame_id": _decode_string(h5_file["frame_id"][frame_index]),
            "keypoints": np.asarray(h5_file["keypoints"][frame_index], dtype=np.float32),
            "csi_amplitude": np.asarray(h5_file["csi_amplitude"][frame_index], dtype=np.float32),
            "csi_phase": np.asarray(h5_file["csi_phase"][frame_index], dtype=np.float32),
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
    """Create one PyTorch DataLoader for the requested HDF5 split."""

    if DataLoader is None:
        raise ImportError(
            "PyTorch is not installed in the current environment. "
            "Install torch to create DataLoader instances."
        )

    del seed, split_ratios

    dataset = MMFiPoseDataset(dataset_root=dataset_root, split=split)
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
    """Create train/val/test DataLoaders from the same HDF5 dataset."""

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


def summarize_splits(dataset_root: str | Path) -> Dict[str, Dict[str, int]]:
    """Return split statistics for the prepacked HDF5 dataset."""

    dataset_path = resolve_h5_dataset_path(dataset_root)
    summary: Dict[str, Dict[str, int]] = {}

    with h5py.File(dataset_path, "r") as h5_file:
        action_dataset = h5_file["action"]
        environment_dataset = h5_file["environment"]

        for split_name in SPLIT_NAMES:
            indices = np.asarray(h5_file[f"{split_name}_indices"], dtype=np.int64)
            actions = {_decode_string(action_dataset[index]) for index in indices}
            environments = {_decode_string(environment_dataset[index]) for index in indices}
            summary[split_name] = {
                "num_frames": len(indices),
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
    """Parse CLI arguments for HDF5 split summary and optional sample preview."""

    parser = argparse.ArgumentParser(description="MM-Fi HDF5 dataloader preview")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the HDF5 dataset file")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Load one sample from each split and print its shapes",
    )
    return parser.parse_args()


def main() -> None:
    """Print HDF5 split statistics and, optionally, one loaded sample per split."""

    args = parse_args()
    dataset_path = resolve_h5_dataset_path(args.dataset_root)
    summary = summarize_splits(dataset_path)

    print(f"dataset_root: {dataset_path}")
    for split_name in SPLIT_NAMES:
        split_info = summary[split_name]
        print(
            f"{split_name}: frames={split_info['num_frames']}, "
            f"actions={split_info['num_actions']}, "
            f"environments={split_info['num_environments']}"
        )

    if args.preview:
        for split_name in SPLIT_NAMES:
            dataset = MMFiPoseDataset(dataset_root=dataset_path, split=split_name)
            print(f"{split_name}_preview: {_preview_sample(dataset)}")
            dataset.close()


if __name__ == "__main__":
    main()
