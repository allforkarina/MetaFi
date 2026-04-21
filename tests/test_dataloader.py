from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import loadmat, savemat

import dataloader

torch = pytest.importorskip("torch")


def _create_raw_dataset(root: Path, frames_per_sample: int) -> None:
    action_dir = root / "A01"

    for sample_index in range(1, 41):
        sample_name = f"S{sample_index:02d}"
        rgb_dir = action_dir / sample_name / "rgb"
        csi_dir = action_dir / sample_name / "wifi-csi"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        csi_dir.mkdir(parents=True, exist_ok=True)

        for frame_index in range(1, frames_per_sample + 1):
            frame_name = f"frame{frame_index:03d}"
            keypoints = np.full(
                dataloader.KEYPOINT_SHAPE,
                fill_value=(sample_index * 10 + frame_index),
                dtype=np.float32,
            )
            amplitude = np.full(
                dataloader.CSI_SHAPE,
                fill_value=(sample_index * 100 + frame_index),
                dtype=np.float32,
            )
            phase = np.full(
                dataloader.CSI_SHAPE,
                fill_value=(sample_index * 100 + frame_index + 0.5),
                dtype=np.float32,
            )

            np.save(rgb_dir / f"{frame_name}.npy", keypoints)
            savemat(csi_dir / f"{frame_name}.mat", {"CSIamp": amplitude, "CSIphase": phase})


def test_build_h5_dataset_preserves_shapes_values_and_split_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_root = tmp_path / "dataset"
    output_path = tmp_path / "mmfi_pose.h5"
    monkeypatch.setattr(dataloader, "FRAMES_PER_SAMPLE", 2)
    _create_raw_dataset(raw_root, frames_per_sample=2)

    summary = dataloader.build_h5_dataset(raw_root, output_path, seed=7)

    assert summary == {
        "num_records": 80,
        "num_train_frames": 48,
        "num_val_frames": 16,
        "num_test_frames": 16,
    }

    first_train_record = dataloader.expand_frame_records(
        dataloader.build_sample_splits(raw_root, seed=7)["train"]
    )[0]
    raw_keypoints = np.load(first_train_record.keypoint_path).astype(np.float32)
    raw_csi = loadmat(first_train_record.csi_path)
    raw_amplitude = np.asarray(raw_csi["CSIamp"], dtype=np.float32)
    raw_phase = np.asarray(raw_csi["CSIphase"], dtype=np.float32)

    with h5py.File(output_path, "r") as h5_file:
        assert h5_file["keypoints"].shape == (80, 17, 2)
        assert h5_file["csi_amplitude"].shape == (80, 3, 114, 10)
        assert h5_file["csi_phase"].shape == (80, 3, 114, 10)
        assert h5_file["train_indices"].shape == (48,)
        assert h5_file["val_indices"].shape == (16,)
        assert h5_file["test_indices"].shape == (16,)
        assert h5_file.attrs["frames_per_sample"] == 2

        np.testing.assert_allclose(h5_file["keypoints"][0], raw_keypoints)
        np.testing.assert_allclose(h5_file["csi_amplitude"][0], raw_amplitude)
        np.testing.assert_allclose(h5_file["csi_phase"][0], raw_phase)
        assert h5_file["action"][0].decode("utf-8") == first_train_record.action
        assert h5_file["sample"][0].decode("utf-8") == first_train_record.sample
        assert h5_file["environment"][0].decode("utf-8") == first_train_record.environment
        assert h5_file["frame_id"][0].decode("utf-8") == first_train_record.frame_stem


def test_h5_dataset_and_dataloader_return_expected_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_root = tmp_path / "dataset"
    output_path = tmp_path / "mmfi_pose.h5"
    monkeypatch.setattr(dataloader, "FRAMES_PER_SAMPLE", 2)
    _create_raw_dataset(raw_root, frames_per_sample=2)
    dataloader.build_h5_dataset(raw_root, output_path, seed=11)

    train_dataset = dataloader.MMFiPoseDataset(output_path, split="train")
    val_dataset = dataloader.MMFiPoseDataset(output_path, split="val")
    test_dataset = dataloader.MMFiPoseDataset(output_path, split="test")

    assert len(train_dataset) == 48
    assert len(val_dataset) == 16
    assert len(test_dataset) == 16

    sample = train_dataset[0]

    assert sample["action"] == "A01"
    assert sample["sample"].startswith("S")
    assert sample["environment"].startswith("env")
    assert sample["frame_id"].startswith("frame")
    assert sample["keypoints"].shape == (17, 2)
    assert sample["csi_amplitude"].shape == (3, 114, 10)
    assert sample["csi_phase"].shape == (3, 114, 10)

    loaders = dataloader.create_data_loaders(output_path, batch_size=4, num_workers=0)
    batch = next(iter(loaders["train"]))

    assert batch["keypoints"].shape == (4, 17, 2)
    assert batch["csi_amplitude"].shape == (4, 3, 114, 10)
    assert batch["csi_phase"].shape == (4, 3, 114, 10)

    train_dataset.close()
    val_dataset.close()
    test_dataset.close()
