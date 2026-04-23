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
            keypoint_base = sample_index * 10 + frame_index
            keypoints = np.empty(dataloader.KEYPOINT_SHAPE, dtype=np.float32)
            keypoints[:, 0] = keypoint_base
            keypoints[:, 1] = keypoint_base * 2
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
    raw_keypoints, cleaned_amplitude, cleaned_phase, phase_cos = dataloader._prepare_raw_frame(
        first_train_record
    )
    train_records = dataloader.expand_frame_records(
        dataloader.build_sample_splits(raw_root, seed=7)["train"]
    )
    expected_train_min, expected_train_max = dataloader._compute_train_amplitude_bounds(train_records)
    expected_keypoint_x_scale, expected_keypoint_y_scale = dataloader._compute_train_keypoint_scales(
        train_records
    )
    expected_normalized_keypoints = dataloader._normalize_keypoints(
        raw_keypoints,
        x_scale=expected_keypoint_x_scale,
        y_scale=expected_keypoint_y_scale,
    )
    expected_normalized_amplitude = dataloader._normalize_csi_amplitude(
        cleaned_amplitude,
        train_min=expected_train_min,
        train_max=expected_train_max,
    )

    with h5py.File(output_path, "r") as h5_file:
        assert h5_file["keypoints"].shape == (80, 17, 2)
        assert h5_file["csi_amplitude"].shape == (80, 3, 114, 10)
        assert h5_file["csi_phase"].shape == (80, 3, 114, 10)
        assert h5_file["csi_phase_cos"].shape == (80, 3, 114, 10)
        assert h5_file["train_indices"].shape == (48,)
        assert h5_file["val_indices"].shape == (16,)
        assert h5_file["test_indices"].shape == (16,)
        assert h5_file.attrs["frames_per_sample"] == 2
        assert h5_file.attrs["amplitude_normalization"] == dataloader.AMPLITUDE_NORMALIZATION_ATTR
        assert h5_file.attrs["amplitude_train_min"] == pytest.approx(expected_train_min)
        assert h5_file.attrs["amplitude_train_max"] == pytest.approx(expected_train_max)
        assert h5_file.attrs["keypoint_normalization"] == dataloader.KEYPOINT_NORMALIZATION_ATTR
        assert h5_file.attrs["keypoint_x_scale"] == pytest.approx(expected_keypoint_x_scale)
        assert h5_file.attrs["keypoint_y_scale"] == pytest.approx(expected_keypoint_y_scale)
        assert h5_file.attrs["phase_cleaning"] == dataloader.PHASE_CLEANING_ATTR

        np.testing.assert_allclose(h5_file["keypoints"][0], expected_normalized_keypoints)
        np.testing.assert_allclose(h5_file["csi_amplitude"][0], expected_normalized_amplitude)
        np.testing.assert_allclose(h5_file["csi_phase"][0], cleaned_phase, atol=1e-6)
        np.testing.assert_allclose(h5_file["csi_phase_cos"][0], phase_cos, atol=1e-6)
        assert h5_file["action"][0].decode("utf-8") == first_train_record.action
        assert h5_file["sample"][0].decode("utf-8") == first_train_record.sample
        assert h5_file["environment"][0].decode("utf-8") == first_train_record.environment
        assert h5_file["frame_id"][0].decode("utf-8") == first_train_record.frame_stem

        train_values = h5_file["csi_amplitude"][h5_file["train_indices"][:]]
        train_keypoints = h5_file["keypoints"][h5_file["train_indices"][:]]
        assert np.isfinite(train_values).all()
        assert np.isfinite(train_keypoints).all()
        assert float(train_values.min()) == pytest.approx(0.0)
        assert float(train_values.max()) == pytest.approx(1.0)
        assert float(train_keypoints[..., 0].max()) == pytest.approx(1.0)
        assert float(train_keypoints[..., 1].max()) == pytest.approx(1.0)


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
    assert sample["csi_phase_cos"].shape == (3, 114, 10)

    loaders = dataloader.create_data_loaders(output_path, batch_size=4, num_workers=0)
    batch = next(iter(loaders["train"]))

    assert batch["keypoints"].shape == (4, 17, 2)
    assert batch["csi_amplitude"].shape == (4, 3, 114, 10)
    assert batch["csi_phase"].shape == (4, 3, 114, 10)
    assert batch["csi_phase_cos"].shape == (4, 3, 114, 10)

    train_dataset.close()
    val_dataset.close()
    test_dataset.close()


def test_build_h5_dataset_cleans_non_finite_amplitude_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_root = tmp_path / "dataset"
    output_path = tmp_path / "mmfi_pose.h5"
    monkeypatch.setattr(dataloader, "FRAMES_PER_SAMPLE", 2)
    _create_raw_dataset(raw_root, frames_per_sample=2)

    corrupted_path = raw_root / "A01" / "S01" / "wifi-csi" / "frame001.mat"
    corrupted = loadmat(corrupted_path)
    amplitude = np.asarray(corrupted["CSIamp"], dtype=np.float32)
    amplitude[0, 0, 0] = -np.inf
    savemat(corrupted_path, {"CSIamp": amplitude, "CSIphase": corrupted["CSIphase"]})

    dataloader.build_h5_dataset(raw_root, output_path, seed=11)

    with h5py.File(output_path, "r") as h5_file:
        assert np.isfinite(h5_file["csi_amplitude"][:]).all()


def test_build_h5_dataset_cleans_non_finite_phase_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_root = tmp_path / "dataset"
    output_path = tmp_path / "mmfi_pose.h5"
    monkeypatch.setattr(dataloader, "FRAMES_PER_SAMPLE", 2)
    _create_raw_dataset(raw_root, frames_per_sample=2)

    corrupted_path = raw_root / "A01" / "S01" / "wifi-csi" / "frame001.mat"
    corrupted = loadmat(corrupted_path)
    phase = np.asarray(corrupted["CSIphase"], dtype=np.float32)
    phase[0, 0, 0] = np.nan
    phase[0, 1, 0] = np.inf
    savemat(corrupted_path, {"CSIamp": corrupted["CSIamp"], "CSIphase": phase})

    dataloader.build_h5_dataset(raw_root, output_path, seed=11)

    with h5py.File(output_path, "r") as h5_file:
        assert np.isfinite(h5_file["csi_phase"][:]).all()
        assert np.isfinite(h5_file["csi_phase_cos"][:]).all()
        assert float(h5_file["csi_phase_cos"][:].min()) >= -1.000001
        assert float(h5_file["csi_phase_cos"][:].max()) <= 1.000001


def test_clean_csi_phase_unwraps_and_removes_linear_trend() -> None:
    subcarriers = np.arange(dataloader.CSI_SHAPE[1], dtype=np.float32)
    linear_phase = 0.2 * subcarriers + 1.5
    wrapped_phase = ((linear_phase + np.pi) % (2 * np.pi)) - np.pi
    phase = np.broadcast_to(
        wrapped_phase.reshape(1, dataloader.CSI_SHAPE[1], 1),
        dataloader.CSI_SHAPE,
    ).astype(np.float32)

    cleaned_phase = dataloader._clean_csi_phase(phase, source=Path("synthetic.mat"))
    phase_cos = dataloader._compute_csi_phase_cos(cleaned_phase)

    assert np.isfinite(cleaned_phase).all()
    assert np.isfinite(phase_cos).all()
    assert float(np.max(np.abs(cleaned_phase))) < 1e-5
    assert float(phase_cos.min()) >= -1.000001
    assert float(phase_cos.max()) <= 1.000001


def test_build_h5_dataset_rejects_non_finite_keypoint_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_root = tmp_path / "dataset"
    output_path = tmp_path / "mmfi_pose.h5"
    monkeypatch.setattr(dataloader, "FRAMES_PER_SAMPLE", 2)
    _create_raw_dataset(raw_root, frames_per_sample=2)

    corrupted_path = raw_root / "A01" / "S01" / "rgb" / "frame001.npy"
    keypoints = np.load(corrupted_path).astype(np.float32)
    keypoints[0, 0] = np.nan
    np.save(corrupted_path, keypoints)

    with pytest.raises(ValueError, match="Keypoints contain non-finite values"):
        dataloader.build_h5_dataset(raw_root, output_path, seed=11)


def test_keypoint_normalize_then_denormalize_restores_values() -> None:
    keypoints = np.array(
        [
            [10.0, 20.0],
            [30.0, 60.0],
        ],
        dtype=np.float32,
    )

    normalized = dataloader._normalize_keypoints(keypoints, x_scale=30.0, y_scale=60.0)
    restored = dataloader.denormalize_keypoints(normalized, x_scale=30.0, y_scale=60.0)

    np.testing.assert_allclose(normalized, np.array([[1.0 / 3.0, 1.0 / 3.0], [1.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(restored, keypoints)
