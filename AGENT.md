# MeteFi Agent Guide

## Language Rules

- Write this document in English.
- Communicate with the user in Chinese for answers, summaries, and status updates.

## Documentation Maintenance

- Update this `AGENT.md` whenever an important project decision, workflow change, or implementation constraint is introduced.
- Keep updates concise and practical so future contributors can learn the project rules quickly.

## Development And Training Workflow

- Write and modify code locally in this repository.
- Activate the dedicated Conda environment with `conda activate WiFiPose` before running project commands whenever environment-dependent execution is needed.
- If required packages are missing in `WiFiPose`, install them in that environment as part of the task.
- Use the GPU-enabled PyTorch package already installed in the `WiFiPose` environment for model development, testing, and training-related execution.
- Use the local machine to verify that code can run and that basic checks pass.
- Run full training jobs on the Linux server, not on the local machine.
- Keep the local machine and the Linux server synchronized through Git.
- The current workflow has already been validated: local `git push` and server-side `git pull` work correctly.
- After each project update, push the latest changes to GitHub so the Linux server can stay in sync.
- Treat `git push` to GitHub as a required final step for every project update.
- Local dataset root: `D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset`.
- Linux dataset root: `/data/WiFiPose/dataset/dataset`.
- To avoid repeated `.mat` parsing during training, prepack the raw dataset into one HDF5 file before formal training.
- Generate the HDF5 dataset on the Linux server and use that HDF5 file for full training runs.

## Project Goal

- Reproduce a paper project that performs high-accuracy human pose estimation from WiFi CSI data using the MM-Fi dataset.

## Dataset Structure

- Dataset roots point to the top-level directory that contains action folders `A01` through `A27`.
- Each action folder contains sample folders `S01` through `S40`.
- Every ten samples correspond to one environment: `S01-S10` for environment 1, `S11-S20` for environment 2, `S21-S30` for environment 3, and `S31-S40` for environment 4.
- Label files are stored as `Axx/Syy/rgb/framexxx.npy`.
- Each label file contains one frame of pose annotations with shape `17 x 2`, ordered by the COCO 17-keypoint convention.
- In the `17 x 2` label format, `17` is the number of body keypoints and `2` corresponds to the `x` and `y` coordinates for each keypoint.
- CSI files are stored as `Axx/Syy/wifi-csi/framexxx.mat`.
- CSI data uses shape `3 x 114 x 10`, where `3` is the number of antennas, `114` is the number of subcarriers, and `10` is the number of time snapshots aligned to one pose frame.
- The last CSI dimension is an alignment-oriented temporal dimension added so one pose frame corresponds to 10 CSI snapshots.
- CSI and label frames are aligned one-to-one.
- A sample sequence contains 297 frames for both labels and CSI.

## Data Split Protocol

- Split the dataset at the sample-sequence level, not at the frame level, to avoid leakage between training, validation, and test sets.
- For each `(action, environment)` group, split the 10 samples with ratio `6:2:2` into train, validation, and test subsets.
- After sample-level splitting, expand each selected sample sequence into its 297 aligned frame pairs.
- The environment setting is mixed across all subsets because every action contributes samples from every environment to train, validation, and test.
- Materialize the train/validation/test frame indices during HDF5 preprocessing instead of recomputing them at training time.
- The current repository provides this logic in `dataloader.py` and `scripts/build_h5_dataset.py`.

## HDF5 Dataset Format

- The formal training dataset is a single HDF5 file, not a directory of individual `.npy` and `.mat` files.
- Store `keypoints`, `csi_amplitude`, `csi_phase`, `csi_phase_cos`, `action`, `sample`, `environment`, and `frame_id` in the HDF5 file.
- Store `train_indices`, `val_indices`, and `test_indices` in the same HDF5 file so training can load splits directly.
- Clean `csi_amplitude` during HDF5 packing before training use; replace frame-local non-finite amplitude values with finite bounds from the same frame.
- Normalize `csi_amplitude` with one global min-max computed from the cleaned train split only, then apply that same normalization to train, validation, and test frames.
- The `csi_amplitude` stored in HDF5 is the cleaned and normalized training input, not the untouched raw amplitude array.
- Clean `csi_phase` during HDF5 packing by interpolating non-finite values along subcarriers, unwrapping phase along the subcarrier dimension, and removing each antenna/time-snapshot subcarrier linear trend plus mean offset.
- Store `csi_phase_cos = cos(csi_phase)` after phase cleaning so later model experiments can use a bounded phase-derived feature without recomputing it online.
- Normalize `keypoints` with one train-split global axis-wise scale: divide x coordinates by `train_keypoint_x_max` and y coordinates by `train_keypoint_y_max`.
- Store the normalized keypoints in HDF5 and save the normalization metadata in attrs as `keypoint_normalization = train_axis_max`, `keypoint_x_scale`, and `keypoint_y_scale`.
- Keep the HDF5 file as a training-plus-diagnosis artifact: retain `action`, `sample`, `environment`, and `frame_id` together with amplitude and cleaned phase features.

## Shared CNN Encoder

- The first model-stage CSI encoder uses amplitude-only input from `csi_amplitude`.
- Cleaned phase features are available from the dataloader as `csi_phase` and `csi_phase_cos`, but the current WPFormer training path does not consume them yet.
- The Shared CNN module expects one frame of CSI input with shape `[B, 3, 114, 10]`.
- The input is split into three single-antenna branches, one branch per receiving antenna.
- Each branch is upsampled from `[B, 1, 114, 10]` to `[B, 1, 136, 32]` with bilinear interpolation.
- All three branches pass through the same CNN backbone instance, so weights are shared across antennas.
- The shared backbone output for each branch is `[B, 512, 17, 4]`.
- Concatenate the three branch outputs along the last dimension to produce `[B, 512, 17, 12]`.
- This Shared CNN implementation lives in `models/shared_cnn.py`.

## Transformer And Decoder Module

- The Transformer and decoder stage consumes Shared CNN features with shape `[B, 512, 17, 12]`.
- Flatten the last two dimensions so the attention input becomes `[B, 512, 204]`.
- Use a learned positional embedding with shape `[1, 512, 204]` and add it directly to the flattened features.
- The self-attention stage uses one layer with 3 heads and treats `512` as the token dimension and `204` as the embedding dimension.
- The attention implementation follows the paper-specific design: compute attention matrices for each head, average the attention matrices across heads, and then apply the averaged attention to the value features.
- Use instance normalization after the attention residual connection.
- Reshape the attention output back to `[B, 512, 17, 12]` before decoding.
- The decoder uses `3x3 Conv: 512 -> 32`, then `1x1 Conv: 32 -> 2`, producing `[B, 2, 17, 12]`.
- Average only across the last dimension of size `12`, then transpose to produce the final pose output shape `[B, 17, 2]`.
- This implementation lives in `models/transformer_decoder.py`.

## WPFormer Wrapper And Training Utilities

- The top-level end-to-end model is `WPFormer`, which chains `SharedCNN` and `TransformerDecoderModule`.
- `WPFormer` takes CSI amplitude input with shape `[B, 3, 114, 10]` and outputs pose predictions with shape `[B, 17, 2]`.
- Keep `WPFormer` as the default amplitude-only baseline model and preserve the default training path as amplitude-only.
- The amp-plus-phase model is `WPFormerAmpPhase`, which uses one independent `SharedCNN` for `csi_amplitude` and another independent `SharedCNN` for `csi_phase_cos`.
- `WPFormerAmpPhase` concatenates the two encoder outputs along the channel dimension, uses a `1x1` convolution to fuse `1024 -> 512` channels, and then reuses the same `TransformerDecoderModule`.
- Use `train.py --input-mode amp` for the default amplitude-only baseline and `train.py --input-mode amp_phase` to train with both amplitude and cleaned phase-cosine features.
- The training loss is plain coordinate-space MSE between predictions and labels.
- The MSE loss is computed in the normalized keypoint coordinate space stored in HDF5.
- Do not add pose adjacency matrix constraints to the loss, because the paper reports that such constraints hurt performance for MetaFi++.
- The main evaluation metric is torso-normalized PCK.
- Use the Euclidean distance between the right shoulder and left hip ground-truth keypoints as the torso length reference.
- Under the COCO 17-keypoint order used in this project, the right shoulder index is `6` and the left hip index is `11`.
- Default evaluation thresholds are `PCK@10` through `PCK@50`, implemented as normalized thresholds `0.10` through `0.50`.
- Before computing PCK, denormalize both predictions and targets back to the original pixel-coordinate space using the HDF5 keypoint scale attrs.
- The default optimizer preparation follows the paper setting: SGDM with batch size `32`, learning rate `0.001`, momentum `0.9`, and lambda-based decay to zero.

## Trainer Engine

- The training engine uses `WPFormer`, `train_loader`, `val_loader`, and an explicit target device.
- Local execution is only for code-path validation and short smoke tests; full 50-epoch training runs on the Linux server.
- The training dataloaders read from the prepacked HDF5 dataset, not from the raw per-frame `.npy` and `.mat` files.
- The default training schedule is 50 epochs with SGDM and lambda-based decay to zero.
- Each training epoch records average `train_loss`, and each validation epoch records average `val_loss` plus `PCK@10` through `PCK@50`.
- Show per-batch `tqdm` progress bars for both training and validation so batch throughput can be observed during interactive runs.
- Keep `tqdm` progress bars disabled automatically in non-interactive or redirected-log runs such as `nohup`, so log files stay readable.
- The trainer tracks `best_val_pck50` and saves the best checkpoint only.
- Loss visualization is required: after each epoch, update a loss curve plot that shows both train loss and val loss across epochs.
- Save the loss curve image inside the training output directory so each training run keeps its own loss-trend figure.
- Keep reusable diagnostic utilities under `tests/` when the user explicitly asks to preserve them for future server-side debugging.

## Code Change Principles

- Preserve code readability in every change.
- Follow the principle of minimal modification: change only what is required for the current task.
- Avoid speculative features, premature abstractions, and unrelated refactors.
- Match the existing project style unless a change is necessary for correctness.
- If a change introduces unused code or imports, clean up only what is made obsolete by that change.

## Working Style Constraints

- Think before coding: state assumptions clearly and do not hide uncertainty.
- Prefer the simplest solution that fully satisfies the request.
- Make surgical edits and avoid touching unrelated files or logic.
- Define clear success criteria for each meaningful task and verify the result.

