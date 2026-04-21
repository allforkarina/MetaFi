# MeteFi Agent Guide

## Language Rules

- Write this document in English.
- Communicate with the user in Chinese for answers, summaries, and status updates.

## Documentation Maintenance

- Update this `AGENT.md` whenever an important project decision, workflow change, or implementation constraint is introduced.
- Keep updates concise and practical so future contributors can learn the project rules quickly.

## Development And Training Workflow

- Write and modify code locally in this repository.
- Use the local machine to verify that code can run and that basic checks pass.
- Run full training jobs on the Linux server, not on the local machine.
- Keep the local machine and the Linux server synchronized through Git.
- The current workflow has already been validated: local `git push` and server-side `git pull` work correctly.
- After each project update, push the latest changes to GitHub so the Linux server can stay in sync.
- Treat `git push` to GitHub as a required final step for every project update.
- Local dataset root: `D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset`.
- Linux dataset root: `/data/WiFiPose/dataset/dataset`.

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
- The current repository provides this logic in `dataloader.py`.

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

