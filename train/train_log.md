# Training Log

## 0. Workspace Initialization

- Date: 2026-08-03
- Completed: created the dedicated `train` branch for model training, added
  the local dataset directory, and reserved `model/motion_library/` for
  generated model outputs.

## 1. Single Rollout Processing

- Date: 2026-08-03
- Completed: added single-rollout command segmentation, failure/reset cleanup,
  source-format motion export, and synchronized video or MuJoCo pose replay.

## 2. Live HUSKY Phase Inspection

- Date: 2026-08-03
- Completed: verified the official fixed phase schedule, added live phase
  output, classified steering from the heading command, tracked skateboard
  heading changes, and refined fall detection.

## 3. Large-Scale Collection Preparation

- Date: 2026-08-04
- Completed: configured the balanced HUSKY command grid, multi-round parallel
  collection, resumable target-based replacement rollouts, and separate raw
  rollout and cleaned expert-motion duration reporting.
