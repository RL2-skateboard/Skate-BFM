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

## 4. Engineering Progress Tracking

- Date: 2026-08-06
- Completed: added a shared project milestone graphic and synchronized
  progress sections for the `train` and `main` branch documentation.

## 5. Training Script Directory Organization

- Date: 2026-08-09
- Completed: moved the project-owned HUSKY rollout collection script and its
  configuration from `train/scripts/` to
  `train/scripts/data_collection/`. Collection plan and summary artifacts were
  placed there as well, and the obsolete `train/scripts/temp/` directory was
  removed. This was directory organization only; no data format, model
  behavior, training logic, or collection parameters were changed.

## 6. M1.1 Skate Expert Integration Test

- Date: 2026-08-09
- Collection script:
  `train/scripts/data_collection/rollout_split.py`
- Collection command:

  ```bash
  python train/scripts/data_collection/rollout_split.py \
    --live --record --headless \
    --robot-xml husky_sim/upstream/test_scene/mjlab_scene.xml \
    --policy husky_sim/upstream/ckpts/test.onnx \
    --device cpu \
    --round-id 901 --rollout-id 001 \
    --episode-id m1_1_rollout_001 \
    --output-dir /tmp/skate_bfm_m1_1.0L6F7i \
    --randomize-physics --physics-seed 20260809 \
    --initial-v 1.0 --initial-h 0.0 \
    --max-policy-frames 50 --status-interval 0.2 \
    --no-render-previews
  ```

- Environment: official HUSKY MuJoCo test scene and ONNX policy on CPU, with
  the existing official HUSKY startup/reset physics randomization.
- Raw rollout:
  `/tmp/skate_bfm_m1_1.0L6F7i/round_901/rollout_001/raw_rollout/m1_1_rollout_001.npz`;
  50 frames at 50 Hz, one second of collection, containing one valid `push`
  segment.
- Conversion script: `train/scripts/convert_husky_to_bfm.py`.
- Converted output:
  `/tmp/skate_bfm_m1_1.0L6F7i/round_901/rollout_001/bfm_motionlib/skate_expert.pkl`;
  one motion, 50 frames, 0.98 seconds of MotionLib duration.
- MotionLib loaded: yes. The official BFM-Zero `MotionLibRobot` loaded both the
  Skate motion and the original 862-motion LAFAN training library.
- Expert batch generated: yes. The official
  `load_expert_trajectories_from_motion_lib()` produced matching Skate and Base
  batches with `state` `[16, 64]`, `last_action` `[16, 29]`, and
  `privileged_state` `[16, 463]`, all finite `torch.float32`.
- Issues recorded without representation changes: HUSKY provides 23 actuated
  joints, so the six BFM wrist joints remain fixed at zero; `smpl_joints` is a
  zero placeholder; and this minimal integration sample covers only `push`,
  not all Skate motion classes.
- Training performed: no. No BFM-Zero model, loss, latent, or optimizer code
  was modified.

## 7. M1.2 Base + Skate Expert Sources

- Date: 2026-08-09
- Original behavior: official BFM-Zero `humanoidverse/train.py` creates one
  MotionLib-backed `expert_slicer`. Agent updates, expert rollout context, and
  prioritization all consume that Base/LAFAN buffer.
- Current sources:
  - The project-owned `train/scripts/train_skate_bfm.py` is based on the
    official BFM-Zero training entry point. The official source directory
    remains unchanged.
  - `train/scripts/isaac_env/humanoidverse/` is the vendored BFM-Zero
    Isaac/HumanoidVerse runtime, including agents, Hydra configuration,
    MotionLib, simulator backends, and robot assets. LAFAN data and checkpoints
    are excluded.
  - The training entry imports its environment config and expert loader from
    this vendored runtime and has no runtime dependency on
    `model/bfm-zero-source/`.
  - `expert_base` is created in the copied `Workspace.train_online()` through
    the original Base MotionLib and observation construction.
  - `expert_skate` is created there only when
    `TrainConfig.skate_expert_motion_file` is set. It uses a separate instance
    of the same official `MotionLibRobot` and the same official expert loader.
  - `expert_slicer` remains an object alias to `expert_base` so existing agent
    updates retain their original behavior. It is not a third expert source.
- Changed files:
  - `train/scripts/train_skate_bfm.py`
  - `train/scripts/isaac_env/`
  - `train/scripts/data_collection/rollout_split.py`
  - `pyproject.toml`
  - `README.md`
  - `train/README.md`
  - `src/skate_bfm/exp/h1_bfm_coverage/core.py`
  - `tests/test_h1_bfm_coverage.py`
  - `train/train_log.md`
- Sampling interface:

  ```python
  batch_base = replay_buffer["expert_base"].sample(batch_size, seq_length)
  batch_skate = replay_buffer["expert_skate"].sample(batch_size, seq_length)
  ```

  Source metadata is stored on each buffer as `source = "base"` or
  `source = "skate"` and is not included in observations or model inputs.
- Validation:
  - Base-only loading and three repeated samples: passed.
  - Skate-only loading and three repeated samples: passed.
  - Both sources initialized simultaneously without replacement: passed.
  - Base source names contain no `skate/` motions; Skate source names all use
    the `skate/` prefix and contain no `fall` motions.
  - Both batches contain `observation` and `next.observation`, each with
    `state` `[16, 64]`, `last_action` `[16, 29]`, and `privileged_state`
    `[16, 463]` as finite `torch.float32`.
- Source statistics:
  - Base: 862 motions, 258,600 source frames at 30 Hz, 8,591.303 seconds in
    official MotionLib, and 430,138 frames in the 50 Hz expert buffer.
  - Skate: one positive `push` motion, 50 frames at 50 Hz, 0.98 seconds, and no
    `fall` motion.
- Not decided in M1.2: Base/Skate training sampling ratio, model adaptation,
  and Base/Skate environment mixing.
- Training entry dependencies: added the official runtime requirements
  `exca`, `POT`, `mediapy`, and `wandb` to the existing `motionlib` optional
  dependency group and installed them in the `skatebfm` environment.
- Independent Isaac runtime: installed the locked Python 3.10 environment
  under `train/scripts/isaac_env/.venv` with Isaac Sim 4.5.0 and Isaac Lab
  2.0.2. The local binary environment is ignored by Git and can be rebuilt
  with `uv sync --locked`.
- Isaac environment validation: launched the vendored headless runtime,
  resolved the G1 XML/USD/mesh assets from
  `train/scripts/isaac_env/humanoidverse/data/robots`, loaded the project LAFAN
  library, created one 29-DoF HumanoidVerse environment, reset it, and
  completed one physics step. No retained test script was added.
- Host warning: Isaac Sim reported duplicate Vulkan ICD entries for the
  NVIDIA GPU. The tested headless environment still initialized and stepped
  successfully, but the host driver installation should be cleaned before
  long training runs.
- Training performed: no. Agent update methods, losses, latent sampling, and
  model architectures were not modified. Temporary validation code was not
  retained.
