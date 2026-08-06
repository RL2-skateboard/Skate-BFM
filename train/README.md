# `train` Branch

This branch owns model-training code, training configuration, data preparation,
checkpoints, and learned motion-library development. It is intentionally
separate from `main`, which contains the BFM0-HUSKY integration and formal H1
evaluation records.

Training datasets belong under [`train/dataset/`](dataset/). Generated
checkpoints, exports, and motion-library artifacts belong under
[`model/motion_library/`](../model/motion_library/).

Training records start from zero in:

- [`train_log.md`](train_log.md): dated work log.
- [`train_res.md`](train_res.md): parameters, checkpoints, metrics, and results.

Large datasets and generated model files are local artifacts and are ignored by
Git.

## Expert rollout collection

Activate the repository environment and run one interactive HUSKY rollout:

```bash
conda activate skatebfm
cd /home/hm/workspace/skate-bfm

python train/scripts/rollout_split.py \
  --live \
  --record \
  --headless \
  --robot-xml husky_sim/upstream/test_scene/mjlab_scene.xml \
  --policy husky_sim/upstream/ckpts/test.onnx \
  --device cpu \
  --rollout-id 001 \
  --episode-id skate_run_001 \
  --dataset-split auto \
  --randomize-physics \
  --output-dir train/dataset/skate-expert-pose
```

With `--headless`, no real-time MuJoCo window is opened, but every phase
`preview.mp4` is still rendered during finalization unless
`--no-render-previews` is used. Omit `--headless` to use the official
interactive viewer and keyboard controls. A confirmed fall, Enter reset,
viewer close, or `--max-policy-frames` ends the rollout.

The checked-in [`rollout_config.json`](scripts/rollout_config.json) defines the
formal 150-minute collection:

```bash
python train/scripts/rollout_split.py --parallel-config
```

The baseline plan uses ten rounds with 15 rollouts per round. Each rollout
targets 3000 frames at 50 Hz, or 60 seconds. Heading commands cover -0.7 through
0.7 in 0.1 steps, and forward commands cover 0.50, 0.75, 1.00, 1.25, and 1.50.
The resulting 150-rollout grid assigns ten rollouts to every heading and 30
rollouts to every velocity. Two workers run concurrently.

The 150-minute target is measured from the actual number of raw frames, not the
planned episode count. Early falls therefore reduce accumulated raw duration.
When the ten baseline rounds do not reach the target, replacement rollouts
are scheduled under additional rounds, up to the configured limit. Motion
categories are accepted in their natural recorded proportions; no category
quota or resampling is applied.

Formal output is organized as:

```text
train/dataset/skate-expert-pose/train/
├── collection_plan.json
├── collection_summary.json
├── round_001/
│   ├── rollout_001/
│   └── ...
└── round_010/
```

`collection_summary.json` reports raw duration separately from cleaned expert
duration. Cleaned expert duration includes only exported non-fall segments
after minimum-duration and failure-margin filtering. Formal collection disables
preview rendering to avoid creating thousands of videos; raw rollout arrays,
phase segments, metadata, and progress files are retained. Re-running the same
command resumes from valid completed rollout summaries.

Bounded collection tests must use the ignored temporary directory rather than
the formal dataset:

```bash
python train/scripts/rollout_split.py --parallel-config \
  --output-dir train/scripts/temp \
  --round-id 900 \
  --round-count 1 \
  --rollouts-per-round 2 \
  --target-raw-minutes 2
```

Edit [`rollout_config.json`](scripts/rollout_config.json) to change the round,
starting rollout ID, headings, frame limit, or output directory. Command-line
arguments override values from the JSON file. The parent process prints the
round, rollout IDs, steering commands, domain-randomization seeds, frame target,
device, policy, and output path before launch. One overall progress bar then
tracks all workers, with each rollout's current frame count and phase shown in
the postfix.

Test batches are grouped without splitting a rollout across directories:

```text
train/scripts/temp/
└── round_001/
    ├── rollout_001/
    └── rollout_002/
```

Numeric round and rollout IDs are zero-padded to at least three digits.
Each rollout also writes `collection_progress.json` atomically so the parent
can report progress without parsing worker logs. At the default 50 Hz and
6-second HUSKY cycle, every complete 300-frame cycle is split exactly into 120
`push`, 30 `push2steer`, 135 `steer`, and 15 `steer2push` frames. Final console
statistics show both raw phase runs/frames and exported segments/frames.

`--dataset-split auto` assigns the complete rollout to a stable 80/10/10
train/validation/test split. Every phase from a rollout remains in the same
split. `--randomize-physics` samples the official HUSKY play-time startup
domain randomization and reset joint offsets once at the beginning of the
rollout; the realization remains fixed for that rollout and is recorded in its
metadata. Use `--physics-seed <int>` for an explicit seed, otherwise the seed is
derived reproducibly from `--rollout-id`. The recorder writes:

```text
train/dataset/skate-expert-pose/
└── <train|validation|test>/rollout_001/
    ├── raw_rollout/<episode_id>.npz
    ├── raw_rollout/<episode_id>.json
    ├── dynamic_motion/<motion_type>/<segment_id>/
    │   ├── pose.npy
    │   ├── state.npz
    │   ├── metadata.json
    │   └── preview.mp4  # when preview rendering is enabled
    └── bfm_motionlib/
        ├── full_rollout/rollout_001.pkl
        ├── subtask_rollouts/<motion_type>/rollout_000.pkl
        ├── failure_rollouts/fall/rollout_000.pkl
        ├── skate_expert.pkl
        └── manifest.json
```

`motion_type` is one of `push`, `push2steer`, `steer_left`,
`steer_right`, `steer_forward`, `steer2push`, or terminal `fall`.
Fall trajectories are retained under `failure_rollouts/` and are never added
to the positive `skate_expert.pkl` expert buffer.

## BFM motion-library conversion

Convert all recorded segments and validate them with the official BFM-Zero
MotionLib and expert-buffer loader:

```bash
ROLLOUT_DIR=$(find train/dataset/skate-expert-pose \
  -mindepth 2 -maxdepth 2 -type d -name rollout_001 -print -quit)

python train/scripts/convert_husky_to_bfm.py \
  --input-root "$ROLLOUT_DIR/dynamic_motion" \
  --bfm-repo model/bfm-zero-source \
  --bfm-reference train/dataset/BFM-Zero/train/lafan_29dof_10s-clipped.pkl \
  --robot-xml model/bfm-zero-source/humanoidverse/data/robots/g1/g1_29dof.xml \
  --output "$ROLLOUT_DIR/bfm_motionlib/skate_expert.pkl" \
  --validate-motionlib
```

The output is a joblib dictionary with the same per-motion schema as the BFM
LAFAN training file. Labels remain in motion keys and `manifest.json`; they are
not appended to BFM observations. `full_rollout/rollout_001.pkl` contains the
continuous source rollout as one BFM motion. Each file under
`subtask_rollouts/` contains exactly one segmented task rollout.
`skate_expert.pkl` is the combined library of all subtask rollouts and does not
duplicate the full rollout or include fall trajectories. Each phase
`state.npz` is a self-contained frame-aligned slice containing commands, phase,
fall/reset flags, humanoid state, and explicit skateboard root and joint state.
