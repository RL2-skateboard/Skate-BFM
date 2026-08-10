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

## Engineering Progress

![Skate-BFM engineering progress](../docs/assets/project_progress.svg)

### Current Development Substage

![Skate-BFM development substages](../docs/assets/development_substage.svg)

**Status as of 2026-08-10:** the project-level BFM0-HUSKY foundation and frozen
capability audit are complete. Base and Skate expert sources, M2.1 Skate
online replay, and M2.2a official BFM0 initialization plus expert/replay merge
are complete. M2.2b-1 completed the first controlled F/B-only Skate adaptation
boundary, and M2.2b-2 completed evaluator fidelity and clean-process
reproducibility auditing. Full FB-CPR-Aux training has not started.
M2.2b-3 completed the intended Base+Skate 50/50 expert-mixture treatment at
independent update 1/10/100 boundaries. The result is mixed at early
milestones and favorable on several representation metrics at update 100, but
is not a downstream skate-task success claim.
Interaction-JEPA and predictive closed-loop control are separate downstream
modules, not part of the current Motion Library milestone.

Update this snapshot, [`train_log.md`](train_log.md), and
[`train_res.md`](train_res.md) together whenever the active milestone changes.

Re-run the three independent treatment boundaries:

```bash
for updates in 1 10 100; do
  SKATE_ONLINE_ENV=skate SKATE_UPDATE_MODE=fb_only \
  SKATE_ADAPTATION_UPDATES=$updates SKATE_MAX_STEPS=1024 \
  SKATE_EXPERT_RATIO=0.5 \
  SKATE_EXPERT_MOTION_FILE=$PWD/train/dataset/skate-expert-pose/motion_library/skate_expert.pkl \
  SKATE_WORK_DIR=$PWD/results/m2.2b-3/base_skate_50_50/update_$updates \
  CUDA_VISIBLE_DEVICES=0 \
  python train/scripts/train_skate_bfm.py
done
```

The canonical Skate artifact is a one-motion, 29-DoF, 50-frame, 50 Hz
MotionLib file. Its SHA256 and the 64/64 complete-sequence mixture are
recorded in [`train_res.md`](train_res.md). The generated checkpoints and
evaluation results remain local under `results/`.

## Isaac training runtime

The complete BFM-Zero Isaac/HumanoidVerse runtime used by Skate-BFM is vendored
under [`scripts/isaac_env/`](scripts/isaac_env/). It includes the Python
package, Hydra configuration, MotionLib, simulator backends, G1 XML/USD/mesh
assets, upstream license, dependency manifest, and lockfile. It intentionally
does not contain LAFAN data, Skate data, checkpoints, or generated outputs.

The root `skatebfm` Python 3.12 environment supports HUSKY collection,
conversion, MotionLib loading, and expert-buffer validation. Full IsaacSim
training uses the upstream-locked Python 3.10 runtime:

```bash
cd /home/hm/workspace/skate-bfm/train/scripts/isaac_env
uv sync --locked
```

Launch the project-owned training entry from the repository root after
activating that runtime:

```bash
SKATE_EXPERT_RATIO=0.5 \
SKATE_EXPERT_MOTION_FILE=/absolute/path/to/skate_expert.pkl \
train/scripts/isaac_env/.venv/bin/python train/scripts/train_skate_bfm.py
```

`train_skate_bfm.py` imports `humanoidverse` exclusively from the vendored
runtime. Base LAFAN data remains under `train/dataset/BFM-Zero/`, and Skate
expert data remains under `train/dataset/skate-expert-pose/`.
`SKATE_EXPERT_RATIO` controls the proportion of complete expert sequences
sampled from Skate data and defaults to `0.5`.

Run the formal HUSKY Workspace path without model updates:

```bash
SKATE_ONLINE_ENV=skate \
SKATE_COLLECT_ONLY=1 \
SKATE_MAX_STEPS=64 \
SKATE_WORK_DIR=/tmp/skate-bfm-m22a \
BFM0_PRETRAINED_CHECKPOINT=/home/hm/workspace/skate-bfm/model/bfm-zero-official \
SKATE_EXPERT_MOTION_FILE=/absolute/path/to/skate_expert.pkl \
SKATE_EXPERT_RATIO=0.5 \
train/scripts/isaac_env/.venv/bin/python train/scripts/train_skate_bfm.py
```

This mode uses one `HuskyBfmOnlineEnv`, writes real transitions to
`train_skate`, and keeps `train` as the same-object compatibility alias. It
loads the complete official BFM0 model strictly from
`BFM0_PRETRAINED_CHECKPOINT`, builds Base and configured Skate expert buffers
through an independent one-environment MotionLib context, and runs the merged
expert/replay forward preflight. A Skate resume checkpoint under
`SKATE_WORK_DIR/checkpoint/` takes precedence over pretrained initialization.
New Skate workdirs cannot fall back to random initialization.

The collect-only path keeps the pretrained model in evaluation mode, compares
parameter and buffer fingerprints before and after the run, and blocks
`agent.update()`. The `fb_only` adaptation mode is fail-closed: it directly
calls the vendored `update_fb()` and allows only F/B optimizer steps. Actor,
discriminator, QD, and Qaux updates remain disabled. Skate auxiliary rewards
and native physical termination remain later dependencies.

## Fixed evaluation protocol

All Skate-BFM method comparisons use the checked-in
[`evaluation_protocol.json`](evaluation_protocol.json). It fixes held-out
rollout IDs, seeds, commands, horizon, seen/unseen dynamics realizations,
context schema, physical behavior projection, entropy binning, and the
official Base tracking evaluator entry.

Run the frozen evaluator from the repository root:

```bash
train/scripts/isaac_env/.venv/bin/python \
  train/scripts/evaluate_skate_bfm.py \
  --checkpoint model/bfm-zero-official \
  --output-dir results/fixed-evaluation/bfm0-pretrained
```

The evaluator writes an isolated `eval_skate_transition` buffer, resolved
manifest, metrics JSON, and behavior occupancy data. It computes held-out FB
objective diagnostics directly from the current BFM-Zero equations without
calling `backward()` or an optimizer, plus Skate-BFM matching/retrieval,
physical rollout metrics, seen/unseen dynamics summaries, and fixed
achieved-behavior entropy. Each result also records checkpoint, source,
runtime, resolved-config, rollout-input, and diagnostic-batch fingerprints.
The canonical protocol remains `skate-bfm-fixed-eval-v1`; the M2.2b-2 audit
confirmed that its FB equations match the vendored `update_fb()` equations.

The current lightweight runtime has no reliable native termination, contact,
slippage, force, or command-to-BFM-latent alignment. Those metrics are
reported as unavailable rather than synthesized. BFB context `h`, RFB
`kappa`, and MEBE density fields are schema hooks only; no BFB, RFB, or
FB-MEBE algorithm is implemented here.

## Expert rollout collection

Activate the repository environment and run one interactive HUSKY rollout:

```bash
conda activate skatebfm
cd /home/hm/workspace/skate-bfm

python train/scripts/data_collection/rollout_split.py \
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

The checked-in [`rollout_config.json`](scripts/data_collection/rollout_config.json) defines the
formal 150-minute collection:

```bash
python train/scripts/data_collection/rollout_split.py --parallel-config
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
python train/scripts/data_collection/rollout_split.py --parallel-config \
  --output-dir train/scripts/data_collection \
  --round-id 900 \
  --round-count 1 \
  --rollouts-per-round 2 \
  --target-raw-minutes 2
```

Edit [`rollout_config.json`](scripts/data_collection/rollout_config.json) to change the round,
starting rollout ID, headings, frame limit, or output directory. Command-line
arguments override values from the JSON file. The parent process prints the
round, rollout IDs, steering commands, domain-randomization seeds, frame target,
device, policy, and output path before launch. One overall progress bar then
tracks all workers, with each rollout's current frame count and phase shown in
the postfix.

Test batches are grouped without splitting a rollout across directories:

```text
train/scripts/data_collection/
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
  --bfm-repo train/scripts/isaac_env \
  --bfm-reference train/dataset/BFM-Zero/train/lafan_29dof_10s-clipped.pkl \
  --robot-xml train/scripts/isaac_env/humanoidverse/data/robots/g1/g1_29dof.xml \
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
