# Skate-BFM

Skate-BFM adapts the official BFM-Zero motion prior to one HUSKY MuJoCo
skateboard environment. The current repository has completed the first formal
M2.6 Phase 100k training run: strict BFM0 initialization, Base plus Skate
expert sampling, native FB-CPR-Aux updates, frozen-policy rollout evaluation,
and MuJoCo visual inspection.

![Project progress](docs/assets/project_progress.svg)

![Training progress](docs/assets/development_substage.svg)

## Current Status

- Online training environment: four independent nominal HUSKY MuJoCo
  environments.
- Action contract: 29D BFM action stored in replay; 23D name-mapped HUSKY
  action executed in simulation.
- Expert batch: 1024 rows = 64 Base sequences + 64 Skate sequences, each of
  length 8.
- Initialization: fresh official BFM0 checkpoint, verified against SHA256
  `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- Training: 100,000 online transitions and 9,900 native
  `FBcprAuxAgent.update()` calls. Updates begin at step 1,500 and run every
  500 transitions for 50 updates per block.
- Reset: uniform expert motion and local frame sampling, followed by direct
  raw HUSKY robot-board `qpos/qvel` injection.
- Latent lifecycle: random BFM latent with refresh every 100 transitions.
- Physics: formal training uses nominal HUSKY parameters and no domain
  randomization.
- Checkpoints: `20k`, `50k`, and `100k`, stored under
  `model/motion_library/2026-08-15_143013/`.

Training and checkpoint integrity passed: replay size, optimizer state,
normalizers, model finiteness, and checkpoint reloads are valid. Behavioral
evaluation did not pass: all 32 frozen-policy episodes for each trained
checkpoint ended in fall before the 1024-step horizon, and trained
checkpoints were less stable than the official BFM0 on the same reset seeds.
Continuous 100k training is therefore paused pending diagnosis.

## Setup

Create the project environment and install the project plus its MotionLib
dependencies:

```bash
conda create -n skatebfm python=3.12 -y
conda activate skatebfm
pip install -e '.[dev,motionlib]'
```

The following local artifacts must exist before training:

```text
model/bfm-zero-official/
train/dataset/BFM-Zero/train/lafan_29dof_10s-clipped.pkl
dataset/sim_collected/phase/motion_library/skate_expert_phase.pkl
```

`train/scripts/isaac_env/` is the vendored BFM-Zero runtime used for the
official agent and MotionLib interfaces. `husky_sim/` is the project-owned
HUSKY runtime boundary.

## Build Phase Expert Data

The production collector writes complete, frame-aligned robot-board rollouts.
The dataset processor scans every raw rollout, uses recorded phase IDs for
strict contiguous segmentation, aggregates all accepted motions, validates the
result with official BFM interfaces, and can generate post-hoc full-scene QC:

The completed M2.5c-P BFM-compatible phase MotionLib, manifests, and QC
videos are hosted under
[phase/](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/phase)
in the Skate-BFM Hugging Face dataset. The shared raw collection is stored
beside it under
[raw/](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/raw).
Restore the raw collection with:

```bash
hf download Yak9Ce3teeh/skate-sim-dataset \
  --repo-type dataset \
  --include "raw/**" \
  --local-dir dataset/sim_collected
```

Restore the Phase artifacts with:

```bash
hf download Yak9Ce3teeh/skate-sim-dataset \
  --repo-type dataset \
  --include "phase/**" \
  --local-dir dataset/sim_collected
```

```bash
python train/scripts/data_collection/convert_phase.py \
  --aggregate-phase \
  --dataset-root dataset/sim_collected/phase/raw \
  --bfm-repo $PWD/train/scripts/isaac_env \
  --bfm-reference $PWD/train/dataset/BFM-Zero/train/lafan_29dof_10s-clipped.pkl \
  --robot-xml $PWD/train/scripts/isaac_env/humanoidverse/data/robots/g1/g1_29dof.xml \
  --husky-xml $PWD/husky_sim/upstream/test_scene/mjlab_scene.xml \
  --output dataset/sim_collected/phase/motion_library/skate_expert_phase.pkl \
  --manifest dataset/sim_collected/phase/motion_library/manifest.json \
  --qc-root dataset/sim_collected/phase/qc \
  --validate-motionlib
```

Phase datasets use `convert_phase.py`; fixed-window continuous datasets use
`convert_continuous.py`. Both consume the same canonical raw collection.

The completed M2.5c-C continuous dataset is published under the same Hugging
Face dataset repository at
[continuous/](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/continuous).
Restore it with:

```bash
hf download Yak9Ce3teeh/skate-sim-dataset \
  --repo-type dataset \
  --include "continuous/**" \
  --local-dir dataset/sim_collected
```

The six absent HUSKY wrist joints are explicitly fixed to zero. Each accepted
record retains board state, action, phase annotations, and source provenance.
The converter rejects malformed arrays, cross-boundary motions, incomplete
sequences, and invalid BFM schemas.

## Train

Use a new work directory for each run:

```bash
CUDA_VISIBLE_DEVICES=0 \
SKATE_EXPERT_DATASET=phase \
SKATE_MAX_STEPS=100000 \
SKATE_WORK_DIR=$PWD/results/m2.6-phase-100k-seed4728 \
python train/scripts/train_skate_bfm.py
```

The formal entrypoint uses the 100k schedule and 50/50 expert mixture. Each
new run gets a timestamped checkpoint directory under
`model/motion_library/`; `SKATE_CHECKPOINT_DIR` can override it. The entrypoint
fails closed when the checkpoint, data, replay schema, optimizer state, or
checkpoint reload contract is invalid.

## Evaluate

Evaluate one frozen formal checkpoint with the same expert-reset and latent
lifecycle used by online training. Add `--viewer` for the realtime MuJoCo
window:

```bash
CUDA_VISIBLE_DEVICES=0 python train/scripts/evaluator.py \
  --checkpoint model/motion_library/2026-08-15_143013/checkpoint_100000 \
  --dataset phase \
  --episodes 4 \
  --viewer
```

The historical official/10k/20k target-conditioned protocol remains available
through `--mode fixed-target`.

## Layout

```text
train/scripts/train_skate_bfm.py  M2.5b training entrypoint
train/scripts/evaluator.py        frozen rollout and fixed-target evaluator
train/scripts/data_collection/    HUSKY expert-motion conversion
train/scripts/isaac_env/          vendored BFM-Zero runtime
train/dataset/                    Base LAFAN and Skate MotionLib data
husky_sim/src/skate_husky/        HUSKY MuJoCo runtime and physical contracts
src/skate_bfm/integration/        BFM action/observation/replay adapters
```

Detailed current records are in [train/train_log.md](train/train_log.md) and
[train/train_res.md](train/train_res.md).
