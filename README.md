# Skate-BFM

Skate-BFM adapts the official BFM-Zero motion prior to one HUSKY MuJoCo
skateboard environment. The current repository contains the final M2.5b
baseline: strict BFM0 initialization, Base plus Skate expert sampling, native
FB-CPR-Aux updates, and fixed target-conditioned evaluation.

![Project progress](docs/assets/project_progress.svg)

![Training progress](docs/assets/development_substage.svg)

## Current Baseline

- Online environment: one nominal HUSKY MuJoCo environment.
- Action contract: 29D BFM action stored in replay; 23D name-mapped HUSKY
  action executed in simulation.
- Expert batch: 1024 rows = 64 Base sequences + 64 Skate sequences, each of
  length 8.
- Initialization: fresh official BFM0 checkpoint, verified against SHA256
  `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- Training: 20,000 online transitions; native `FBcprAuxAgent.update()` begins
  at step 1,500 and runs every 500 transitions for 50 updates per block.
- Checkpoints: saved and reloaded at 10,000 and 20,000 transitions.
- Physics: training uses nominal HUSKY parameters. Fixed evaluation uses the
  official HUSKY play-time randomization, deterministically seeded per rollout.

The completed baseline produced 20,000 replay rows and 1,900 native updates.
Checkpoint reload passed. Its fixed evaluation is `INCONCLUSIVE`: all 60
episodes reached the confirmed native-fall terminal condition before the
128-step horizon, so displacement is not claimed as task success.

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
train/dataset/skate-expert-pose/motion_library/skate_expert.pkl
```

`train/scripts/isaac_env/` is the vendored BFM-Zero runtime used for the
official agent and MotionLib interfaces. `husky_sim/` is the project-owned
HUSKY runtime boundary.

## Train

Use a new work directory for each run:

```bash
CUDA_VISIBLE_DEVICES=0 \
SKATE_EXPERT_MOTION_FILE=$PWD/train/dataset/skate-expert-pose/motion_library/skate_expert.pkl \
SKATE_WORK_DIR=$PWD/results/m2.5b-original-bfm-baseline \
python train/scripts/train_skate_bfm.py
```

The entrypoint accepts only the M2.5b 20k schedule and 50/50 expert mixture.
It fails closed when the checkpoint, data, replay schema, optimizer state, or
checkpoint reload contract is invalid.

## Evaluate

After a successful training run, evaluate the official BFM0, 10k checkpoint,
and 20k checkpoint without updating any training state:

```bash
CUDA_VISIBLE_DEVICES=0 python train/scripts/eval_target.py \
  --official-checkpoint model/bfm-zero-official \
  --checkpoint-10k results/m2.5b-original-bfm-baseline/checkpoint_10000 \
  --checkpoint-20k results/m2.5b-original-bfm-baseline/checkpoint_20000 \
  --training-summary results/m2.5b-original-bfm-baseline/summary.json \
  --output-dir results/m2.5b-original-bfm-baseline/fixed_eval
```

## Layout

```text
train/scripts/train_skate_bfm.py  M2.5b training entrypoint
train/scripts/eval_target.py      frozen fixed evaluator
train/scripts/isaac_env/          vendored BFM-Zero runtime
train/dataset/                    Base LAFAN and Skate MotionLib data
husky_sim/src/skate_husky/        HUSKY MuJoCo runtime and physical contracts
src/skate_bfm/integration/        BFM action/observation/replay adapters
```

Detailed current records are in [train/train_log.md](train/train_log.md) and
[train/train_res.md](train/train_res.md).
