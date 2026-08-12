# Skate-BFM Training

This directory contains data and records for the final M2.5b Skate-BFM
baseline. Training logic is deliberately limited to two project-owned scripts:

```text
scripts/train_skate_bfm.py  strict official BFM0 -> HUSKY closed-loop training
scripts/eval_target.py      frozen target-conditioned fixed evaluation
scripts/isaac_env/          vendored BFM-Zero runtime
```

The final training path is equivalent in role to BFM-Zero's upstream
`humanoidverse/train.py`: it constructs the official FB-CPR-Aux agent, loads
MotionLib experts, grows replay from online interaction, and calls the native
agent update. Project code owns only HUSKY integration, data selection, and
the fixed M2.5b schedule.

## Required Data

```text
dataset/BFM-Zero/train/lafan_29dof_10s-clipped.pkl
dataset/skate-expert-pose/motion_library/skate_expert.pkl
```

The LAFAN source provides Base motions. The Skate MotionLib file provides the
single current Skate expert source. No data conversion or rollout collection
script is retained in this branch.

## Formal Run

```bash
CUDA_VISIBLE_DEVICES=0 \
SKATE_EXPERT_MOTION_FILE=$PWD/train/dataset/skate-expert-pose/motion_library/skate_expert.pkl \
SKATE_WORK_DIR=$PWD/results/m2.5b-original-bfm-baseline \
python train/scripts/train_skate_bfm.py
```

The fixed configuration is 20,000 online transitions, 38 update blocks, 50
native updates per block, and 1,900 updates total. It samples 64 complete Base
sequences and 64 complete Skate sequences per update.

See [train_log.md](train_log.md) for the current engineering record and
[train_res.md](train_res.md) for the completed baseline result.
