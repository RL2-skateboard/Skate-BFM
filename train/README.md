# Training

`train/` contains Skate-BFM-owned training, evaluation, data-collection, and
dataset-conversion entrypoints. The official BFM-Zero implementation remains
vendored under `scripts/isaac_env/`; it is not copied into a second training
implementation.

## Code Layout

```text
scripts/train_skate_bfm.py                 formal BFM0 + HUSKY training
scripts/evaluator.py                       frozen checkpoint rollout/evaluation
scripts/data_collection/rollout_split.py   raw HUSKY rollout collection
scripts/data_collection/convert_phase.py   phase MotionLib conversion and QC
scripts/data_collection/convert_continuous.py
                                            continuous MotionLib conversion and QC
scripts/data_collection/rollout_config.json collection parameters
scripts/isaac_env/                          vendored BFM-Zero runtime
dataset/base/                               official Base/LAFAN motion
dataset/sim_collected/                      raw, Phase, and Continuous Skate data
```

## Data Source

Formal Skate data is published on Hugging Face:

- [raw collection](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/raw)
- [phase MotionLib](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/phase)
- [continuous MotionLib](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/continuous)

Restore the selected dataset under `train/dataset/`:

```bash
hf download Yak9Ce3teeh/skate-sim-dataset \
  --repo-type dataset \
  --include "raw/**" "phase/**" \
  --local-dir train/dataset/sim_collected
```

Replace `phase/**` with `continuous/**` when needed, but retain `raw/**`
because training resets use the source robot-board state. The official
Base/LAFAN training file is a separate BFM-Zero dependency. Restore it with:

```bash
mkdir -p train/dataset/base
curl -L \
  https://media.githubusercontent.com/media/LeCAR-Lab/BFM-Zero/main/humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  -o train/dataset/base/lafan_29dof_10s-clipped.pkl
```

The official BFM0 checkpoint is a separate local artifact at
`model/bfm-zero-official/`.

## Commands

Collect raw HUSKY data if not downloading from Huggingface:

```bash
python train/scripts/data_collection/rollout_split.py \
  --parallel-config train/scripts/data_collection/rollout_config.json
```

Convert an existing raw collection:

```bash
python train/scripts/data_collection/convert_phase.py \
  --aggregate-phase \
  --dataset-root train/dataset/sim_collected/raw \
  --bfm-repo train/scripts/isaac_env \
  --bfm-reference train/dataset/base/lafan_29dof_10s-clipped.pkl \
  --robot-xml train/scripts/isaac_env/humanoidverse/data/robots/g1/g1_29dof.xml \
  --husky-xml husky_sim/upstream/test_scene/mjlab_scene.xml \
  --output train/dataset/sim_collected/phase/motion_library/skate_expert_phase.pkl \
  --manifest train/dataset/sim_collected/phase/motion_library/manifest.json \
  --qc-root train/dataset/sim_collected/phase/qc \
  --validate-motionlib
```

Build Continuous from the same raw collection:

```bash
python train/scripts/data_collection/convert_continuous.py \
  --aggregate-continuous \
  --dataset-root train/dataset/sim_collected/raw \
  --bfm-repo train/scripts/isaac_env \
  --bfm-reference train/dataset/base/lafan_29dof_10s-clipped.pkl \
  --robot-xml train/scripts/isaac_env/humanoidverse/data/robots/g1/g1_29dof.xml \
  --husky-xml husky_sim/upstream/test_scene/mjlab_scene.xml \
  --output train/dataset/sim_collected/continuous/motion_library/skate_expert_continuous.pkl \
  --manifest train/dataset/sim_collected/continuous/motion_library/manifest.json \
  --qc-root train/dataset/sim_collected/continuous/qc \
  --validate-motionlib
```

Run formal training:

```bash
python train/scripts/train_skate_bfm.py \
  --dataset phase \
  --max-steps 100000 \
  --online-envs 4 \
  --seed 4728 \
  --work-dir results/m2.6-phase-100k-seed4728-v2 \
  --checkpoint-dir model/motion_library/m2.6-phase-100k-seed4728-v2 \
  --pretrained-checkpoint model/bfm-zero-official
```

Use a new `model/motion_library/<model_name>` for every run. Restore the
published diagnostic checkpoint:

```bash
hf download Yak9Ce3teeh/skate-bfm \
  --include "motion_library/m2.6-phase-100k-seed4728/**" \
  --local-dir model
```

Run a frozen checkpoint with the MuJoCo viewer:

```bash
python train/scripts/evaluator.py \
  --checkpoint model/motion_library/m2.6-phase-100k-seed4728/checkpoint_100000 \
  --dataset phase \
  --episodes 4 \
  --horizon 1024 \
  --viewer
```

Use `--mode fixed-target` for the historical target-conditioned evaluator.

## Records

- [train.md](train.md): objectives, methods, issues, and validation status.
- [train_res.md](train_res.md): tables and links to retained artifacts.
- [train_log.md](train_log.md): one short sentence per work date.
