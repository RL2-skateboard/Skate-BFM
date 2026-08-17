# Training

`train/` contains Skate-BFM-owned training, evaluation, data-collection, and
dataset-conversion entrypoints. The official BFM-Zero implementation remains
vendored under `scripts/isaac_env/`; it is not copied into a second training
implementation.

## Code Map

```text
scripts/train_skate_bfm.py                 formal BFM0 + HUSKY training
scripts/evaluator.py                       frozen checkpoint rollout/evaluation
scripts/data_collection/rollout_split.py   raw HUSKY rollout collection
scripts/data_collection/convert_phase.py   phase MotionLib conversion and QC
scripts/data_collection/convert_continuous.py
                                            continuous MotionLib conversion and QC
scripts/data_collection/rollout_config.json collection parameters
scripts/isaac_env/                          vendored BFM-Zero runtime
```

## Data Source

Do not use the checked-in `train/dataset/skate-expert-pose/` files as the
formal Skate expert source. Formal Skate data is published on Hugging Face:

- [raw collection](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/raw)
- [phase MotionLib](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/phase)
- [continuous MotionLib](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/continuous)

Restore the selected dataset into the repository-level `dataset/` directory:

```bash
hf download Yak9Ce3teeh/skate-sim-dataset \
  --repo-type dataset \
  --include "phase/**" \
  --local-dir dataset/sim_collected
```

Replace `phase/**` with `continuous/**` or `raw/**` when needed. The official
Base/LAFAN training file remains the BFM-Zero dependency at
`train/dataset/BFM-Zero/train/lafan_29dof_10s-clipped.pkl`; it is not a Skate
dataset. The official BFM0 checkpoint is a separate local artifact at
`model/bfm-zero-official/`.

## Commands

Collect raw HUSKY data:

```bash
python train/scripts/data_collection/rollout_split.py \
  --parallel-config train/scripts/data_collection/rollout_config.json
```

Convert an existing raw collection:

```bash
python train/scripts/data_collection/convert_phase.py --help
python train/scripts/data_collection/convert_continuous.py --help
```

Run formal training:

```bash
CUDA_VISIBLE_DEVICES=0 \
SKATE_EXPERT_DATASET=phase \
SKATE_MAX_STEPS=100000 \
SKATE_WORK_DIR=$PWD/results/m2.6-phase-100k \
python train/scripts/train_skate_bfm.py
```

Run a frozen checkpoint with the MuJoCo viewer:

```bash
CUDA_VISIBLE_DEVICES=0 python train/scripts/evaluator.py \
  --checkpoint model/motion_library/YYYY-MM-DD_HHMMSS/checkpoint_100000 \
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
