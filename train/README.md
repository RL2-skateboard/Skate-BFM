# Skate-BFM Training

This directory contains data and records for formal M2.6 Skate-BFM training.
Training logic is deliberately limited to two project-owned scripts:

```text
scripts/train_skate_bfm.py  strict official BFM0 -> HUSKY closed-loop training
scripts/evaluator.py        frozen rollout and fixed-target evaluation
scripts/data_collection/rollout_split.py  canonical HUSKY raw rollout collection
scripts/data_collection/convert_phase.py  phase dataset build, validation, and QC
scripts/data_collection/convert_continuous.py  continuous dataset build, validation, and QC
scripts/data_collection/rollout_config.json  parallel collection configuration
scripts/isaac_env/          vendored BFM-Zero runtime
```

The final training path is equivalent in role to BFM-Zero's upstream
`humanoidverse/train.py`: it constructs the official FB-CPR-Aux agent, loads
MotionLib experts, grows replay from online interaction, and calls the native
agent update. Project code owns only HUSKY integration, data selection, and
the parameterized M2.6 schedule.

Completed checkpoints are organized under
`model/motion_library/YYYY-MM-DD_HHMMSS/`. Evaluate any compatible checkpoint
by passing its directory to `scripts/evaluator.py --checkpoint`.

## Required Data

```text
dataset/BFM-Zero/train/lafan_29dof_10s-clipped.pkl
dataset/sim_collected/phase/motion_library/skate_expert_phase.pkl
dataset/sim_collected/phase/motion_library/manifest.json
dataset/sim_collected/continuous/motion_library/skate_expert_continuous.pkl
dataset/sim_collected/continuous/motion_library/manifest.json
```

The LAFAN source provides Base motions. `SKATE_EXPERT_DATASET` selects either
the formal Phase or Continuous Skate MotionLib; Phase is the default.
`SKATE_EXPERT_MOTION_FILE` remains a higher-priority explicit override. Both
formal datasets derive from the same canonical HUSKY robot-board raw
collection.

The short parallel collection test uses the checked-in configuration:

```bash
python train/scripts/data_collection/rollout_split.py \
  --parallel-config train/scripts/data_collection/rollout_config.json
```

The production configuration writes raw data to
`dataset/sim_collected/raw/`. The collector preserves each rollout,
organizes output as `round_NNN/rollout_NNN`, reports raw duration, and applies
the official HUSKY per-rollout randomization. Accepted expert duration is
computed only by the converter.

The shared M2.5c-P raw collection is available from the
[raw/](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/raw)
directory:

```bash
hf download Yak9Ce3teeh/skate-sim-dataset \
  --repo-type dataset \
  --include "raw/**" \
  --local-dir dataset/sim_collected
```

The validated M2.5c-C continuous dataset is available from the
[Skate-BFM Hugging Face dataset](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/continuous).
Restore only the Continuous files with:

```bash
hf download Yak9Ce3teeh/skate-sim-dataset \
  --repo-type dataset \
  --include "continuous/**" \
  --local-dir dataset/sim_collected
```

The formal M2.5c-P Phase dataset is stored beside raw under
[phase/](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/phase).
Restore the Phase files with:

```bash
hf download Yak9Ce3teeh/skate-sim-dataset \
  --repo-type dataset \
  --include "phase/**" \
  --local-dir dataset/sim_collected
```

## Formal Run

Phase:

```bash
CUDA_VISIBLE_DEVICES=0 \
SKATE_EXPERT_DATASET=phase \
SKATE_MAX_STEPS=100000 \
SKATE_WORK_DIR=$PWD/results/m2.6-phase-100k-seed4728 \
python train/scripts/train_skate_bfm.py
```

Continuous:

```bash
CUDA_VISIBLE_DEVICES=0 \
SKATE_EXPERT_DATASET=continuous \
SKATE_MAX_STEPS=100000 \
SKATE_WORK_DIR=$PWD/results/m2.6-continuous-100k-seed4728 \
python train/scripts/train_skate_bfm.py
```

The Phase 100k run has completed. The Continuous command is retained as the
next controlled experiment, but must not be launched until the frozen-policy
diagnostic is resolved.

The default contract uses 100,000 online transitions, 1,024 warmup
transitions, updates beginning at transition 1,500, 50 native updates every
500 transitions, and checkpoints at 20k, 50k, and 100k. Replay capacity
defaults to the transition budget and may be increased with
`SKATE_BUFFER_SIZE`, but it cannot be smaller than the budget. Every update
samples 64 complete Base sequences and 64 complete Skate sequences at sequence
length 8. The formal path uses four independent HUSKY online environments,
per-environment episode horizons and latent lifecycles, batched Actor
inference, and expert-conditioned reset states restored from the selected
dataset's canonical raw `qpos`/`qvel` frame. Online domain randomization remains
disabled.

The final pre-formal audit clears transient MuJoCo state before every expert
reset and validates each canonical raw source against its adjacent `nq`/`nv`,
dtype, joint-order, quaternion-order, and source-XML metadata before direct
`qpos`/`qvel` injection.

## M2.6 Phase Result

The formal Phase run produced `100000` replay transitions and `9900` native
updates. Numerical integrity passed, but every completed training episode
ended in fall: `3109` terminated episodes and `0` horizon truncations.

Frozen evaluation on the same 32 reset/latent seeds gave the following mean
survival times:

| Checkpoint | Mean survival | Fall rate |
| :--- | ---: | ---: |
| Official BFM0 | `1.264 s` | `32/32` |
| `checkpoint_20000` | `0.604 s` | `32/32` |
| `checkpoint_50000` | `0.519 s` | `32/32` |
| `checkpoint_100000` | `0.643 s` | `32/32` |

These are frozen-policy stability diagnostics, not skating-success metrics.
The trained checkpoints are currently less stable than the official BFM0, so
Continuous training is blocked pending reset/latent alignment and action/
observation diagnostics.

Run a visual evaluation by selecting any checkpoint:

```bash
STEP=100000
CUDA_VISIBLE_DEVICES=0 python train/scripts/evaluator.py \
  --checkpoint model/motion_library/2026-08-15_143013/checkpoint_${STEP} \
  --dataset phase \
  --episodes 4 \
  --horizon 1024 \
  --viewer
```

See [train_log.md](train_log.md) for the current engineering record and
[train_res.md](train_res.md) for the formal Phase result and evaluation.
