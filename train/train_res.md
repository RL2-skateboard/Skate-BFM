# Training Results

This file records retained results only. A checkbox means that the item was
verified; an unchecked item was not generated or is not yet a conclusion.

## A. Workspace Initialization

| Check | Result | Reference |
|---|---|---|
| BFM0/HUSKY MuJoCo stepping | [x] PASS | [`husky_sim/`](../husky_sim/) |
| State/action contract | [x] PASS | `state=64`, `privileged_state=463`, `action=29 -> HUSKY action=23` |
| Native B/F-only update boundary | [x] PASS | [`results/m2.2b-3/`](../results/m2.2b-3/) |
| Full task-performance validation | [ ] Not established | Frozen-policy diagnostics remain negative |

**Caption.** The table records interface validation, not skating success.

## B. Data Preparation and Collection

### Formal Phase Collection

| Quantity | Result |
|---|---:|
| Raw rollouts | 158 (`150` baseline + `8` replacements) |
| Raw frames | 452,885 |
| Raw duration | 150.962 min |
| Accepted Phase motions | 6,038 |
| Accepted expert frames | 452,291 |
| Accepted duration | 148.751 min |
| Conversion rejection | 0 |
| Official MotionLib / Seq8 validation | PASS |

**Caption.** Raw duration counts every original HUSKY episode; accepted
duration counts only phase segments retained by the converter.

Phase output and QC are published at
[Hugging Face / phase](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/phase).

### Continuous Dataset

| Quantity | Result |
|---|---:|
| Clip length | 500 frames |
| Clip duration | 10.0 s |
| Motion count | 890 |
| Fall/reset crossing | 0 |
| Publication | [Hugging Face / continuous](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/continuous) |

**Caption.** Continuous clips are fixed-window MotionLib records and may cross
normal phase transitions, but do not cross fall or reset boundaries.

### Data Artifacts

- [x] Raw collection:
  [Hugging Face / raw](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/raw)
- [x] Phase QC videos:
  [Hugging Face / phase/qc/videos](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/phase/qc/videos)
- [x] Continuous QC video:
  [Hugging Face / continuous/qc](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/continuous/qc)
- [ ] Held-out validation/test dataset: not collected.

## C. Training and Adaptation

### M2.2b Boundary Evaluation

| Updates | Base-only Top-1 | Base+Skate Top-1 | Base-only Top-5 | Base+Skate Top-5 |
|---:|---:|---:|---:|---:|
| 1 | 0.3750 | 0.4062 | 0.7031 | 0.6719 |
| 10 | 0.3125 | 0.3594 | 0.7500 | 0.7812 |
| 100 | 0.3281 | 0.3281 | 0.6875 | 0.8281 |

**Caption.** Fixed evaluator retrieval metrics; higher is better. This is an
FB representation boundary result, not a Skate task-success result. The
Base+Skate treatment was correctly configured only in the rerun recorded here.

### M2.5b Original BFM0 Skate Baseline

| Item | Result |
|---|---:|
| Online transitions | 20,000 |
| Native updates | 1,900 |
| Terminal falls | 389 |
| Horizon truncations | 19 |
| Fixed target-conditioned rollouts | 60 |
| Rollouts reaching confirmed fall before 128 steps | 60/60 |

**Caption.** The baseline completed numerically but did not demonstrate
stable target-conditioned Skate behavior.

### M2.6 Phase 100k

| Checkpoint | Mean frozen survival | Frozen falls |
|---|---:|---:|
| Official BFM0 | 1.264 s | 32/32 |
| 20k | 0.604 s | 32/32 |
| 50k | 0.519 s | 32/32 |
| 100k | 0.643 s | 32/32 |

**Caption.** Same 32 reset/latent seeds and 1024-step horizon; lower survival
and 32/32 falls indicate that training did not improve behavioral stability.

### M2.6-P0 Frozen Matched Closed-Loop Preflight

| Metric | `FORMAL_D2.2` | `PURE_RANDOM_Z` |
|---|---:|---:|
| Matched resets | 512 | 512 |
| Mean survival steps / 51 | 49.816 | 49.705 |
| Failure by 20 steps | 0.00% | 0.00% |
| Failure by 50 steps | 26.17% | 26.76% |
| Root tilt p95 | 66.71 deg | 66.93 deg |
| Waist `|a| >= 0.95` | 82.33% | 82.69% |

**Caption.** `FORMAL_D2.2` uses aligned tracking-z for expert slots and
background z for free slots; `PURE_RANDOM_Z` uses only background random z.
The small difference does not establish a meaningful tracking-z improvement.

Preflight structural result:

- [x] Exact qpos/qvel reset and source-physics restoration.
- [x] History, action, wrist-zero, finite-value, and latent lifecycle checks.
- [x] Model, normalizer, and buffer hashes unchanged.
- [x] No optimizer, backward, `agent.update`, replay update, or training.
- [ ] Behavioral readiness for short retraining: **not established**.

Raw JSON result: `/tmp/m26_p0_frozen_closed_loop.json` (local diagnostic
artifact; intentionally not committed).

### M2.6 Alignment Diagnostics

| Audit | Result |
|---|---|
| Source physics alignment | [x] Exact source realization restored |
| 29D BFM to 23D HUSKY action subspace | [x] Shared joints exact; six wrists zero |
| Observation distribution | [ ] Blocked by upstream/current `base_ang_vel` scale asymmetry |
| Expert action bridge | [x] Exact affine bridge plus explicit `PROJECTED` fallback |
| Temporal context | [ ] Moderate context loss, especially `steer2push` |

**Caption.** These are compatibility diagnostics used to choose the next
training experiment; they are not model-performance curves.

## Curves, Visualizations, and Videos

The current training branch did not retain a committed training-loss curve or
training rollout video. The available numerical artifacts are linked above
and the frozen-policy table is the authoritative behavioral comparison.

- [ ] Training loss curve: not generated for the formal Phase 100k run.
- [ ] Parameter/gradient norm curve: not recorded by the current trainer.
- [ ] Committed training rollout video: not retained.
- [x] HUSKY Phase QC videos:
  [Hugging Face / phase/qc/videos](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/phase/qc/videos)

**Caption.** Missing curves and videos are recorded as missing rather than
reconstructed from unrelated smoke outputs.

## Result Interpretation

- [x] Numerical training/checkpoint integrity passed.
- [x] Formal raw and MotionLib data validation passed.
- [ ] Frozen policy behavioral improvement passed.
- [ ] Skate motion-library quality passed.
- [ ] Held-out validation/test generalization passed.

The current evidence supports further controlled diagnosis of the action,
observation, and expert-context contracts. It does not support claiming that
the trained checkpoint is a usable Skate-BFM motion library.
