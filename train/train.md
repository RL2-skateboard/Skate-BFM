# Skate-BFM Training Plan

## Scope

The training branch develops a Skate-BFM motion library from the official
BFM-Zero model and HUSKY skateboard expert motion. The current boundary is
BFM0 + Skate expert integration; it is not yet a validated skateboarding
controller.

## Stage A: Workspace Initialization

**Goal:** connect the official BFM0 interfaces to the HUSKY MuJoCo runtime
without changing the upstream model semantics.

**Method**

- BFM0 keeps `z in R^256`, `state in R^64`, `privileged_state in R^463`,
  and normalized action `a_bfm in R^29`.
- HUSKY executes the name-mapped physical subspace `a_husky in R^23`;
  the six absent wrist coordinates are zero.
- Online history is initialized as `history_actor = 0` and `last_action = 0`,
  then contains only actions produced by the current online rollout.
- Official BFM-Zero code is used through the project-owned copy under
  [`scripts/isaac_env/`](scripts/isaac_env/). The original source is not a
  runtime dependency of the formal entrypoint.

**Evidence**

- [x] HUSKY viewer and headless MuJoCo stepping work.
- [x] BFM0 forward/backward interfaces load with the official checkpoint.
- [x] The 29D-to-23D action contract is finite and name-aligned.
- [x] A native BFM-Zero update path was exercised without changing the
  vendored algorithm.
- [ ] Full training behavior is validated.

**Known issue and response**

The trained policy falls early. We froze training and ran source-physics,
reset, action, observation, latent, and frozen closed-loop audits before
starting another training run.

## Stage B: Data Preparation and Collection

**Goal:** produce phase-structured HUSKY expert MotionLib data while retaining
raw robot, board, action, phase, and source provenance.

**Method**

- Raw collection records synchronized robot and skateboard state at 50 Hz.
- `phase_id` is used for contiguous segmentation; confirmed falls and resets
  are hard boundaries.
- HUSKY 23DoF poses are mapped to BFM 29DoF by joint name, with absent wrists
  fixed to zero.
- Each accepted motion must satisfy the official MotionLib and Seq8 loader
  schema. No interpolation or synthetic expert action is added.

**Evidence**

- [x] Formal raw collection: `452,885` frames, `150.962` minutes.
- [x] Phase MotionLib: `6,038` motions, `452,291` accepted frames,
  `148.751` minutes.
- [x] Continuous MotionLib: `890` fixed 500-frame clips.
- [x] Raw, phase, and continuous artifacts are published on
  [Hugging Face](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset).
- [ ] A separate validation split has not been collected; current training
  experiments use the selected Hugging Face MotionLib for expert sampling.

## Stage C: BFM0 + Skate Expert Training

**Goal:** adapt the BFM0 representation and policy to Skate expert dynamics
while preserving the official training algorithm as far as the current
integration contract allows.

**Method**

The expert mixture samples complete sequences, not individual frames:

```text
batch = 0.5 * Base(LAFAN1) + 0.5 * Skate(Phase or Continuous)
sequence length = 8
batch size = 1024 = 64 Base sequences + 64 Skate sequences
```

The online transition is:

```text
z_t -> BFM Actor -> a_bfm(29)
    -> project_husky_bfm_action
    -> name map -> a_husky(23)
    -> HUSKY PD controller -> next robot/board state
```

The formal 100k schedule uses 1,024 warmup transitions, starts updates at
1,500 transitions, and performs 50 native updates every 500 transitions.
No command or board state is appended to the Actor observation.

For source-action diagnostics only, the affine bridge is:

```text
a_bfm_eq[j] = (q0_src[j] + s_src[j] * a_src[j] - q0_bfm[j])
               / (5 * s_bfm[j])
```

Representable components use this exact inverse. Out-of-range components are
marked `PROJECTED` and clipped; this bridge is not injected into formal Actor
rollouts.

**Current issues and responses**

| Issue | Current response |
|---|---|
| Source physics and canonical reset mismatch risk | Restore the recorded physics realization and exact raw robot/board `qpos/qvel`; fail closed on metadata mismatch. |
| HUSKY has 23 physical DoFs while BFM0 has 29 | Keep BFM0 width 29 and execute only the name-aligned 23D subspace; zero six wrists. |
| Expert and current control targets differ at hip pitch | Keep exact affine translation plus explicit `PROJECTED` diagnostics; do not silently reinterpret raw actions. |
| Expert temporal context is incomplete at some phase boundaries | Quantify current/history/strict-5-context coverage; do not pad or invent action semantics without a later decision. |
| Fresh BFM0 closed loop shows waist saturation and late falls | Run frozen matched reset diagnostics before short retraining. |

**Verification status**

- [x] Structural reset/history/action/z contract.
- [x] Official checkpoint and normalizer unchanged during frozen preflight.
- [x] No early (`<=20` step) systematic fall in the 512-reset preflight.
- [x] Formal Phase 100k training and checkpoint reload.
- [ ] Stable Skate behavior after training.
- [ ] Native termination and auxiliary reward are sufficient for task success.
- [ ] A held-out validation/test split and final motion-library benchmark.

## Current Decision

The implementation is structurally ready for controlled diagnosis, but the
behavioral result is not yet a successful motion-library training result.
The next experiment must address the observed action/observation/control
alignment issues and report frozen-policy metrics before claiming improvement.

## Source Index

- Training entry: [`scripts/train_skate_bfm.py`](scripts/train_skate_bfm.py)
- Frozen evaluator: [`scripts/evaluator.py`](scripts/evaluator.py)
- Collection and conversion:
  [`scripts/data_collection/`](scripts/data_collection/)
- BFM/HUSKY adapters:
  [`../src/skate_bfm/integration/`](../src/skate_bfm/integration/)
- Formal dataset source:
  [Skate-BFM Hugging Face dataset](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset)
