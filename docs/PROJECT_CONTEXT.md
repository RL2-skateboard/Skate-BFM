# Skate-BFM Project Context

> Maintained for future coding and research sessions. This document is a
> compact, evidence-based map of the repository, runtime contracts, experiments,
> and current blockers. It is not a replacement for `train/train.md` or
> `train/train_res.md`.

## Snapshot

- Snapshot date: 2026-09-25.
- Repository: `RL2-skateboard/Skate-BFM`.
- Remote heads currently visible: `main` and `train` only.
- `main`: `69706db` (`整理 H1 实验文档`).
- `train`: `b3140e6` (`add latent inset to evaluator videos`).
- Current checkout: `train`, tracking `origin/train`.
- `main` is not an ancestor of `train`; the branches diverged during the H1
  documentation phase. Treat `train` as the research/training line, not as a
  small patch on top of the current `main` snapshot.
- HUSKY submodule: `TeleHuman/humanoid_skateboarding` at
  `d93833e80deff7f927c0b80ef9c435d8b5c488fe`.
- Current worktree is dirty: `train/train.md`, `train/train_log.md`, and
  `train/train_res.md` are modified; `logs/` contains four untracked local
  training logs. Preserve these changes unless explicitly asked to clean them.

## Project Goal

Skate-BFM adapts the official BFM-Zero motion prior to a HUSKY humanoid
skateboard MuJoCo environment. The project connects a 29-DoF BFM policy and
latent-conditioned FB-CPR-Aux training loop to a 23-actuator HUSKY robot plus
board dynamics.

The research question is not merely whether the code runs. The unresolved
question is whether the adapted policy can reproduce steady skateboard skills
and, especially, the `push2steer` and `steer2push` transitions under held-out
raw physics and exact expert-reset provenance.

## Repository Map

### Project-owned runtime

- `src/skate_bfm/integration/actions.py`: BFM29-to-HUSKY23 mapping, wrist
  projection, source-action physical-target bridge, official hard-waist gains.
- `src/skate_bfm/integration/observations.py`: 64D state, 463D privileged state,
  29D previous action, and 372D four-frame actor history.
- `src/skate_bfm/integration/online.py`: one online transition boundary,
  MuJoCo stepping, replay serialization, latent/action/termination metadata.
- `husky_sim/src/skate_husky/lite_env.py`: headless MuJoCo runtime, actuator
  validation, canonical source-physics application, auxiliary penalties, and
  persistent fall detector.
- `tests/`: integration and observation-contract tests. The tests cover joint
  mapping, hard-waist gains, wrist projection, reset provenance, source physics,
  fall confirmation, latent lifecycle, and replay schema.

### Training and evaluation

- `train/scripts/train_skate_bfm.py`: formal training entrypoint and
  project-owned orchestration around the vendored BFM agent.
- `train/scripts/train_runner.py`: strict checkpoint loading, raw provenance,
  source-physics validation, hashes, optimizer checks, and finite-state checks.
- `train/scripts/evaluator.py`: frozen Phase/Raw evaluation, exact case replay,
  task metrics, videos, latent analysis, and Val/Test case selection.
- `train/scripts/data_collection/rollout_split.py`: canonical HUSKY rollout
  collection, phase clock, command grid, physics randomization, and raw metadata.
- `train/scripts/data_collection/convert_phase.py`: phase-pure MotionLib build,
  provenance audit, official MotionLib validation, and QC videos.
- `train/scripts/data_collection/convert_continuous.py`: non-overlapping
  500-frame Continuous MotionLib build and QC.
- `train/scripts/isaac_env/`: vendored BFM-Zero/HumanoidVerse runtime. Do not
  replace its algorithms with a second project-owned implementation without a
  specific research decision.

### Data and artifacts

- `train/dataset/`: authoritative local layout for Base, Train, Val, and Test;
  binary data is normally restored from the Hugging Face dataset.
- `model/bfm-zero-official/`: official BFM0 initialization only.
- `model/motion_library/<model_name>/`: Skate-BFM checkpoints and replay buffers.
- `train/eval_res/`: tracked evaluation summaries, JSON, plots, and video
  metadata. MP4 and large binary artifacts may be ignored or externally hosted.
- `results/`: local training summaries and diagnostics, generally not tracked.
- `logs/`: local raw training logs, currently untracked.

## Runtime Contracts

### Action contract

BFM emits `a_bfm in R^29`. HUSKY physically executes 23 shared joint actions.
The six unavailable wrist DoFs are always zero:

```text
left/right wrist roll, pitch, yaw -> zero
remaining 23 joints -> copied by joint name
```

Projection is installed on every native BFM action consumer, including forward,
target-forward, critic, target-critic, auxiliary critic, and target auxiliary
critic. This prevents nonphysical wrist coordinates from influencing either
training or execution.

The online hard-waist target contract is:

```text
G_bfm[j] = 5 * 0.25 * effort_hard_waist[j] / Kp_hard_waist[j]
q_target_bfm[j] = q0_hard_waist[j] + G_bfm[j] * a_bfm[j]
```

Source HUSKY actions use:

```text
q_target_src[j] = q0_src[j] + s_src[j] * a_src[j]
a_bfm_eq[j] = (q_target_src[j] - q0_hard_waist[j]) / G_bfm[j]
```

Out-of-range `a_bfm_eq` components are explicitly classified as `PROJECTED`
and clipped; they must not be silently described as exact matches.

### Observation contract

- `state`: 64D = 29 position offsets + 29 joint velocities + 3 projected
  gravity + 3 scaled angular velocity.
- `privileged_state`: 463D official BFM max-local-self observation.
- `last_action`: 29D BFM action, scaled by BFM action rescale in the online
  observation path.
- `history_actor`: 372D = four frames of 29D actions, 3D angular velocity,
  29D positions, 29D velocities, and 3D gravity.
- `z`: 256D latent with norm approximately 16 after BFM projection.

The privileged state is built from HUSKY body positions, quaternions, linear
and angular velocities in the root heading frame. A synthetic head body is
appended using the torso pose and a 0.35m local offset to match BFM's 31-body
schema.

### Online physics and reset

- Formal training uses four independent `HuskyBfmOnlineEnv` instances.
- Control step is 0.02s: MuJoCo timestep 0.002s with decimation 10.
- Training does not add new online domain randomization. It restores the exact
  source rollout physics realization before injecting raw source `qpos/qvel`.
- Source physics includes COM offsets, robot/deck/foot/wheel friction, and
  initial joint offsets. Reapplication restores nominal model fields first to
  avoid cumulative drift.
- Fall detection requires persistent bad frames, with default tilt threshold
  70 degrees, root height threshold 0.45m, and 0.2s confirmation time.
- Auxiliary penalties are finite nonnegative raw values for torque square,
  action rate, DoF position limits, torque limits, undesired contact, foot
  orientation, ankle roll, and surface-relative slippage.

## Training Loop

The trainer starts from a fresh official BFM0 checkpoint, builds official Base
MotionLib and Skate MotionLib expert buffers, then combines:

- 64 Base sequences of length 8;
- 64 Skate sequences of length 8;
- 1,024 expert rows per update preflight;
- 50/50 Base/Skate sequence mixture;
- CPU `DictBuffer` replay for online HUSKY transitions;
- four online environments;
- stochastic pretrained-Actor warmup of 1,024 transitions;
- first update at transition 1,500;
- update every 500 transitions;
- 50 native `FBcprAuxAgent.update()` calls per update block;
- 200k runs: 398 blocks and 19,900 native updates.

Each native update consumes online replay and expert sequences, updates the
discriminator, FB forward/backward maps, main critic, auxiliary critic, and
Actor, then soft-updates target networks. The auxiliary reward is a weighted
sum of eight HUSKY penalties; it is not a forward-speed or board-displacement
reward.

Latent lifecycle:

- expert-role rollout environments receive reset-aligned tracking latents
  computed from future expert observations;
- free/background latents refresh every 100 environment steps from the z-buffer
  when available, otherwise from the model prior sampler;
- tracking is capped by the configured maximum tracking length and can fall back
  to the background latent after the aligned window ends.

## Dataset Pipeline

Canonical raw HUSKY rollouts are recorded at 50Hz from the official HUSKY test
scene and source policy. The command grid is 5 velocities `{0.50, 0.75, 1.00,
1.25, 1.50}` times 15 headings from `-0.7` to `+0.7`, for 75 cells. The formal
collection target was 150 raw minutes using 150 baseline episodes plus possible
replacement capacity.

Train artifacts currently document:

- 158 accepted raw rollouts;
- 452,885 raw frames / 150.962 raw minutes;
- Phase: 6,038 motions / 452,291 frames / 148.751 minutes;
- Continuous: 890 clips / 445,000 frames / 148.333 minutes;
- Continuous clips are 500 frames / 10 seconds with zero overlap;
- Phase and Continuous official MotionLib validation passed;
- raw provenance and QC passed;
- Val and Test are source-disjoint evaluator-only splits.

Phase segments are contiguous and phase-pure, excluding fall tails, failure
margin, reset cuts, and segments shorter than Seq8 requirements. Continuous
windows preserve normal phase transitions but cannot cross a fall or reset.
Both converters preserve raw source frame ranges, command values, physics seed,
board state, action, phase annotations, and joint mapping provenance.

## Evaluation Protocols

There are two important fixed benchmarks:

1. Historical fixed Test comparison: 80 cases, 20 each for push, steer,
   push2steer, and steer2push; rollout-balanced without replacement; seed 4728;
   identical case identity for Phase 20k/50k/100k and Continuous comparisons.
2. Current Phase-Val model-selection bank: 80 cases with the same 20-per-
   behavior structure and seed 4728; used for Phase 120k/150k/200k and current
   task-oriented metrics. It is Val, not a new Test result.

Frozen evaluation must use deterministic Actor mean action, exact Raw reset
provenance, checkpoint-specific tracking latent reconstruction, no training,
no backward pass, no optimizer update, and no normalizer mutation. Binary task
success remains `NOT_YET_CALIBRATED`; proposed thresholds are model-selection
candidates, not frozen success criteria.

## Current Results

### Phase 200k

The local Phase 200k run is a fresh restart from official BFM0 with seed 4728,
6,038 Phase motions, 200,000 transitions, 398 update blocks, and 19,900 native
updates. Checkpoints at 20k, 50k, 100k, 120k, 150k, and 200k reload with finite
model/optimizer/normalizer state. Final model SHA256:

```text
ec0f4c04195024e5c47a3d8003e89de7b7cf71a9d82bdd86e96a4a39c6c4627b
```

Training diagnostics: 3,831 episodes, 3,807 falls, 24 horizon truncations,
43,158 tracking-latent transitions, realized tracking ratio 0.21579, maximum
tracking window 133 steps, and full z-buffer occupancy 8,192.

Current fixed Phase-Val result:

| Behavior | Full completion | Mean completion ratio | Board XY error | Coupling XY error |
|---|---:|---:|---:|---:|
| push | 0.650 | 0.8405 | 0.0990m | 0.0634m |
| steer | 1.000 | 1.0000 | 0.0809m | 0.0379m |
| push2steer | 0.500 | 0.7686 | 0.8830m | 0.6640m |
| steer2push | 0.250 | 0.7628 | 0.6307m | 0.2907m |

Interpretation: steady steer is currently the strongest validated behavior.
Transition reproduction remains incomplete. The 200k latent analysis used
11,338 actual tracking points plus 4,096 deterministic prior points; shared PCA
ratios are 0.0720, 0.0490, 0.0477; nearest-centroid accuracy is 0.5062 and
cosine silhouette is 0.0155. These are latent-access diagnostics, not skill
success evidence.

### Continuous 200k

The local Continuous 200k run also completed 200,000 transitions, 398 blocks,
and 19,900 native updates from fresh official BFM0. Final model SHA256:

```text
a90776e247c1e5e466e09e412dd01829530712b0b645d1eb56ebc71efc2cfac6
```

Its summary reports 3,583 episodes, 3,567 falls, 16 truncations, and a
realized tracking ratio of 0.41511. The run has finite reloadable checkpoints,
but no formal 200k performance evaluation is currently recorded. Do not infer
Continuous 200k behavior from the training summary alone.

### Phase vs Continuous at 100k

The tracked paired Val comparison shows Continuous improves several steady
board/coupling means, but completion is lower and both transition categories
still terminate in all sampled cases. The difference cannot be attributed only
to cross-phase history because Phase and Continuous also alter MotionLib and
reset sampling distributions.

## Experiment Status

- Experiment 1, integration: structural and numerical contracts pass; stable
  Skate behavior is not implied.
- Experiment 2, data: raw/Phase/Continuous collection, provenance, MotionLib
  validation, and QC pass; equal motion quality across all commands/phases is
  not established.
- Experiment 3, Phase BFM training: 20k/50k/100k Test comparison and 200k Val
  extension complete; steady steer improved, transitions remain unresolved.
- Experiment 4, Continuous BFM training: fresh 10k health run, formal 100k,
  fixed comparisons, and Phase-vs-Continuous Val comparison complete; no robust
  transition advantage is established. Local Continuous 200k training exists,
  but its behavior evaluation is pending.
- Experiment 5, BFB/RFB dynamics-conditioned training: planning only. No BFB/RFB
  implementation, training result, or held-out behavioral result exists.

## Known Risks and Open Questions

- Top-level `README.md` still describes the older Phase 100k status. Use
  `train/train.md`, `train/train_res.md`, and this file for the current train
  branch state.
- `train/train.md`, `train/train_res.md`, and `train/train_log.md` contain local
  uncommitted 200k updates. Do not overwrite them during cleanup.
- Local 200k model/result directories and `logs/` are not necessarily tracked
  by Git. Reproducibility requires restoring the external binary artifacts or
  rerunning training.
- The formal code still contains Hydra warnings about missing `version_base` and
  `_self_`; these did not prevent the recorded runs but should be cleaned in a
  separate maintenance change if desired.
- Observation-scale mismatch was audited: expert root angular velocity uses
  scale 1.0 while online uses 0.25, with severe waist distribution drift in the
  recorded audit. It is documented but not fully resolved by a new behavioral
  training result.
- Structural hard-waist and action-bridge checks pass, but exact target mapping
  and one-step parity do not prove closed-loop dynamics equivalence.
- Do not call a checkpoint a usable Skate motion library until held-out steady
  and transition behavior is robust and binary task criteria are calibrated.

## Safe Next Actions

1. Preserve or commit the current local documentation changes before branching
   into new experiments.
2. Evaluate Continuous 200k on the same fixed Phase-Val bank using the existing
   evaluator and exact case identity rules.
3. Compare Phase 200k and Continuous 200k on identical Val cases, reporting both
   completion and physical board/coupling metrics.
4. Resolve or explicitly control the observation-scale mismatch before another
   expensive training run.
5. Only then implement Experiment 5 BFB/RFB, with fixed-batch parity tests before
   any long training run.

## Primary References

- `README.md`: project entrypoint and reproduction setup; currently older than
  the local 200k train records.
- `train/README.md`: training commands and code layout.
- `train/train.md`: methods, semantics audits, experiment design, and status.
- `train/train_res.md`: numerical results, artifacts, and verified conclusions.
- `train/train_log.md`: date-based work log.
- `DOCUMENTATION_RULES.md`: required evidence/verification discipline.
- `src/skate_bfm/integration/`: project-owned BFM/HUSKY boundary.
- `husky_sim/src/skate_husky/lite_env.py`: MuJoCo physics/controller boundary.
- `train/scripts/train_skate_bfm.py`: formal trainer.
- `train/scripts/evaluator.py`: frozen evaluator.
- `train/eval_res/2026-08-24/p2-200k-v20b-latent_val_phase_eval/`: current Phase
  200k Val summary, latent metrics, and video metadata.
- `results/m2.6-p2-phase-200k-seed4728/summary.json`: local Phase 200k run
  provenance and integrity summary.
- `results/m2.6-c2-continuous-200k-seed4728/summary.json`: local Continuous
  200k run provenance and integrity summary.

## Context Refresh Checklist

Before making a new change, run:

```bash
git status --short --branch
pytest -q
python -m compileall -q src husky_sim/src train/scripts
```

Then inspect the relevant local block in this file, the authoritative train
document, and the exact checkpoint/result artifacts for the experiment being
changed. Never infer current progress from the top-level README alone.
