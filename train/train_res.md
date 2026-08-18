# Skate-BFM Training Results

Results use the same three experiment names as [`train.md`](train.md).
`[x]` marks completed verification; `[ ]` marks an unverified conclusion.

## Experiment 1: Training Workspace and BFM-HUSKY Integration

### Stage Results

| Check | Result |
|---|---|
| Independent `train` branch | PASS |
| Vendored BFM runtime construction | PASS |
| Official checkpoint strict load | 537 tensors, PASS |
| HUSKY headless step | PASS |
| Interactive MuJoCo viewer | PASS |
| BFM action to HUSKY action | `[29] -> [23]`, PASS |
| State / privileged / last action / history | `[64] / [463] / [29] / [372]`, PASS |
| Latent | `[256]`, finite, norm 16 |

**Caption.** Bracketed values are tensor widths, not physical magnitudes:
state/privileged state/last action/history contain 64/463/29/372 values, and
the latent has 256 values. `DoF` means degree of freedom; the norm-16 latent
uses the BFM convention for a 256-dimensional vector. `Strict load` means all
537 checkpoint tensors matched by name and shape; `PASS` records a completed
interface check, not policy quality.

One initial 50-frame HUSKY sample loaded beside the original 862-motion LAFAN
library. The official expert loader produced finite Base and Skate batches
with `state [16,64]`, `last_action [16,29]`, and
`privileged_state [16,463]`.

### Verified and Unverified Conclusions

- [x] BFM0 and HUSKY communicate through one reproducible tensor and joint-name
  contract.
- [x] The training workspace no longer needs the original BFM source directory
  at runtime.
- [ ] Stable Skate behavior follows from interface compatibility alone.

**Caption.** These are software/interface results, not model-performance
results.

## Experiment 2: HUSKY Expert Dataset Collection and Construction

### Stage Results

**Raw collection**

| Quantity | Result |
|---|---:|
| Command cells | 75 |
| Baseline episodes | 150 |
| Replacement episodes | 8 |
| Worker failures | 0 |
| Full 60 s episodes | 125 |
| Fall-terminated episodes | 33 |
| Frames | 452,885 |
| Duration | 150.962 min |
| Duration target | PASS |

**Caption.** `Command cells` are distinct `(velocity, heading)` settings;
`baseline episodes` are the planned two samples per cell and `replacement
episodes` extend collection after short fall-terminated rollouts. `Full` and
`fall-terminated` partition the 158 completed episodes; `worker failures` are
process-level collection errors. A frame is one 50 Hz sample, and duration is
accumulated simulation minutes. A confirmed fall is a valid episode boundary,
not a failed worker.

**Phase dataset**

| Phase | Motions | Frames | Duration |
|---|---:|---:|---:|
| push | 1,522 | 182,296 | 60.258 min |
| push2steer | 1,516 | 45,471 | 14.652 min |
| steer_left | 685 | 91,205 | 30.173 min |
| steer_forward | 109 | 14,670 | 4.854 min |
| steer_right | 717 | 96,314 | 31.866 min |
| steer2push | 1,489 | 22,335 | 6.949 min |
| **Total** | **6,038** | **452,291** | **148.751 min** |

**Caption.** `Motions` counts contiguous MotionLib records, `frames` counts
their source time samples, and duration is `(frames-1)/50` seconds per motion
summed over records. The phase labels are command-clock categories; counts
are descriptive, not target proportions.

Discarded data:

| Reason | Frames / segments |
|---|---:|
| Confirmed fall | 330 frames |
| Pre-fall margin | 240 frames |
| Too short for Seq8 | 6 segments |
| Complete rollout rejection | 0 |

**Caption.** A confirmed fall and its pre-fall margin are removed from training
segments. `Seq8` needs eight transitions plus one next-state frame; fewer
frames cannot produce a complete sample. Fewer discarded frames/segments is
preferable, provided unsafe fall tails are excluded.

**Continuous dataset**

| Quantity | Result |
|---|---:|
| Clips | 890 |
| Clip size | 500 frames / 10.0 s |
| Stride / overlap | 500 / 0 |
| Frames | 445,000 |
| Duration | 148.333 min |
| Tail discarded | 7,310 frames |
| Clips crossing normal phase transitions | 890/890 |
| Clips crossing fall/reset | 0 |

**Caption.** `Clips` are continuous records, each `500 frames = 10.0 s` at
50 Hz. `Stride` is the frame offset between clip starts and `overlap` is the
number of shared frames. A tail is the valid remainder shorter than one clip;
normal phase transitions are allowed, but fall/reset boundaries are not.

### Verified and Unverified Conclusions

- [x] Official MotionLib loading and Seq8 expert sampling pass for Phase and
  Continuous.
- [x] Every processed record retains raw frame provenance.
- [x] Phase QC contains 60 samples; Continuous QC contains 10 samples.
- [x] [Raw collection](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/raw)
- [x] [Phase MotionLib and QC](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/phase)
- [x] [Continuous MotionLib and QC](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/continuous)
- [ ] Held-out validation/test source collection.
- [ ] Equal expert quality across every phase and command cell.

**Caption.** MotionLib and provenance validation establish data correctness,
not learned motion quality.

The linked Phase QC videos visualize robot-plus-skateboard replays for 10
samples per phase; Continuous QC visualizes 10 fixed 500-frame clips. These
are dataset-quality checks, not trained-policy performance videos.

## Experiment 3: BFM + Skate Expert Training and Semantics Alignment

### Training Process Results

Fixed 1,024-transition replay:

| Check | Result |
|---|---:|
| Normal / terminal / truncated transitions | 1,009 / 14 / 1 |
| Auxiliary reward keys | 8/8 finite |
| Base/Skate expert sequences | 64 / 64, length 8 |
| Native update tests | 1, 10, 100 |
| Six optimizer steps after final test | 100 each |
| z-buffer | 8,192 / 8,192 |
| Model and normalizers finite | PASS |

**Caption.** A transition is one `(state, action, next state)` record.
`Normal`, `terminal`, and `truncated` counts distinguish continuing samples,
failure-ended episodes, and time-limit-ended episodes. `Seq8` is eight
consecutive transitions. `Auxiliary reward keys` counts the eight configured
penalty terms; `native update tests` are cumulative update depths; `six
optimizer steps` checks each network optimizer reached step 100. The z-buffer
stores 8,192 latent vectors; `finite` means no NaN or Inf was found.

Representative fixed-replay optimization:

| Metric | Update 1 | Update 10 | Update 100 |
|---|---:|---:|---:|
| FB loss | 1,283,845 | 1,053,397 | 286,643 |
| Discriminator loss | 0.4494 | 0.0580 | 0.0125 |
| Main critic loss | 1,796.0 | 1,977.3 | 39.6 |
| Auxiliary critic loss | 248.9 | 171.0 | 53.5 |
| Actor loss | 50,068 | 34,344 | 22,388 |

**Caption.** `FB` is the forward-backward representation objective;
discriminator loss separates expert from online transitions, critic losses fit
value estimates, and Actor loss updates the policy. These are scalar
optimization objectives with implementation-specific scales; finite values
and downward trends are diagnostics, not proof of policy improvement.

**20k closed-loop bring-up**

| Quantity | Result |
|---|---:|
| Online transitions | 20,000 |
| Update blocks / native updates | 38 / 1,900 |
| Normal / terminal / truncated | 19,592 / 389 / 19 |
| Checkpoints | 10k, 20k |
| Checkpoint reload | PASS |
| Fixed-evaluation falls | 60/60 |

**Caption.** `Online transitions` are environment steps, `update blocks` are
scheduled groups of updates, and `native updates` are optimizer update calls.
`Normal/terminal/truncated` respectively count continuing, fall-ended, and
time-limit-ended transitions. `10k/20k checkpoints` are snapshots saved at
those transition counts. Reload verifies serialized state; fixed-evaluation
falls are terminated test episodes, where fewer is better.

| Metric | Start | 10k | 20k |
|---|---:|---:|---:|
| FB loss | 986,467 | 23,839 | 11,141 |
| Discriminator loss | 0.4382 | 0.3296 | 0.2129 |
| Main critic loss | 1,129.67 | 37.51 | 45.73 |
| Auxiliary critic loss | 166.75 | 8.68 | 36.62 |
| Actor loss | 48,756.79 | 21,856.67 | 6,030.91 |

**Caption.** `10k` and `20k` denote online transition counts. Loss values are
training diagnostics rather than rewards; lower is often desirable for the
same objective, but matched frozen behavior is the deciding criterion.

**Conclusion.** Optimization and checkpointing worked, but every fixed
evaluation episode fell; physical performance was inconclusive.

**Formal Phase 100k run**

| Quantity | Result |
|---|---:|
| Online transitions | 100,000 |
| Update blocks / native updates | 198 / 9,900 |
| Normal transitions | 96,891 |
| Fall terminations | 3,109 |
| Horizon completions | 0 |
| Expert resets | 3,113 |
| Model/optimizer/normalizer finite | PASS |

**Caption.** `Normal transitions` neither terminate nor truncate; `fall
terminations` end an episode by the persistent fall detector; `horizon
completions` are time-limit truncations; `expert resets` reset to selected
expert source frames. For stability, fewer falls and more horizon completions
are preferable.

Checkpoint integrity:

| Checkpoint | Model SHA256 prefix | Optimizer step | Reload |
|---|---|---:|---|
| 20k | `6c76f0a6ba20` | 1,900 | PASS |
| 50k | `4e7eda31be77` | 4,900 | PASS |
| 100k | `04c51a9bc938` | 9,900 | PASS |

**Caption.** `SHA256 prefix` is a shortened checkpoint content hash,
`optimizer step` is the number of saved update calls, and `reload` verifies
that the checkpoint can reconstruct the model state. These establish artifact
integrity, not behavior.

Optimization endpoints:

| Step | FB loss | Disc. loss | Critic loss | Aux critic loss | Actor loss |
|---:|---:|---:|---:|---:|---:|
| 1,500 | 325,215.56 | 0.02593 | 856.73 | 80.34 | 18,889.73 |
| 20,000 | 6,070.78 | 0.20310 | 52.17 | 45.79 | 8,131.50 |
| 50,000 | -3,871.08 | 0.15599 | 12.75 | 3.10 | 954.92 |
| 100,000 | -10,048.30 | 0.08824 | 5.08 | 2.30 | 1,116.23 |

**Caption.** `FB loss` may be negative because its objective contains a
negative diagonal term. `Disc.` is discriminator loss; `Critic` and `Aux
critic` are main and auxiliary value losses; `Actor` is policy loss. Lower is
not by itself evidence of better control.

Matched frozen evaluation:

| Checkpoint | Mean survival | Min / max | Falls |
|---|---:|---:|---:|
| Official BFM0 | 1.264 s | 0.84 / 2.48 s | 32/32 |
| Phase 20k | 0.604 s | 0.42 / 0.84 s | 32/32 |
| Phase 50k | 0.519 s | 0.36 / 0.66 s | 32/32 |
| Phase 100k | 0.643 s | 0.38 / 1.16 s | 32/32 |

**Caption.** Survival is elapsed simulation time before termination; higher
mean and minimum are better. `Min / max` are the observed episode range, and
`Falls` is terminated episodes over the fixed 32-episode evaluation set;
lower is better.

**Conclusion.** Numerical execution passed, but every trained checkpoint was
less stable than official BFM0. Behavioral adaptation failed.

### Failure Diagnosis and Semantics Correction Results

These rows are diagnostic steps within Experiment 3:

| Process step | Measured result | Current conclusion |
|---|---|---|
| Source physics + exact reset | max qpos/qvel error `0`; physics mismatch `0` | reset contract corrected |
| Active action subspace | shared-action error `0`; six wrists `0` | 23DoF projection correct |
| Observation distribution | waist reaches about 20-27 sigma; expert/online angular velocity 1.0/0.25 | observation asymmetry unresolved |
| Temporal action context | Phase strict-5 coverage 64.8366%; Continuous 85.8434%; steer2push 12.1379% | Phase context loss is material |
| Source-to-BFM action bridge | physical target RMSE `1.97e-17`; exact components 99.600839%; exact frames 92.1503% | exact where representable; hip tail projected |
| Diagnostic hip-tail refinement | next-state RMSE 0.4689 -> 0.4482; held-out ratio 0.9610 | improvement too weak/inconsistent; not adopted |
| Same-reset tracking z on old 100k | step-20 root tilt 47.8 -> 25.2 deg; saturation 17.1% -> 0.29% | short old-checkpoint improvement only |

**Caption.** `qpos/qvel` are MuJoCo generalized position/velocity; their reset
errors are expected to be zero. `sigma` is standard deviation from the
reference observation distribution, so 20-27 sigma indicates severe drift.
`RMSE` is root-mean-square error in radians, where lower is better. Root tilt
is torso inclination in degrees; action saturation is the fraction at the
normalized action limit, where lower usually leaves more control margin.
`Strict-5 coverage` is the fraction with all five required source actions
representable; the held-out ratio is refined RMSE divided by baseline RMSE, so
values below 1 improve and values near 1 improve little.

**Skate expert action matching process**

The source policy and BFM Actor do not share the same normalized action
coordinates. The tested physical-target equations were:

```text
q_target_src[j] = q0_src[j] + s_src[j] * a_src[j]
q_target_bfm[j] = q0_bfm[j] + 5 * s_bfm[j] * a_bfm[j]

a_bfm_eq[j] =
    (q0_src[j] + s_src[j] * a_src[j] - q0_bfm[j])
    / (5 * s_bfm[j])
```

Dominant hip rows:

```text
left hip pitch:  a_bfm_eq = +0.090089 + 0.493240 * a_src
right hip pitch: a_bfm_eq = -0.540537 + 0.493240 * a_src
```

| Source-to-BFM method | Physical target RMSE | Decision |
|---|---:|---|
| Copy raw source action | 1.330264 rad | Rejected |
| Divide raw source action by 5 | 0.530560 rad | Rejected |
| Affine inverse, no projection where valid | `1.97e-17 rad` on valid components | Retained |
| Clip out-of-range components to `[-1,1]` | 0.021775 rad global target RMSE | Retained as explicit `PROJECTED` fallback |

**Caption.** Physical-target `RMSE` is the root-mean-square joint-angle error
in radians between source and reconstructed BFM targets; lower is better and
zero is exact. `PROJECTED` means at least one normalized BFM action was clipped
to `[-1,1]`, preserving the BFM interface but losing exact target equality.

Coverage and mismatch:

| Quantity | Result |
|---|---:|
| Raw source action range | `[-4.933043, 6.152792]` |
| Raw components outside `[-1,1]` | 32.1595% |
| Affine BFM-equivalent range | `[-2.343024,1.716985]` |
| Exact component coverage | 99.600839% |
| Exact full-frame coverage | 92.1503% |
| Projected frames | 35,550 |
| Maximum projected hip-tail error | 1.490767 rad |
| Violations recovered by removing default offset | left 3.76% / right 79.41% |
| Range multiplier for 99.9% target coverage | left 1.510x / right 1.860x |
| Range multiplier for full observed coverage | left 2.156x / right 2.343x |

**Caption.** Range values are normalized action coordinates; percentages are
sample coverage, where higher is better. `Exact component coverage` counts
individual joint-frame values inside BFM range; `exact full-frame coverage`
requires every active joint in a frame to be exact. `Projected frames` fail
that full-frame condition. Offset recovery tests recentering only; a range
multiplier enlarges hip-action width. Projected count/error should be lower.

Temporal alignment and controller comparison:

| Check | Result |
|---|---:|
| Phase current/history/strict-five context | 79.9611% / 66.6396% / 64.8366% |
| Continuous current/history/strict-five context | 92.2596% / 87.1721% / 85.8434% |
| Phase `steer2push` current hip violation | 60.15% |
| Phase `steer2push` strict-five failure | 87.86% |
| Hip violation run length | mean 4.32 frames, p95 9, max 14 |
| 4,096-transition refinement RMSE | 0.4689 -> 0.4482 |
| Held-out refinement RMSE ratio | 0.9610 |

**Caption.** `Current valid` checks the selected action, `history valid` also
checks its available history, and `strict five-action context` requires the
current plus four preceding actions. A `hip violation` is an equivalent hip
action outside `[-1,1]`; run length is consecutive violating frames, with
mean/95th percentile/maximum shown. Refinement RMSE compares predicted and
source next state over 4,096 transitions; its held-out ratio is refined over
baseline RMSE. Higher coverage and lower failure/RMSE are better.

**Post-alignment frozen preflight**

Fresh official BFM0, 512 matched Phase resets, 51 steps:

| Metric | Formal aligned/mixed | Pure random |
|---|---:|---:|
| Mean survival | 49.816 | 49.705 |
| Failure by step 20 | 0.00% | 0.00% |
| Failure by step 50 | 26.17% | 26.76% |
| Root tilt p95 | 66.71 deg | 66.93 deg |
| Waist action saturation | 82.33% | 82.69% |
| All-active saturation | 12.05% | 12.13% |

**Caption.** Mean survival is measured in control steps out of a 51-step
horizon, so higher is better. Failure percentages should be lower. Root tilt
is in degrees and `p95` is the 95th percentile. Waist/all-active saturation
is the fraction of waist/all 23 physical actions at the normalized limit;
lower generally indicates more control margin. `Formal aligned/mixed` uses
tracking z in expert-role slots and background z elsewhere; `pure random`
uses background random z in every slot.

First-action comparison:

| Source-action context | Count | Actor/expert cosine | Actor/expert L2 |
|---|---:|---:|---:|
| Fully exact | 421 | 0.4351 | 1.0477 |
| Contains projected component | 91 | 0.1853 | 1.6058 |

**Caption.** Cosine similarity is directional agreement, where higher is
better; `L2` is Euclidean distance between normalized action vectors, where
lower is better. Counts are reset contexts, and these first-action metrics do
not establish closed-loop success.

**Conclusion.** Structural status is `PASS`; behavioral status is
`BEHAVIORAL_DIAGNOSTIC_REQUIRED`.

### Verified and Unverified Conclusions

Verified:

- [x] Base/Skate expert sequences enter the native BFM update at the required
  50/50 sequence ratio.
- [x] Fixed-replay and closed-loop optimization remain finite through the
  completed 20k and 100k runs.
- [x] Checkpoints reload with model, optimizer, and normalizer state.
- [x] The 100k trained checkpoints are less stable than official BFM0 under
  matched frozen evaluation.
- [x] Source physics, robot-board state, active action subspace, and current
  P0 structural contracts pass.
- [x] Source action timing, HUSKY target reconstruction, BFM target
  reconstruction, and exact/projected coverage were experimentally checked.
- [x] The final production action rule is recorded as exact affine translation
  plus explicit projected fallback.
- [x] The P0 check made zero training/update/backward/optimizer calls and did
  not change model, normalizer, or buffer hashes.

Unverified:

- [ ] Tracking z materially improves frozen official BFM0.
- [ ] Hip-pitch and observation semantics are fully resolved.
- [ ] A single action translation reproduces every source physical target.
- [ ] The projected source action is a valid universal BFM expert action.
- [ ] A post-alignment retraining improves policy survival.
- [ ] The current checkpoint is a usable Skate motion library.
- [ ] Held-out validation/test generalization.

- [ ] Periodic training-loss curves committed to the repository.
- [ ] Parameter and gradient norm curves.
- [ ] Phase-conditioned frozen evaluation during training.
- [ ] Formal training rollout videos.
- [ ] Held-out validation/test evaluation.

Available dataset QC videos remain under the Phase and Continuous Hugging Face
directories. Missing training evidence is not reconstructed from smoke output.
