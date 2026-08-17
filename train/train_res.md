# Skate-BFM Training Results

This file contains measured results. `[x]` marks a completed verification.
Unverified work is written as `Pending` rather than as a checked item.

## 1. Dataset Production

### 1.1 Raw Collection

| Quantity | Result |
|---|---:|
| Command cells | 75 |
| Baseline rollouts | 150 |
| Replacement rollouts | 8 |
| Worker failures | 0 |
| Full 60 s rollouts | 125 |
| Fall-terminated rollouts | 33 |
| Frames | 452,885 |
| Duration | 150.962 min |
| Target achieved | [x] |

**Caption.** Raw duration includes complete and fall-terminated episodes. A
fall is a valid episode boundary, not a failed collection worker.

### 1.2 Phase MotionLib

| Phase | Motions | Frames | Duration |
|---|---:|---:|---:|
| push | 1,522 | 182,296 | 60.258 min |
| push2steer | 1,516 | 45,471 | 14.652 min |
| steer_left | 685 | 91,205 | 30.173 min |
| steer_forward | 109 | 14,670 | 4.854 min |
| steer_right | 717 | 96,314 | 31.866 min |
| steer2push | 1,489 | 22,335 | 6.949 min |
| **Total** | **6,038** | **452,291** | **148.751 min** |

**Caption.** A Phase motion is one phase-pure contiguous segment after fall,
pre-fall margin, reset, and minimum-length filtering.

Validation:

- [x] 158/158 raw rollouts processed; zero rollout rejection.
- [x] Official MotionLib loading and Seq8 expert sampling.
- [x] 60 QC samples: 10 per phase, seed `20260813`.
- [x] Raw frame provenance and robot/board/action/phase alignment.

Artifacts:
[Hugging Face / phase](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/phase)

### 1.3 Continuous MotionLib

| Quantity | Result |
|---|---:|
| Clips | 890 |
| Clip size | 500 frames / 10.0 s |
| Stride / overlap | 500 / 0 frames |
| Expert frames | 445,000 |
| Expert duration | 148.333 min |
| Tail frames discarded | 7,310 |
| Clips crossing normal phase transitions | 890/890 |
| Clips crossing fall/reset | 0 |

**Caption.** Continuous clips preserve temporal transitions but never cross a
fall or reset boundary.

Validation:

- [x] Official MotionLib and Seq8 loader.
- [x] Ten post-hoc QC clips.
- [x] Frame-level source provenance.

Artifacts:
[Hugging Face / continuous](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/continuous)

## 2. Expert Mapping Experiments

### 2.1 Pose and Observation Mapping

| Check | Result |
|---|---|
| Shared HUSKY23 -> BFM29 joints | [x] Exact by joint name |
| Missing wrist joints | [x] Six values fixed to zero |
| Root quaternion | [x] normalized `wxyz`, converted to `xyzw` and axis-angle |
| MotionLib record | [x] finite float32 official schema |
| Expert current/next state | [x] `[16,64]` |
| Expert latent | [x] `[1024,256]`, norm 16 |

**Caption.** These checks establish representation compatibility, not
closed-loop physical equivalence.

### 2.2 Source Action Translation

| Metric | Result |
|---|---:|
| Physical-target reconstruction RMSE | `1.97e-17 rad` |
| Maximum reconstruction error | `4.44e-16 rad` |
| Exact component coverage | `99.600839%` |
| Exact full-frame coverage | `92.1503%` |
| Frames with projected hip-tail component | 35,550 |
| Raw-action direct-map target RMSE | `1.330264 rad` |
| Raw-action/5 target RMSE | `0.530560 rad` |
| Exact bridge with clipping target RMSE | `0.021775 rad` |

**Caption.** Exact coverage means the source physical target lies inside the
BFM normalized action range. Remaining errors are concentrated in hip pitch.

Diagnostic 4,096-transition projected-tail refinement:

| Metric | Baseline projection | Refined diagnostic |
|---|---:|---:|
| Normalized next-state RMSE | 0.4689 | 0.4482 |
| Held-out RMSE ratio | 1.0000 | 0.9610 |

**Conclusion.** The refinement was inconsistent for one-sided hip tails and
was not adopted. Production remains exact affine translation plus explicit
clipping marked `PROJECTED`.

## 3. Training-system Validation

### 3.1 Auxiliary Reward Replay

Fixed 1,024-transition replay:

| Raw penalty | Mean | P99 | Nonzero | Weight |
|---|---:|---:|---:|---:|
| torque square | 5,035.389 | 12,397.271 | 100.0% | 0.0 |
| action rate | 0.00994 | 0.21524 | 95.6% | -0.1 |
| DoF position limit | 0.23317 | 0.26182 | 99.6% | -10.0 |
| torque limit | 0.00406 | 0.01449 | 1.8% | 0.0 |
| undesired contact | 0.94531 | 1.00000 | 94.5% | -1.0 |
| feet orientation | 0.89169 | 1.87563 | 97.9% | -0.4 |
| ankle roll | 0.00892 | 0.03058 | 100.0% | -4.0 |
| surface-relative slippage | 0.03366 | 0.47460 | 97.9% | -2.0 |

**Caption.** Values are positive raw penalties. Weighting and normalization
are applied inside the vendored agent.

- [x] All eight keys finite with shape `[1024,1]`.
- [x] Runtime HUSKY joint position and torque limits used.
- [x] Board-relative slippage replaced misleading world-frame slippage.

### 3.2 Termination Replay

| Transition type | Count |
|---|---:|
| Normal | 1,009 |
| Confirmed fall (`terminated`) | 14 |
| Horizon (`truncated`) | 1 |
| Terminal/truncated overlap | 0 |

**Caption.** Ten persistent bad frames are required; temporary feet-off-board
does not terminate the episode.

### 3.3 Native Update Stability

Same 1,024-transition replay and official BFM0 initialization:

| Metric | Update 1 | Update 10 | Update 100 |
|---|---:|---:|---:|
| FB loss | 1,283,845 | 1,053,397 | 286,643 |
| Discriminator loss | 0.4494 | 0.0580 | 0.0125 |
| Main critic loss | 1,796.0 | 1,977.3 | 39.6 |
| Auxiliary critic loss | 248.9 | 171.0 | 53.5 |
| Actor loss | 50,068 | 34,344 | 22,388 |

Completed checks:

- [x] 1/10/100 native update calls.
- [x] F, B, discriminator, main critic, auxiliary critic, Actor, and targets
  changed and remained finite.
- [x] Six optimizer states reached steps 1/10/100.
- [x] z-buffer reached finite capacity 8,192.

**Conclusion.** Fixed-replay optimization is numerically stable over 100
updates; it does not establish policy improvement.

## 4. Adaptation Experiments

### 4.1 B/F-only Base vs Base+Skate

Fixed evaluator, 512 held-out transitions:

| Updates | Group | FB loss | Top-1 | Top-5 | Mean rank |
|---:|---|---:|---:|---:|---:|
| 1 | Base only | 934,192 | 0.3750 | 0.7031 | 5.4531 |
| 1 | Base+Skate | 927,561 | 0.4062 | 0.6719 | 4.9844 |
| 10 | Base only | 1,134,626 | 0.3125 | 0.7500 | 4.7031 |
| 10 | Base+Skate | 1,310,974 | 0.3594 | 0.7812 | 5.0156 |
| 100 | Base only | 1,793,956 | 0.3281 | 0.6875 | 5.2813 |
| 100 | Base+Skate | 1,638,452 | 0.3281 | 0.8281 | 3.8125 |

**Caption.** Top-k is higher-is-better; FB loss and mean rank are
lower-is-better. The 100-update result favors Base+Skate on FB loss, Top-5,
and mean rank, but no physical task or Base-retention evaluation was run.

### 4.2 20k Closed-loop Baseline

| Quantity | Result |
|---|---:|
| Online transitions | 20,000 |
| Update blocks / native updates | 38 / 1,900 |
| Normal / terminal / truncated | 19,592 / 389 / 19 |
| Checkpoints | 10k, 20k |
| Checkpoint reload | [x] |
| Fixed evaluation falls | 60/60 |

Training metric endpoints:

| Metric | Start | 10k | 20k |
|---|---:|---:|---:|
| FB loss | 986,467 | 23,839 | 11,141 |
| Discriminator loss | 0.4382 | 0.3296 | 0.2129 |
| Main critic loss | 1,129.67 | 37.51 | 45.73 |
| Auxiliary critic loss | 166.75 | 8.68 | 36.62 |
| Actor loss | 48,756.79 | 21,856.67 | 6,030.91 |

**Conclusion.** Numerical optimization and checkpointing passed; all fixed
evaluation episodes fell, so task performance is inconclusive.

### 4.3 Formal Phase 100k

| Quantity | Result |
|---|---:|
| Online transitions | 100,000 |
| Update blocks / native updates | 198 / 9,900 |
| Final replay | 100,000 |
| Normal transitions | 96,891 |
| Falls | 3,109 |
| Horizon completions | 0 |
| Expert resets | 3,113 |
| Model/optimizer/normalizer finite | [x] |

Checkpoint integrity:

| Checkpoint | Model SHA256 prefix | Optimizer step | Reload |
|---|---|---:|---|
| 20k | `6c76f0a6ba20` | 1,900 | [x] |
| 50k | `4e7eda31be77` | 4,900 | [x] |
| 100k | `04c51a9bc938` | 9,900 | [x] |

Native update-block endpoint metrics:

| Environment step | FB loss | Disc. loss | Critic loss | Aux critic loss | Actor loss |
|---:|---:|---:|---:|---:|---:|
| 1,500 | 325,215.56 | 0.02593 | 856.73 | 80.34 | 18,889.73 |
| 20,000 | 6,070.78 | 0.20310 | 52.17 | 45.79 | 8,131.50 |
| 50,000 | -3,871.08 | 0.15599 | 12.75 | 3.10 | 954.92 |
| 100,000 | -10,048.30 | 0.08824 | 5.08 | 2.30 | 1,116.23 |

| Environment step | Q_FB | Q_discriminator | Q_aux | B norm | z norm |
|---:|---:|---:|---:|---:|---:|
| 1,500 | 1,093.18 | -323.85 | -74.21 | 16.0 | 16.0 |
| 20,000 | 1,020.97 | -141.49 | -93.65 | 16.0 | 16.0 |
| 50,000 | 325.94 | -65.02 | -32.16 | 16.0 | 16.0 |
| 100,000 | 535.58 | -48.74 | -27.65 | 16.0 | 16.0 |

**Caption.** Values are the final metric in the 50-update block at each
environment step. FB loss can become negative because its objective contains
a negative diagonal term; decreasing loss alone is not evidence of better
closed-loop behavior.

Matched frozen evaluation, 32 episodes each:

| Checkpoint | Mean survival | Min / max | Falls |
|---|---:|---:|---:|
| Official BFM0 | 1.264 s | 0.84 / 2.48 s | 32/32 |
| Phase 20k | 0.604 s | 0.42 / 0.84 s | 32/32 |
| Phase 50k | 0.519 s | 0.36 / 0.66 s | 32/32 |
| Phase 100k | 0.643 s | 0.38 / 1.16 s | 32/32 |

**Conclusion.** Execution and checkpoint integrity passed. Every trained
checkpoint was less stable than official BFM0; behavioral adaptation failed.

## 5. Post-failure Diagnostics

### 5.1 Observation and Control Alignment

| Diagnostic | Result | Conclusion |
|---|---:|---|
| Reset Phase/Continuous shared-frame equality | exact on 32 sampled frames | segmentation did not change representation |
| Waist normalized shift by step 20 | approximately 20-27 sigma | closed-loop divergence |
| Expert/online root angular-velocity scale | 1.0 vs 0.25 | unresolved upstream/current asymmetry |
| Phase strict 5-action context | 64.8366% | moderate context loss |
| Continuous strict 5-action context | 85.8434% | better temporal context |
| `steer2push` strict context | 12.1379% | hardest phase; hip-pitch dominated |

### 5.2 Same-reset Tracking-z

Twenty-step diagnostic:

| Checkpoint | Condition | Completed | Root tilt at step 20 | Action saturation |
|---|---|---:|---:|---:|
| Official BFM0 | random z | 32/32 | not improved | waist divergence remained |
| Official BFM0 | aligned z | 32/32 | not improved | waist divergence remained |
| Historical 100k | random z | 31/32 | 47.8 deg | 17.1% |
| Historical 100k | aligned z | 32/32 | 25.2 deg | 0.29% |

**Caption.** The historical checkpoint benefits in this short diagnostic, but
the official checkpoint does not; this is not a retrained result.

### 5.3 P0 Frozen Matched Preflight

Fresh official BFM0, 512 matched Phase resets, 51-step horizon:

| Metric | Formal mixed/tracking z | Pure random z |
|---|---:|---:|
| Mean survival | 49.816 | 49.705 |
| Failure by step 20 | 0.00% | 0.00% |
| Failure by step 50 | 26.17% | 26.76% |
| Root tilt p95 | 66.71 deg | 66.93 deg |
| Waist saturation `|a|>=0.95` | 82.33% | 82.69% |
| All-active saturation `|a|>=0.95` | 12.05% | 12.13% |

Structural checks:

- [x] 512 matched reset identities and source physics.
- [x] Maximum qpos/qvel reset error `0`; physics mismatches `0`.
- [x] State/history/action/z shapes and finite values.
- [x] Six wrist actions exactly zero.
- [x] Model, normalizer, and buffer hashes unchanged.
- [x] Zero `agent.update`, backward, optimizer, and replay update.

First-step expert comparison:

| Context | Count | Formal Actor/expert cosine | Formal Actor/expert L2 |
|---|---:|---:|---:|
| Fully exact source action | 421 | 0.4351 | 1.0477 |
| At least one projected component | 91 | 0.1853 | 1.6058 |

| First physical-target jump | Formal mixed/tracking z | Pure random z |
|---|---:|---:|
| Mean | 2.1291 | 2.1861 |
| P95 | 3.0798 | 3.0403 |
| Maximum | 4.5244 | 4.5244 |

**Caption.** Expert bridge actions are diagnostic references and were not
executed. Lower cosine and larger L2 in projected contexts show a harder
takeover boundary, not a training loss.

Board-state association:

- largest absolute Spearman correlation with survival: `0.136`;
- failure by step 20 was zero, so no fall-20 correlation was estimable;
- no reset board/relative variable strongly predicted failure.

**Conclusion.** Structural status is `PASS`. Behavioral status is
`BEHAVIORAL_DIAGNOSTIC_REQUIRED`: aligned tracking-z gives no meaningful
advantage and waist saturation remains severe.

## 6. Artifacts and Missing Evidence

Retained public artifacts:

- [x] [Raw HUSKY collection](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/raw)
- [x] [Phase MotionLib and QC videos](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/phase)
- [x] [Continuous MotionLib and QC video](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/continuous)

Pending evidence:

- committed periodic training-loss curves;
- parameter and gradient norm curves;
- phase-conditioned frozen evaluation curves during training;
- committed formal training rollout videos;
- held-out validation/test collection;
- a successful post-alignment retraining result.

No curve or video is reconstructed from unrelated smoke output.
