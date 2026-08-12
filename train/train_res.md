# Training Results

## Experiment 0A: Base-only Control, Original FB + Skate B/F-only Adaptation

- Date: 2026-08-10
- Status: `completed_boundary_validation`
- Objective: measure whether the original BFM-Zero FB objective can adapt the
  pretrained F/B representation to Skate-only online dynamics without
  changing the Actor, discriminator, QD, or Qaux.
- Pretrained source: `model/bfm-zero-official`
- Training replay source: two HUSKY MuJoCo `train_seen_*` rollouts generated
  from the seen dynamics defined by `train/evaluation_protocol.json`.
- Training transitions: `1024`
- Dynamics split: `seen` only
- Unseen dynamics in training: `0`
- Evaluation transitions in training: `0`
- Expert source in the executed checkpoints: Base-only. The saved configs
  have `skate_expert_motion_file=null`; the configured Skate ratio `0.5` did
  not take effect because no `expert_skate` buffer was present. The results
  must not be interpreted as the requested Base+Skate expert-mixture
  experiment.
- Configured Skate expert ratio: `0.5`
- FB latent preparation: `encode_expert()` + vendored `sample_mixed_z()` +
  `relabel_ratio=0.8`; no RFB/vMF or dynamics context.
- Update milestones: `0`, `1`, `10`, `100`; each starts from the same official
  BFM0 checkpoint.
- Learning rates: F `0.0003`, B `0.00001`
- FB configuration: discount `0.98`, `fb_target_tau=0.01`,
  `ortho_coef=100.0`, `q_loss_coef=0.0`, gradient clipping disabled.
- Observation normalizer: changed according to the upstream FBcprAux update
  semantics from online `train_obs` and `train_next_obs`.
- Optimizer boundary: only F/B optimizer steps; full `agent.update()` calls
  `0`.
- Checkpoint paths:
  `results/m2.2b-1/update_1/checkpoint`,
  `results/m2.2b-1/update_10/checkpoint`,
  `results/m2.2b-1/update_100/checkpoint`

### Fixed Evaluation

Protocol: `skate-bfm-fixed-eval-v1`, 512 held-out transitions, identical
rollout/dynamics seeds for all checkpoints.

| Updates | FB loss | FB diag | FB offdiag | Orth loss | Margin | Top-1 | Top-5 | Mean rank | Median rank | Entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1692189.5000 | -124.2924 | 40884.6680 | 16514.2910 | 846.7319 | 0.2656 | 0.7344 | 5.2969 | 3 | 4.703526 |
| 1 | 934192.1250 | 120.5836 | 20737.3379 | 9133.3418 | 1274.1001 | 0.3750 | 0.7031 | 5.4531 | 2 | 4.853298 |
| 10 | 1134625.5000 | 251.0973 | 21534.2480 | 11128.4014 | 769.3938 | 0.3125 | 0.7500 | 4.7031 | 2 | 4.115609 |
| 100 | 1793956.3750 | 187.2596 | 10608.6309 | 17831.6055 | 520.5139 | 0.3281 | 0.6875 | 5.2813 | 2 | 3.558375 |

The result is a fixed diagnostic baseline, not a claim of downstream Skate
success. The best single metric is not used to select a checkpoint:
`update_10` has the lowest mean rank, while `update_1` has the lowest FB loss.

### Boundary Validation

- `1`, `10`, and `100` direct `update_fb()` calls: `PASS`
- F/B gradients and update losses finite: `PASS`
- F, B, target F, target B changed: `PASS`
- Actor unchanged: `PASS`
- Discriminator unchanged: `PASS`
- QD and target QD unchanged: `PASS`
- Qaux and target Qaux unchanged: `PASS`
- Skate-only replay and alias preservation: `PASS`
- Unseen dynamics leakage: `PASS`, count `0`
- Evaluation leakage: `PASS`, count `0`
- Fixed Skate evaluation repeatability: `PASS`
- Base retention: `NOT RUN`, because IsaacLab is unavailable and the
  one-environment MuJoCo fallback is not comparable to the fixed 1024-env
  protocol.
- Native termination: `UNRESOLVED`
- Qaux: `UNRESOLVED`
- Command-aligned downstream evaluation: `UNRESOLVED`
- Formal full training: `NO`

Local experiment artifacts are intentionally excluded from Git by the
`results/` ignore rule because the three checkpoints and optimizer states are
multi-gigabyte files.

## Experiment 0B: Base + Skate 50/50 Treatment, Original FB + Skate B/F-only Adaptation

- Date: 2026-08-10
- Status: `completed_boundary_validation`
- Objective: rerun the intended Base+Skate expert-mixture treatment with the
  same B/F-only adaptation boundary as Experiment 0A.
- Control: Experiment 0A, Base-only expert plus the same Skate replay.
- Treatment: Base expert plus Skate expert at a complete-sequence ratio of
  `50/50`, plus the same Skate replay.
- Pretrained source: `model/bfm-zero-official`
- Skate expert artifact:
  `train/dataset/skate-expert-pose/motion_library/skate_expert.pkl`
- Skate artifact SHA256:
  `660c18145a21457d3541b49ccc802ba3f99170804836cedaebb9d245b837fd86`
- Skate artifact schema: 1 motion, 50 frames, 50 Hz, 29 DoF, duration 0.98 s.
- Expert mixture: 64 complete Base sequences + 64 complete Skate sequences,
  sequence length 8, total 128 sequences.
- Training replay: 1024 Skate transitions, 0 evaluation transitions, 0 unseen
  transitions. The `train` alias remained the same object as `train_skate`.
- Replay tensor fingerprint:
  `ad1b476ed5dd266572001050e6db809e90b9eb46e1bd254ec67b4e0101f65fbf`
- Transition-ID fingerprint:
  `fcb90076cd23ab76fe273ef7abea44eabd7f83249121847e5f48c9c51e364c96`
- Dynamics-realization fingerprint:
  `26999c6e7ea8d62989048951708027c421163f9693f9fe50cbffc28ea683aa69`
- Update milestones: `1`, `10`, and `100`; each is an independent run from
  the official checkpoint.
- Optimizer boundary: direct vendored `update_fb()` only. F/B and target F/B
  changed; Actor, discriminator, QD, Qaux, and their target modules did not.
- All three runs: 1024 transitions, 1/10/100 direct `update_fb()` calls,
  forbidden optimizer calls `0`, evaluation leakage `0`, unseen leakage `0`.
- Fixed evaluator: `skate-bfm-fixed-eval-v1`, 512 held-out transitions, with
  the same protocol and evaluator inputs as Experiment 0A.

### Base-only vs Base+Skate Fixed Evaluation

Delta is treatment minus control. Lower is better for FB loss and mean rank;
higher is better for margin, Top-1, and Top-5. Entropy is a coverage monitor.

| Updates | Group | FB loss | FB diag | FB offdiag | Orth loss | Margin | Top-1 | Top-5 | Mean rank | Median rank | Entropy |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Base-only control | 934192.1250 | 120.5836 | 20737.3379 | 9133.3418 | 1274.1001 | 0.3750 | 0.7031 | 5.4531 | 2 | 4.853298 |
| 1 | Base+Skate treatment | 927561.1875 | 117.4835 | 21954.1328 | 9054.8955 | 1294.0538 | 0.4062 | 0.6719 | 4.9844 | 2 | 4.792666 |
| 10 | Base-only control | 1134625.5000 | 251.0973 | 21534.2480 | 11128.4014 | 769.3938 | 0.3125 | 0.7500 | 4.7031 | 2 | 4.115609 |
| 10 | Base+Skate treatment | 1310973.5000 | 217.3763 | 16474.6445 | 12942.8154 | 667.7080 | 0.3594 | 0.7812 | 5.0156 | 2 | 4.349761 |
| 100 | Base-only control | 1793956.3750 | 187.2596 | 10608.6309 | 17831.6055 | 520.5139 | 0.3281 | 0.6875 | 5.2813 | 2 | 3.558375 |
| 100 | Base+Skate treatment | 1638451.7500 | 180.8057 | 10464.7832 | 16278.0615 | 544.9000 | 0.3281 | 0.8281 | 3.8125 | 2 | 3.591432 |

### Treatment Deltas

| Updates | FB loss | Margin | Top-1 | Top-5 | Mean rank | Entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -6630.9375 | +19.9537 | +0.0312 | -0.0312 | -0.4688 | -0.060631 |
| 10 | +176348.0000 | -101.6858 | +0.0469 | +0.0312 | +0.3125 | +0.234153 |
| 100 | -155504.6250 | +24.3861 | 0.0000 | +0.1406 | -1.4688 | +0.033057 |

### Treatment Conclusion

The treatment is `PASS` as an executed Base+Skate boundary experiment and is
strictly more auditable than the old run because the expert source, sequence
mixture, replay tensors, transition IDs, and dynamics realization are
fingerprinted. The metric result is mixed at update 1/10 and favorable at
update 100 for FB loss, margin, Top-5, and mean rank. It is not evidence of
downstream skate-task success: Base retention was not run, native termination,
Qaux, and command-aligned evaluation remain unresolved. The treatment should
therefore be described as a promising but non-conclusive representation
adaptation result, not as a generally better model.

The historical Base-only control has no replay tensor fingerprint, so exact
historical tensor identity cannot be retroactively proven. Its rollout IDs and
dynamics seeds remain comparable under the same canonical protocol.

## M2.2b-2 Baseline Reproducibility + Evaluator Fidelity Audit

- Date: 2026-08-10
- Status: `completed_with_provenance_correction`
- Experiment 0A remains the Base-only control boundary. The previously
  misconfigured Base+Skate treatment was rerun as Experiment 0B in M2.2b-3;
  no control checkpoint was retrained.
- FB evaluator matches vendored `FBAgent.update_fb()`: `PASS`
- Suspected `fb_diag` issue: `NOT CONFIRMED`
- Canonical evaluator version: `skate-bfm-fixed-eval-v1`
- Protocol conditions: unchanged
- Current official checkpoint `model.safetensors` SHA256:
  `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`
- Historical checkpoint identity: `UNVERIFIABLE`
- Historical commit reproduction: `PARTIAL`; current official bytes under
  the d4 evaluator reproduced the current v1 metrics, not the old unproven
  M2.2b-0 record.
- Clean-process repeatability: `PASS`
- Base retention: `NOT RUN`
- Native termination: `UNRESOLVED`
- Qaux: `UNRESOLVED`
- Command-aligned downstream: `UNRESOLVED`

### Canonical v1 Results

All rows below use the same evaluator version, current checkpoint policy,
fixed protocol, and provenance-complete artifacts.

| Updates | FB loss | FB diag | FB offdiag | Orth loss | Margin | Top-1 | Top-5 | Mean rank | Median rank | Entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1692189.5000 | -124.2924 | 40884.6680 | 16514.2910 | 846.7319 | 0.2656 | 0.7344 | 5.2969 | 3 | 4.703526 |
| 1 | 934192.1250 | 120.5836 | 20737.3379 | 9133.3418 | 1274.1001 | 0.3750 | 0.7031 | 5.4531 | 2 | 4.853298 |
| 10 | 1134625.5000 | 251.0973 | 21534.2480 | 11128.4014 | 769.3938 | 0.3125 | 0.7500 | 4.7031 | 2 | 4.115609 |
| 100 | 1793956.3750 | 187.2596 | 10608.6309 | 17831.6055 | 520.5139 | 0.3281 | 0.6875 | 5.2813 | 2 | 3.558375 |

The previous M2.2b-0 and M2.2b-1 values are retained as historical records,
but the old M2.2b-0 baseline is superseded for canonical comparison because
its checkpoint identity cannot be verified. The numeric v1 formula itself was
not changed.

### Audit Conclusion

- Baseline drift root cause: unresolved beyond Level 1 because the historical
  checkpoint hash was never recorded. The current checkpoint, current v1
  evaluator, and d4 evaluator are mutually reproducible.
- Shared normalizer affects frozen Actor behavior: `YES`
- Parameter-frozen Actor: `YES`
- Functionally-frozen policy: `NO`
- Ready for next adaptation decision: `NO`
- Historical correction: the executed M2.2b-1 checkpoints were Base-only, not
  the intended Base+Skate expert mixture. Experiment 0B now records the
  correctly configured treatment separately.

## M2.3a-0 Target Bank + Command Alignment Audit

- Date: 2026-08-11
- Status: `completed_read_only_audit`
- Training: `NO`
- Rollout: `NO`
- Actor execution: `NO`
- Optimizer steps: `0`
- Target bank schema: `skate-bfm-target-bank-v1`
- Target bank output:
  `train/dataset/skate-expert-pose/target_bank/target_bank.json`
- Source raw rollout:
  `/tmp/skate_bfm_m1_1.0L6F7i/round_901/rollout_001/raw_rollout/m1_1_rollout_001.npz`
- Raw rollout SHA256:
  `5476a280ec013f3834dbb4a5cef1a9d80c0df6728fe7ebc98dc3a2e3e1f11c53`
- Raw metadata SHA256:
  `72fdde180d753744645a7da77fb7e388e7e16165b7eb4853cb146825dc9ebc58`
- Expert MotionLib SHA256:
  `660c18145a21457d3541b49ccc802ba3f99170804836cedaebb9d245b837fd86`
- Raw source fields audited: robot root pose/velocity, board pose/velocity,
  23-DoF joint pose/velocity, 23-DoF action, phase, timestamps, fall/reset,
  and per-frame `command_v`/`command_h`. All required fields are present,
  frame-aligned, finite, and no unavailable field was substituted.
- Global physical behavior: one continuous `push` phase over 50 frames,
  board displacement `[0.6941, 0.0027, -0.0005] m`, mean forward board
  velocity `0.7001 m/s`, mean lateral velocity `0.0029 m/s`, and board
  heading delta `+1.1642 deg`. No fall or reset occurred.
- Command audit: `command_v=1.0` is the forward linear-velocity command
  scalar; `command_h=0.0` is the relative heading/steering command in radians.
  Both are present in metadata and every frame. The physical behavior is
  `aligned` for forward motion with zero heading command.
- No steer-left/right target was created. `steer_left`, `steer_right`, and
  dynamic turning are `NOT_FOUND` in this artifact.

### Target Bank

| Target | Frames | Time | Physical target | Alignment |
| :--- | ---: | ---: | :--- | :--- |
| `skate_target_00` | 24-31 | 0.48-0.62 s | Forward board acceleration / push | `aligned` |

Selection used disjoint 8-frame windows, excluded fall windows, and selected
the highest forward board velocity subject to mean lateral velocity at most
`0.01 m/s` and board yaw change at most `1 deg`. The selected window has mean
forward velocity `0.8116 m/s`, mean lateral velocity `-0.0023 m/s`, and yaw
delta `+0.3580 deg`.

Target latent inference used the exact `encode_expert()` mathematical path:
normalized expert next observations, `B` per frame, sequence mean over 8
frames, then `project_z`. It was run independently for three frozen
checkpoints:

| Checkpoint | z norm | Latent fingerprint |
| :--- | ---: | :--- |
| Official BFM0 | 16.000000 | `097548cb634b08a1e688ea0565c692c0c084b381fbd1ef819a9e144678ec2d75` |
| Base-only update100 | 15.999999 | `8cd703835dbca2f350e2a087de976938e3c038648070980057fbab4d79cb25b7` |
| Base+Skate update100 | 16.000000 | `fc203e2097ec26cc3c3b9187aa994efa2bf5b161175c8685ac8667491e325228` |

Cosine similarities are diagnostics only:

- Official vs Base-only: `0.102484`
- Official vs Base+Skate: `0.102463`
- Base-only vs Base+Skate: `0.999963`

The target label was assigned from raw physical state and command metadata,
never from latent similarity. The target bank is a definition/audit artifact,
not evidence of command-aligned downstream task success.

## M2.3b-0 Downstream Target-Conditioned Evaluation

- Date: `2026-08-11`
- Status: `completed_evaluation_only`
- Target: `skate_target_00`, frames `24-31`, aligned forward push.
- Checkpoints: `official_bfm0`, `base_only_update100`, and
  `base_skate_update100`.
- Random bank: fixed seeds `2026081101`, `2026081102`, `2026081103`, and
  `2026081104`; each checkpoint used its own matched random controls.
- Dynamics: `seen_001`, `seen_002`, `unseen_001`, and `unseen_002`.
- Horizon: `128` steps at `0.02 s`; canonical reset; no frame-24 teleport;
  no command injection into the Actor.
- Rollout count: `60` (`3 x 4 x (4 random + 1 target)`).
- Result artifact:
  `results/m2.3b-0-target-conditioned/target_conditioned_metrics.json`.
- Target latent fingerprints and norms:

  | Checkpoint | z fingerprint | z norm |
  | :--- | :--- | ---: |
  | Official BFM0 | `097548...` | 16.000000 |
  | Base-only update100 | `8cd703...` | 15.999999 |
  | Base+Skate update100 | `fc203e...` | 16.000000 |

### Seen Dynamics

Values are `random mean +/- std; target mean; target minus random mean`.

| Checkpoint | Forward displacement (m) | Forward velocity (m/s) | Lateral drift (m) | Heading drift (deg) |
| :--- | ---: | ---: | ---: | ---: |
| Official BFM0 | `-0.104 +/- 0.279; 1.239; +1.343` | `-0.041 +/- 0.109; 0.485; +0.526` | `0.058 +/- 0.081; 0.088; +0.030` | `10.116 +/- 10.738; 5.543; -4.573` |
| Base-only update100 | `1.007 +/- 0.345; 1.641; +0.634` | `0.394 +/- 0.135; 0.642; +0.248` | `0.092 +/- 0.036; 0.177; +0.085` | `5.650 +/- 1.656; 6.677; +1.028` |
| Base+Skate update100 | `0.880 +/- 0.517; 1.637; +0.758` | `0.344 +/- 0.202; 0.641; +0.297` | `0.086 +/- 0.041; 0.187; +0.102` | `6.071 +/- 1.272; 7.199; +1.129` |

### Unseen Dynamics

| Checkpoint | Forward displacement (m) | Forward velocity (m/s) | Lateral drift (m) | Heading drift (deg) |
| :--- | ---: | ---: | ---: | ---: |
| Official BFM0 | `-0.364 +/- 0.835; 0.209; +0.573` | `-0.142 +/- 0.328; 0.082; +0.224` | `0.070 +/- 0.081; 0.307; +0.237` | `9.992 +/- 8.832; 17.528; +7.536` |
| Base-only update100 | `1.218 +/- 0.172; 1.781; +0.563` | `0.477 +/- 0.067; 0.697; +0.221` | `0.113 +/- 0.040; 0.186; +0.073` | `5.695 +/- 1.991; 6.540; +0.845` |
| Base+Skate update100 | `1.203 +/- 0.236; 1.734; +0.532` | `0.471 +/- 0.092; 0.679; +0.208` | `0.106 +/- 0.036; 0.175; +0.068` | `5.617 +/- 1.958; 6.350; +0.733` |

### Boundary Checks

- Target-bank schema and command alignment: `PASS`.
- Runtime target encoding: `PASS` for all three checkpoint fingerprints.
- Canonical reset reproducibility: `PASS` for all four dynamics IDs.
- Full parameter, component, and buffer mutation: `NO`.
- Optimizer steps: `0`; backward calls: `0`; `agent.update`: `0`;
  `update_fb`: `0`.
- Target latent was runtime-recomputed from the frozen checkpoint and the
  expert window. The tracked JSON stores fingerprints and metadata, not the
  full 256-dimensional latent values.

### Conclusion

The target shows a **consistent forward-response advantage** over the matched
random bank for both seen and unseen dynamics on all three checkpoints.
However, lateral drift and heading drift are not consistently reduced, and
the target can increase them. The overall target-conditioned physical
response is therefore **mixed**, not a task-success result.

Base-only and Base+Skate update100 have similar target forward response.
The current preflight does **not support** a downstream physical advantage
for Base+Skate over Base-only. Native termination, Base retention, Qaux, and
full FB-CPR-Aux training remain outside this evaluation.

## M2.4-0 Project Code Cleanup

This is an engineering-only refactor record, not a training experiment.

- Date: `2026-08-11`
- Training: `NO`
- Optimizer steps: `0`
- Model, loss, replay, sampling, observation, action, termination, reward,
  metrics, and evaluation protocol behavior: unchanged.
- Current entrypoints:
  - `train/scripts/build_target_bank.py`
  - `train/scripts/eval_target.py`
  - `train/scripts/evaluate_skate_bfm.py` (historically retained filename)
- The two target entrypoints no longer import each other or the canonical
  evaluator. Shared checkpoint, hash, expert MotionLib, and target-encoding
  operations use the project-owned training runtime.
- Vendored BFM-Zero source changes: `0`.
- Regression status: target-bank byte equality, target rollout fingerprint
  equality, and canonical 512-transition evaluator numerical equality:
  `PASS`.

## M2.4a Training Readiness

This is a read-only dependency audit, not a training experiment.

- Date: `2026-08-11`
- Resolved agent: `FBcprAuxAgent`
- Full native update call graph: `PASS`
- Runtime batch / sequence length: `1024 / 8`
- Replay: `PARTIAL`
  - `train is train_skate`: yes
  - core observation, 29D action, 256D z, next observation, terminated, and
    truncated contracts: finite and shape-compatible
  - configured auxiliary reward dictionary: absent
- Expert: `READY`
  - Base source: `862` LAFAN motions
  - Skate source: `1` motion, `50` frames, `50 Hz`, forward push
  - sampled mixture: `64 Base + 64 Skate` complete 8-frame sequences
- Termination: `PARTIAL`
  - current Skate transitions never set `terminated=True`
  - bounded horizon sets `truncated=True`
  - fall, invalid state, and board separation are not terminal
  - reset-crossing transitions are prevented
- Discriminator: `READY`
- F/B: `READY`
- Main critic / QD: `READY`
- Qaux network: `READY`
- Qaux data: `BLOCKED`
- Actor network and 29D output: `READY`
- Actor training interface: `BLOCKED` by missing Qaux reward data
- Target F, B, QD, and Qaux: `READY`
- Observation and auxiliary reward normalizer state: `READY`
- Configured auxiliary rewards:
  `penalty_torques`, `penalty_action_rate`, `limits_dof_pos`,
  `limits_torque`, `penalty_undesired_contact`, `penalty_feet_ori`,
  `penalty_ankle_roll`, `penalty_slippage`.
- Auxiliary reward data: `BLOCKED`.
  `penalty_action_rate`, `limits_dof_pos`, and `penalty_ankle_roll` have at
  least partial source-state support; exact torque, contact-gated foot
  orientation, undesired-contact, and slippage contracts are unavailable.
  Zero-scaled torque keys are still accessed by upstream and cannot be
  omitted or filled silently.

| Readiness judgment | Result |
| :--- | :--- |
| Representation training ready | `YES` |
| Critic/discriminator interface ready | `YES` |
| Actor training interface ready | `NO` |
| Full `FBcprAuxAgent.update()` ready | `NO` |

The first hard blocker is the missing configured Skate auxiliary reward
contract. The next milestone is:

`M2.4b — Skate Auxiliary Reward Contract`

The single 50-frame push expert is a performance/coverage limitation rather
than a technical smoke-training blocker. Previous frozen-Actor behavior does
not predict full FB-CPR-Aux training performance.

Validation:

- Full 1024-item forward checks: finite for expert z, mixed z,
  discriminator logits/reward, F/B, QD, Qaux, Actor, and all targets.
- Actor output: `[1024, 29]`.
- Parameter mutation: `NO`.
- Model-buffer mutation: `NO`.
- Optimizer steps: `0`.
- Backward calls: `0`.
- `agent.update`, `update_fb`, `update_actor`, `update_critic`,
  `update_aux_critic`, and `update_discriminator` calls: `0`.
- Training performed: `NO`.

## M2.4b-1 Phase-wise Expert Reward Audit

This is a read-only reward-semantics audit, not a training experiment.

- Date: `2026-08-11`
- Training, optimizer steps, backward calls, `agent.update`, `update_fb`, and
  `update_actor`: `NO`
- Formal replay modification: `NO`
- Auxiliary reward training-semantics modification: `NO`
- Formal MotionLib scope: one 50-frame forward-push expert. It is insufficient
  for steer-phase analysis.
- Diagnostic-only sources: one recorded left-steer and one recorded
  right-steer HUSKY policy rollout. They were not added to MotionLib or
  training replay.
- Recorded phase coverage: push `342` frames, push2steer `58`, steer `270`,
  steer2push `30`; `steer_forward` and `fall`: `PHASE NOT AVAILABLE`.
- Phase-local MuJoCo fidelity: `PASS` for both rollouts. Joint RMSE was
  `6.79e-6` / `6.04e-6` rad; root-position RMSE was `6.23e-6` /
  `4.13e-6` m; board-position RMSE was `5.56e-6` / `4.99e-6` m.

| Phase | 29D action rate | World slip | Board-relative slip | World feet ori | Surface feet ori | Ankle roll | Original aux | Surface candidate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| push | 30.070 | 1.028 | 0.020 | 0.035 | 0.040 | 0.011 | -5.130 | -3.206 |
| push2steer | 66.355 | 2.883 | 0.202 | 0.122 | 0.113 | 0.013 | -12.501 | -7.204 |
| steer | 1.496 | 3.126 | 0.044 | 0.297 | 0.294 | 0.024 | -6.615 | -0.450 |
| steer2push | 46.363 | 1.754 | 0.204 | 0.075 | 0.085 | 0.013 | -8.228 | -5.275 |

Conclusions:

- `penalty_slippage`: `REDEFINE`. World-frame slippage penalizes legitimate
  board-supported foot transport during steer; board-relative slippage removes
  that sustained conflict.
- `penalty_action_rate`: `KEEP_WITH_MAPPING`. The fixed BFM wrist dimensions
  contribute zero; high action rates are confined to phase transitions.
- `limits_dof_pos`: `KEEP_WITH_MAPPING`; `penalty_undesired_contact`: `KEEP`;
  `penalty_feet_ori`: `KEEP`.
- `penalty_torques` and `limits_torque`: `DIAGNOSTIC_ONLY`.
- `penalty_ankle_roll`: `ABLATION_REQUIRED`; its steer increase was not
  consistently coupled to board roll across the two directions.

The ignored detailed report is
`results/m2.4b-1-reward-audit/summary.json`, with a frame trace and four
diagnostic figures. Full `FBcprAuxAgent.update()` remains `NOT READY`. Next
milestone: `M2.4b-2 — Skate Aux Reward Contract`.

## M2.4b-2 Skate Auxiliary Reward Contract

This is a collect-only replay validation, not a training experiment.

- Date: `2026-08-11`
- Formula authority: vendored BFM-Zero reward definitions.
- Physical constraint authority: the active HUSKY MuJoCo `MjModel`.
- Physical mapping: all 23 robot actuators are one-to-one `mjTRN_JOINT`
  hinge transmissions with finite symmetric force ranges and
  `gear=[1, 0, 0, 0, 0, 0]`. `qfrc_actuator[joint_dof]` and the derived
  `forcerange * gear[0]` joint-torque limit are therefore in the same units.
- Position limits: all 23 HUSKY limits match the mapped upstream positions.
- Torque-limit provenance difference: `g1_29dof_hard_waist` has two
  hip-pitch limits at `139` where HUSKY uses `88`; `g1_29dof` instead differs
  at the two hip-roll limits. These are not used as HUSKY physical limits.
- Formal replay: `1024` transitions; `train is train_skate`: `YES`.
- Reward keys: `8 / 8`, all finite `[1024,1]`; reward normalizer updates:
  `0`.
- Training, optimizer steps, backward calls, `agent.update`, `update_fb`, and
  `update_actor`: `0`.

### Raw Contract

The replay stores positive raw penalty magnitudes only. Scaling remains in the
vendored agent and is unchanged:

| Key | Physical definition | Scale |
| :--- | :--- | ---: |
| `penalty_torques` | sum of squared 23D `qfrc_actuator` joint torques | `0.0` |
| `penalty_action_rate` | squared delta of consecutive clipped executed 23D actions | `-0.1` |
| `limits_dof_pos` | sum outside the 95% HUSKY joint-position soft range | `-10.0` |
| `limits_torque` | sum above 95% of HUSKY derived joint torque limits | `0.0` |
| `penalty_undesired_contact` | binary pelvis/shoulder/hip contact with ground or board above force threshold | `-1.0` |
| `penalty_feet_ori` | contact-gated world-horizontal foot-normal penalty | `-0.4` |
| `penalty_ankle_roll` | left plus right ankle-roll joint square | `-4.0` |
| `penalty_slippage` | dominant-contact surface-relative tangential foot velocity | `-2.0` |

No task, command, balance, steering, board-displacement, or forward reward was
added. The 29D BFM action remains the replay action; only action-rate uses the
executed 23D HUSKY action.

### Formal Replay Distribution

| Key | Mean | Std | P50 | P90 | P99 | Max | Nonzero |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `penalty_torques` | 5035.389 | 1272.722 | 4807.947 | 5031.573 | 12397.271 | 18206.387 | 1.000 |
| `penalty_action_rate` | 0.00994 | 0.06245 | 0.000002 | 0.00442 | 0.21524 | 0.97349 | 0.956 |
| `limits_dof_pos` | 0.23317 | 0.03750 | 0.25138 | 0.25611 | 0.26182 | 0.33150 | 0.996 |
| `limits_torque` | 0.00406 | 0.06398 | 0.00000 | 0.00000 | 0.01449 | 1.55018 | 0.018 |
| `penalty_undesired_contact` | 0.94531 | 0.22737 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.945 |
| `penalty_feet_ori` | 0.89169 | 0.38899 | 0.81264 | 1.84089 | 1.87563 | 1.93118 | 0.979 |
| `penalty_ankle_roll` | 0.00892 | 0.00613 | 0.00736 | 0.01380 | 0.03058 | 0.09096 | 1.000 |
| `penalty_slippage` | 0.03366 | 0.09158 | 0.00741 | 0.08801 | 0.47460 | 1.08866 | 0.979 |

The weighted raw auxiliary diagnostic has mean `-3.73774`, standard deviation
`0.46718`, P50 `-3.90735`, P90 `-3.54140`, P99 `-1.76675`, and max
`-0.05905`. It is a sanity metric only; the upstream agent retains the
weighted-sum implementation and the normalizer was not updated.

### M2.4b-1 Regression

Two independent phase-rich HUSKY policy rollouts replayed with phase-local
fidelity `PASS`. Production reward computation agreed with the audit for
action rate, DoF limits, undesired contact, world-horizontal feet orientation,
ankle roll, and surface-relative slippage across all `350 + 350` frames.

| Rollout | Steer world slip | Production surface-relative slip |
| :--- | ---: | ---: |
| left steer | 3.14840 | 0.04082 |
| right steer | 3.10389 | 0.04815 |

This preserves M2.4b-1's conclusion: world-frame foot velocity is high during
legitimate board transport, while the production contact-surface-relative
penalty remains low.

| Readiness judgment | Result |
| :--- | :--- |
| Replay | `READY` |
| Auxiliary reward data | `READY` |
| Qaux data | `READY` |
| Actor training interface | `READY` |
| Native termination | `PARTIAL` |
| Full `FBcprAuxAgent.update()` | `NO` |

The next blocker is native physical termination. Next milestone:
`M2.4c — Native Termination Contract`.

## M2.4c Native Fall Termination Contract

This is a collect-only termination validation, not training.

- Date: `2026-08-11`
- Fall source: the shared `LiveFallDetector` used by raw Skate expert
  collection and `HuskyLiteEnv`.
- Fall definition: persistent root tilt above `70 deg`, or persistent root
  height below `0.45 m` with an illegal collision. Confirmation is `0.2 s`,
  computed from control time (`10` frames at 50 Hz).
- Online fall: `terminated=True`, `truncated=False`.
- Fixed horizon: `terminated=False`, `truncated=True`.
- Precedence: a confirmed fall on the final collection step is terminal, not
  horizon-truncated.
- Board separation alone: `terminated=False`.
- Temporary feet-off-board alone: `terminated=False`.
- Fall recovery: `NOT SUPPORTED`.

### Controlled Validation

| Case | Result |
| :--- | :--- |
| Normal canonical state | `terminated=False` |
| Single 90-degree tilt frame | `terminated=False` |
| Persistent 90-degree tilt | `terminated=True` |
| Persistent `0.2 m` root height plus illegal contact | `terminated=True` |
| Board moved 5 m away | `terminated=False` |
| Collection / online detector implementation | `PASS` (same class) |
| Online terminal then step without reset | `PASS` (raises) |

### Formal Replay

- Formal collect-only replay: `1024` transitions; `train is train_skate`:
  `YES`.
- `next.terminated` and `next.truncated`: boolean `[1024,1]` tensors.
- Counts: `14` terminated, `1` truncated, `1009` normal, `0` overlap.
- Discount preflight: upstream `gamma * ~terminated`; terminal rows have
  discount `0`, non-terminal rows have `gamma`.
- Model parameters and buffers: unchanged.
- `agent.update`, backward, and optimizer calls: `0`.

| Readiness judgment | Result |
| :--- | :--- |
| Replay | `READY` |
| Auxiliary reward data | `READY` |
| Qaux data | `READY` |
| Actor training interface | `READY` |
| Native termination | `READY` |
| Full `FBcprAuxAgent.update()` dependencies | `READY` |

Next milestone: `M2.4d — Native Full-Update Smoke`.

## M2.4d-1 Native Full-Update Smoke

This is one native full-update execution smoke, not a skill-quality result.

- Date: `2026-08-11`
- Checkpoint: official BFM0, strict 537-key model load; no optimizer state was
  restored.
- Expert mixture: 128 complete sequences of length 8, with 64 Base and 64
  Skate sequences.
- Replay: 1024 HUSKY transitions, 14 terminated, 1 truncated, 1009 normal;
  `train is train_skate`: `YES`.
- Native `FBcprAuxAgent.update()` calls: `1`; direct `update_fb()` calls from
  Skate project code: `0`.
- All eight raw auxiliary reward keys were read. `penalty_torques` and
  `limits_torque` retained scale `0.0`.
- No smoke checkpoint was saved. This result is diagnostic-only and is not an
  M2.5 initialization.

### Native Metrics

| Metric | Value |
| :--- | ---: |
| `fb_loss` | 1283845.2500 |
| `disc_loss` | 0.4494 |
| `critic_loss` | 1796.0070 |
| `aux_critic_loss` | 248.8505 |
| `mean_aux_reward` | -3.1395 |
| `actor_loss` | 50068.2305 |
| `Q_discriminator` | -332.1558 |
| `Q_aux` | -101.3127 |
| `Q_fb` | 2837.1506 |

All returned native metrics were finite. The ignored machine-readable report
is `results/m2.4d-1-native-full-update/summary.json`.

| Audit | Result |
| :--- | :--- |
| F / B / discriminator / QD / Qaux / Actor mutation | `PASS` |
| target F / B / QD / Qaux mutation | `PASS` |
| Six fresh optimizer states | `PASS`, all at step `1` |
| Observation normalizer finite and changed | `PASS` |
| Auxiliary reward normalizer finite and changed | `PASS` |
| z-buffer | `0 -> 1024`, `PASS` |
| Full native update | `PASS` |

Next milestone: `M2.4d-2 — Short Multi-Update Stability Smoke`.

## M2.4d-2 Short Multi-Update Stability Smoke

This is a fixed-replay numerical stability smoke, not a performance or
convergence experiment.

- Date: `2026-08-12`
- Checkpoint: official BFM0; SHA256
  `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- Expert mixture per update: 64 Base and 64 Skate complete sequences,
  sequence length `8`, batch size `1024`.
- Replay collected once: `1024` transitions, `14` terminated, `1` truncated,
  `1009` normal; `train is train_skate`: `YES`.
- Native update calls: `10`; no direct project `update_fb`, actor, critic,
  optimizer, or soft-update calls.

### Metric Table

| Update | `fb_loss` | `disc_loss` | `critic_loss` | `aux_critic_loss` | `actor_loss` | `Q_fb` | `Q_discriminator` | `Q_aux` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1283845 | 0.4494 | 1796.0 | 248.9 | 50068 | 2837.2 | -332.2 | -101.3 |
| 2 | 1307751 | 0.2481 | 1481.9 | 169.7 | 50001 | 2842.5 | -331.4 | -100.5 |
| 3 | 1221122 | 0.1429 | 1908.0 | 223.7 | 48599 | 2743.6 | -334.2 | -100.1 |
| 4 | 1179586 | 0.0933 | 2607.9 | 279.1 | 44554 | 2536.0 | -332.1 | -98.1 |
| 5 | 1145578 | 0.0753 | 1486.6 | 154.7 | 45453 | 2615.0 | -329.3 | -94.6 |
| 6 | 1104790 | 0.0731 | 1778.6 | 182.1 | 41595 | 2389.9 | -330.0 | -92.8 |
| 7 | 1104880 | 0.0817 | 1124.2 | 120.6 | 37569 | 2159.7 | -329.9 | -90.2 |
| 8 | 1030170 | 0.0744 | 2190.1 | 201.8 | 37766 | 2193.5 | -328.0 | -87.8 |
| 9 | 1051220 | 0.0865 | 1500.6 | 143.0 | 33949 | 1973.6 | -328.8 | -85.6 |
| 10 | 1053397 | 0.0580 | 1977.3 | 171.0 | 34344 | 2004.4 | -327.3 | -85.7 |

No monotonicity requirement is imposed. The observed values show no
non-finite or runaway behavior over this short fixed-data window.

### Stability Audit

| Check | Result |
| :--- | :--- |
| 10/10 native updates complete | `PASS` |
| All scalar metrics finite | `PASS` |
| F / B / D / QD / Qaux / Actor changed | `PASS` |
| Target F / B / QD / Qaux changed | `PASS` |
| Six optimizer steps | `10` each |
| Observation normalizer finite | `PASS` at every update |
| Auxiliary reward normalizer finite | `PASS` at every update |
| z-buffer | `1024 -> 8192`, capacity `8192` |
| Model parameters finite | `PASS` |
| Numerical stability | `PASS` |

The ignored machine-readable result is
`results/m2.4d-2-short-stability/summary.json`. No checkpoint was saved and no
skating performance was evaluated.

Next milestone: `M2.4d-3 — 100-Update Stability Smoke`.

## M2.4d-3 100-Update Stability Smoke

This is a fixed-replay stability diagnostic, not a convergence or control
performance experiment.

- Date: `2026-08-12`
- Checkpoint SHA256:
  `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- Replay: 1024 transitions, 14 terminated, 1 truncated, 1009 normal;
  `train is train_skate`: `YES`.
- Expert mixture: 64 Base plus 64 Skate complete sequences per update.
- Native updates: `100 / 100`; no project-owned direct component update,
  optimizer, or target-soft-update call.

### Core Metric Summary

| Metric | First | Min | Max | Last |
| :--- | ---: | ---: | ---: | ---: |
| `fb_loss` | 1283845 | 277162 | 1307751 | 286643 |
| `disc_loss` | 0.4494 | 0.0119 | 0.4494 | 0.0125 |
| `critic_loss` | 1796.0 | 29.2 | 2607.9 | 39.6 |
| `aux_critic_loss` | 248.9 | 28.0 | 279.1 | 53.5 |
| `actor_loss` | 50068 | 20833 | 50068 | 22388 |
| `Q_fb` | 2837.2 | 1238.8 | 2842.5 | 1490.2 |
| `Q_discriminator` | -332.2 | -334.2 | -289.5 | -289.5 |
| `Q_aux` | -101.3 | -101.3 | -67.9 | -67.9 |

`B_norm` and `z_norm` remained exactly `16.0`. QD scale remained finite:
`target_Q` ended at `-305.3`, `Q1` at `-306.0`, and `unc_Q` at `1.28`.
Qaux scale remained finite: `target_auxQ` ended at `-73.7`, `auxQ1` at
`-74.1`, and `unc_auxQ` at `1.76`.

| Stability audit | Result |
| :--- | :--- |
| All returned scalar metrics finite | `PASS` |
| Monitored 100x scale warning | `NONE` |
| F / B / D / QD / Qaux / Actor changed | `PASS` |
| target F / B / QD / Qaux changed | `PASS` |
| Six optimizer steps | `100` each |
| Optimizer state finite | `PASS` |
| Observation normalizer finite | `PASS` at all 100 updates |
| Auxiliary reward normalizer finite | `PASS` at all 100 updates |
| z-buffer | `0 -> 8192`, capacity `8192` |
| Numerical stability | `PASS` |

The ignored machine-readable result is
`results/m2.4d-3-100-update-stability/summary.json`. No smoke checkpoint was
saved. M2.5 must begin from a fresh official BFM0 checkpoint, not this smoke
state.

M2.4 Training Preparation: `COMPLETE`.

Next milestone: `M2.5 — Original BFM-Zero Skate Baseline`.
