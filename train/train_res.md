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
