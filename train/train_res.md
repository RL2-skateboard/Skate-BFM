# Training Results

## Experiment 0: Original FB + Skate, B/F-only Adaptation

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
- Expert source: Base + Skate
- Skate expert ratio: `0.5`
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
