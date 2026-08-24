# Phase100k / Continuous100k Fixed Val Comparison

- Split: Phase-Val + canonical Val Raw; this is model selection, not Test generalization.
- Cases: 80 exact-replay scenarios, 20 per behavior and 20 unique source rollouts per behavior.
- Selection: rollout-balanced without replacement, seed 4728.
- Case identity SHA256: `e6101d4fcd7d9f55a77cf6334c493123a3ce6b22dc2bcb36962632136094f9bc`.
- Frozen inference: deterministic Actor mean; no training, backward, update, normalizer mutation, or Continuous dataset dependency.

## Task Metrics

| Behavior | Model | Completion | Full | Planar vel. | Speed | Vel. dir. | Dir. valid | Heading | Final heading | Final XY | Feet | Off-board streak | Coupling XY | Tilt p95/max | Joint MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | Phase | 0.813 | 0.400 | 0.371 | 0.310 | 8.89 | 0.995 | 6.20 | 16.19 | 0.274 | 0.808 | 0.227 | 0.196 | 71.65 / 78.39 | 0.528 |
| push | Continuous | 0.679 | 0.300 | 0.346 | 0.271 | 13.83 | 0.987 | 5.99 | 10.18 | 0.278 | 0.773 | 0.154 | 0.152 | 74.14 / 81.03 | 0.535 |
| steer | Phase | 0.780 | 0.500 | 0.499 | 0.307 | 12.72 | 1.000 | 10.95 | 22.31 | 0.480 | 0.864 | 0.095 | 0.169 | 58.14 / 62.80 | 0.472 |
| steer | Continuous | 0.744 | 0.500 | 0.299 | 0.175 | 8.55 | 0.996 | 6.91 | 13.78 | 0.283 | 0.909 | 0.063 | 0.116 | 49.38 / 51.77 | 0.423 |
| push2steer | Phase | 0.291 | 0.000 | 0.670 | 0.463 | 25.36 | 0.990 | 3.16 | 8.54 | 0.758 | 0.854 | 0.284 | 0.214 | 82.93 / 94.98 | 0.529 |
| push2steer | Continuous | 0.253 | 0.000 | 0.636 | 0.458 | 30.38 | 0.992 | 7.81 | 13.64 | 0.735 | 0.930 | 0.071 | 0.194 | 83.24 / 86.92 | 0.521 |
| steer2push | Phase | 0.449 | 0.000 | 0.496 | 0.336 | 21.71 | 0.990 | 15.10 | 31.87 | 0.797 | 0.840 | 0.335 | 0.256 | 80.55 / 90.06 | 0.463 |
| steer2push | Continuous | 0.362 | 0.000 | 0.388 | 0.232 | 17.92 | 0.997 | 8.66 | 21.26 | 0.544 | 0.927 | 0.108 | 0.153 | 84.74 / 91.49 | 0.435 |

**Caption.** Completion is executed/planned steps and full is horizon-completion fraction (higher is better). Planar velocity and speed errors are m/s; velocity direction, heading, final heading, and root tilt are degrees; final XY and coupling are metres; off-board streak is seconds; joint MAE is radians. All errors/streak/tilt are lower-is-better. Direction is summarized only when both speeds exceed 0.05 m/s, while `Dir. valid` reports the valid executed-frame fraction.

`Feet` is the fraction of executed frames where at least one left/right foot
collision geom contacts the deck. Brief two-foot contact gaps affect this
ratio and the off-board streak but do not independently trigger termination.

## Paired Direction

| Behavior | Continuous advantages | Phase advantages | Incomplete evidence |
|---|---|---|---|
| push | Lower speed and coupling means; shorter off-board streak | Higher completion/retention and lower velocity-direction error | Continuous/Phase terminate in 70%/60% of cases |
| steer | Lower board errors, coupling, tilt, and joint MAE on most paired cases | Slightly higher mean completion | Both complete only 50% of cases |
| push2steer | Better retention and shorter off-board streak | Better direction/heading and slightly higher completion | Both terminate in all 20 cases |
| steer2push | Better board trajectory, retention, coupling, and joint MAE | Higher completion and lower tilt p95 | Both terminate in all 20 cases |

Continuous100k is not globally better. It improves the board/coupling vector for steer and steer2push, but lowers completion and does not complete either transition category. Phase100k also fails both transition categories. Completion, board tracking, retention, stability, and control must therefore be judged jointly.

## Proposed Task Success Thresholds

`PROPOSED ONLY`, not frozen in evaluator code:

| Quantity | Proposed per-case threshold | Physical interpretation |
|---|---:|---|
| Horizon | no termination and completion = 1.0 | The complete 2 s steady or 5 s transition task must execute. |
| Mean board speed error | <= 0.30 m/s | Keep scalar board speed near the same-command expert trajectory. |
| Mean board velocity direction error | <= 15 deg with valid fraction >= 0.95 | Track travel direction on nearly all physically meaningful frames. |
| Mean board heading error | <= 15 deg | Keep board yaw near the expert turn trajectory. |
| Final board XY error | <= 0.50 m | End the executed horizon near the expert displacement. |
| Feet-on-board ratio | >= 0.85 | Permit brief contact loss without accepting persistent separation. |
| Longest off-board streak | <= 0.20 s (10 steps) | Reject sustained loss of board contact while tolerating short transients. |
| Mean coupling XY error | <= 0.20 m | Keep robot-root/board relative XY close to expert coupling. |
| Root tilt p95 | <= 60 deg | Reject near-fall posture over the episode tail. |
| Hard action saturation | <= 5% | Reject policies relying persistently on clipped normalized actions. |

These values use expert/Val physical scale and current model distributions, but need human review with representative videos. Binary success remains `NOT_YET_CALIBRATED` until approved and frozen in a later task.

## Artifacts

- Phase: [`../../2026-08-20/p100k-v20b_val_phase_eval/summary.md`](../../2026-08-20/p100k-v20b_val_phase_eval/summary.md)
- Continuous: [`../c100k-v20b_val_phase_eval/summary.md`](../c100k-v20b_val_phase_eval/summary.md)
- Machine-readable comparison: [`comparison.json`](comparison.json)

Both evaluation directories contain full logs, `cases.json`, `results.jsonl`, summaries, and representative model/expert videos for all four behaviors.
