# H1 Expert-Guided BFM0 Behavior Coverage

- Experiment: `h1_bfm_coverage_smoke_20260801_114246`
- Run type: `smoke`
- Checkpoint: `/tmp/h1_bfm0_temporary_smoke.pt`
- Result directory: `docs/res/h1_bfm_coverage_smoke_20260801_114246`

> Smoke-only pipeline validation with a temporary checkpoint. The values below are not scientific results.

## Dataset status

| Dataset | Shape | Scoring enabled | BFM input |
|---|---:|---:|---:|
| push_start_pose | `[30, 7]` | True | false |
| steer_start_pose | `[30, 7]` | False | false |
| human_push_1 | `[180, 36]` | False | false |
| human_push_2 | `[221, 36]` | False | false |

## Coverage results

| Expert target | Encoded anchor | Global best | CEM best | Robust success | Angular support | Coverage type |
|---|---:|---:|---:|---:|---:|---|
| push_start_pose | false | 0.003848 | 0.003848 | 1.000 | 1.000 | naturally_covered |

## Limitations

- This run uses a temporary random checkpoint and supports pipeline validation only.
- Expert arrays do not provide complete BFM0 observations; encoded anchors are disabled.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
