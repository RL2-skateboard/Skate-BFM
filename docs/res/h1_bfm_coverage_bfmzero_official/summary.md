# H1 Expert-Guided BFM0 Behavior Coverage

- Experiment: `h1_bfm_coverage_bfmzero_official`
- Run type: `formal`
- Checkpoint: `model/bfm-zero-official`
- Result directory: `docs/res/h1_bfm_coverage_bfmzero_official`

## Dataset status

| Dataset | Shape | Scoring enabled | BFM input |
|---|---:|---:|---:|
| push_start_pose | `[30, 7]` | True | false |
| steer_start_pose | `[30, 7]` | True | false |
| human_push_1 | `[180, 36]` | True | false |
| human_push_2 | `[221, 36]` | True | false |

## Coverage results

| Expert target | Encoded anchor | Global best | CEM best | Robust success | Angular support | Coverage type |
|---|---:|---:|---:|---:|---:|---|
| push_start_pose | false | -0.154998 | 0.017661 | 0.000 | 0.000 | fragile |
| steer_start_pose | false | -0.914587 | -0.553239 | 0.000 | 0.000 | not_covered |
| human_push_1_window_00 | false | -0.390123 | -0.311494 | 0.800 | 1.000 | locally_covered |
| human_push_1_window_01 | false | -0.315596 | -0.315596 | 0.500 | 1.000 | fragile |
| human_push_1_window_02 | false | -0.321111 | -0.321111 | 0.650 | 1.000 | fragile |
| human_push_2_window_00 | false | -0.347950 | -0.317644 | 0.950 | 1.000 | locally_covered |
| human_push_2_window_01 | false | -0.366666 | -0.324695 | 0.650 | 1.000 | fragile |
| human_push_2_window_02 | false | -0.340893 | -0.304406 | 0.900 | 1.000 | locally_covered |

## Limitations

- Expert arrays do not provide complete BFM0 observations; encoded anchors are disabled.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
