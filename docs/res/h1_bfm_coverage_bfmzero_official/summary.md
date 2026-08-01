# H1 Expert-Guided BFM0 Behavior Coverage

- Experiment: `h1_bfm_coverage_bfmzero_official`
- Run type: `formal`
- Checkpoint: `model/bfm-zero-official`
- Result directory: `docs/res/h1_bfm_coverage_bfmzero_official`

## Dataset status

| Dataset | Shape | Scoring enabled | BFM input |
|---|---:|---:|---:|
| push_start_pose | `[30, 7]` | True | True |
| steer_start_pose | `[30, 7]` | True | True |
| human_push_1 | `[180, 36]` | True | True |
| human_push_2 | `[221, 36]` | True | True |

## Coverage results

| Expert target | Encoded score | Encoded robust | Global best | CEM best | CEM angle | CEM robust | Angular support | Coverage type |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| push_start_pose | -1.138241 | 0.000 | -1.030625 | -1.045835 | 12.37 deg | 0.000 | 0.000 | not_covered |
| steer_start_pose | -1.098281 | 0.000 | -0.847677 | -0.998107 | 12.83 deg | 0.000 | 0.000 | not_covered |
| human_push_1_window_00 | -0.601383 | 0.000 | -0.604842 | -0.553971 | 12.37 deg | 0.000 | 0.000 | not_covered |
| human_push_1_window_01 | -0.592272 | 0.000 | -0.592723 | -0.552415 | 12.50 deg | 0.000 | 0.000 | not_covered |
| human_push_1_window_02 | -0.565941 | 0.000 | -0.584074 | -0.531432 | 11.78 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_00 | -0.574201 | 0.000 | -0.575069 | -0.549567 | 11.48 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_01 | -0.584426 | 0.000 | -0.615353 | -0.557333 | 11.39 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_02 | -0.566790 | 0.000 | -0.572806 | -0.533452 | 12.36 deg | 0.000 | 0.000 | not_covered |

## Limitations

- Static expert velocities are set to zero when constructing backward observations.
- Dynamic expert velocities are reconstructed by finite differences at 50 Hz.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
