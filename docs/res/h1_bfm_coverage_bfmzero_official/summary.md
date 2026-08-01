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

| Expert target | Latent steps | Encoded score | Encoded robust | Global best | CEM best | CEM max angle | CEM robust | Angular support | Coverage type |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| push_start_pose | 1 | -0.727388 | 0.000 | -0.662178 | -0.687604 | 13.85 deg | 0.000 | 0.000 | not_covered |
| steer_start_pose | 1 | -0.608622 | 0.000 | -0.368672 | -0.499037 | 14.15 deg | 0.000 | 0.000 | not_covered |
| human_push_1_window_00 | 24 | -0.590235 | 0.000 | -0.604842 | -0.560348 | 13.17 deg | 0.000 | 0.000 | not_covered |
| human_push_1_window_01 | 24 | -0.565269 | 0.000 | -0.592723 | -0.541290 | 12.10 deg | 0.000 | 0.000 | not_covered |
| human_push_1_window_02 | 24 | -0.575500 | 0.000 | -0.584074 | -0.546026 | 13.40 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_00 | 24 | -0.581676 | 0.000 | -0.575069 | -0.543916 | 13.47 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_01 | 24 | -0.590527 | 0.000 | -0.615353 | -0.561113 | 14.04 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_02 | 24 | -0.560243 | 0.000 | -0.572806 | -0.529089 | 13.09 deg | 0.000 | 0.000 | not_covered |

## Limitations

- Static expert velocities are set to zero when constructing backward observations.
- Dynamic expert velocities are reconstructed by finite differences at 50 Hz.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
- Dynamic score-angle plots use trajectory midpoints for display; CEM constraints and reported maximum angles use every original latent step.
