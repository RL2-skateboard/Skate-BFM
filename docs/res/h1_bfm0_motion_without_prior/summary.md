# H1 Frozen BFM0 Motion Coverage

- Experiment: `h1_bfm0_motion_without_prior`
- Run type: `formal`
- Prior mode: `without_prior`
- Checkpoint: `model/bfm-zero-official`
- Result directory: `docs/res/h1_bfm0_motion_without_prior`

## Dataset status

| Dataset | Shape | Scoring enabled | BFM input |
|---|---:|---:|---:|
| push_start_pose | `[30, 7]` | True | False |
| steer_start_pose | `[30, 7]` | True | False |
| human_push_1 | `[180, 36]` | True | False |
| human_push_2 | `[221, 36]` | True | False |

## Coverage results

| Expert target | Latent steps | Encoded score | Encoded robust | Global best | CEM best | CEM max angle | CEM robust | Angular support | Coverage type |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| push_start_pose | 0 | n/a | n/a | -0.662178 | -0.523622 | 41.24 deg | 0.000 | 0.000 | not_covered |
| steer_start_pose | 0 | n/a | n/a | -0.368672 | -0.287402 | 40.77 deg | 0.000 | 0.000 | fragile |
| human_push_1_window_00 | 0 | n/a | n/a | -0.604842 | -0.526947 | 33.94 deg | 0.000 | 0.000 | not_covered |
| human_push_1_window_01 | 0 | n/a | n/a | -0.592723 | -0.524287 | 43.12 deg | 0.000 | 0.000 | not_covered |
| human_push_1_window_02 | 0 | n/a | n/a | -0.584074 | -0.323566 | 37.15 deg | 0.850 | 0.938 | locally_covered |
| human_push_2_window_00 | 0 | n/a | n/a | -0.575069 | -0.465430 | 37.27 deg | 0.350 | 0.875 | fragile |
| human_push_2_window_01 | 0 | n/a | n/a | -0.615353 | -0.360626 | 37.13 deg | 0.950 | 1.000 | locally_covered |
| human_push_2_window_02 | 0 | n/a | n/a | -0.572806 | -0.311128 | 34.20 deg | 1.000 | 1.000 | locally_covered |

## Limitations

- Static expert velocities are set to zero when constructing backward observations.
- Dynamic expert velocities are reconstructed by finite differences at 50 Hz.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
- Without-prior score-angle plots use the searched constant-latent CEM anchor as their reference.
