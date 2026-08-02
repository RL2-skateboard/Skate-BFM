# H1 Frozen BFM0 Motion Coverage

- Experiment: `h1_bfm0_motion_with_prior`
- Run type: `formal`
- Prior mode: `with_prior`
- Checkpoint: `model/bfm-zero-official`
- Result directory: `docs/res/h1_bfm0_motion_with_prior`

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
| human_push_1_window_00 | 24 | -0.530916 | 0.000 | -0.506456 | -0.500381 | 13.96 deg | 0.000 | 0.000 | not_covered |
| human_push_1_window_01 | 24 | -0.571654 | 0.000 | -0.556736 | -0.530974 | 12.17 deg | 0.000 | 0.000 | not_covered |
| human_push_1_window_02 | 24 | -0.568220 | 0.000 | -0.526268 | -0.519531 | 13.37 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_00 | 24 | -0.586573 | 0.000 | -0.541474 | -0.534871 | 13.15 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_01 | 24 | -0.565684 | 0.000 | -0.566813 | -0.517439 | 13.42 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_02 | 24 | -0.546455 | 0.000 | -0.475021 | -0.508025 | 13.18 deg | 0.000 | 0.000 | not_covered |

## Limitations

- Static expert velocities are set to zero when constructing backward observations.
- Dynamic expert velocities are reconstructed by finite differences at 50 Hz.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- Human-push files do not include synchronized skateboard state; initialization aligns the right support foot to the push-start deck reference and preserves the left push foot's expert-relative position and velocity.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
- Dynamic score-angle plots use trajectory midpoints for display; CEM constraints and reported maximum angles use every original latent step.
