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
| human_push_1_window_00 | 0 | n/a | n/a | -0.496919 | -0.251098 | 42.46 deg | 0.950 | 1.000 | locally_covered |
| human_push_1_window_01 | 0 | n/a | n/a | -0.567508 | -0.475055 | 41.20 deg | 0.400 | 1.000 | fragile |
| human_push_1_window_02 | 0 | n/a | n/a | -0.541270 | -0.367334 | 36.78 deg | 0.750 | 0.938 | locally_covered |
| human_push_2_window_00 | 0 | n/a | n/a | -0.568968 | -0.522360 | 40.29 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_01 | 0 | n/a | n/a | -0.571243 | -0.504313 | 39.05 deg | 0.150 | 0.000 | not_covered |
| human_push_2_window_02 | 0 | n/a | n/a | -0.359733 | -0.292182 | 44.72 deg | 0.550 | 1.000 | fragile |

## Limitations

- Static expert velocities are set to zero when constructing backward observations.
- Dynamic expert velocities are reconstructed by finite differences at 50 Hz.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- Human-push files do not include synchronized skateboard state; initialization aligns the expert feet to the HUSKY deck while preserving expert pose and velocity.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
- Without-prior score-angle plots use the searched constant-latent CEM anchor as their reference.
