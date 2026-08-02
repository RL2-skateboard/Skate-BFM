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
| human_push_1_window_00 | 0 | n/a | n/a | -0.506456 | -0.235852 | 40.79 deg | 0.750 | 1.000 | locally_covered |
| human_push_1_window_01 | 0 | n/a | n/a | -0.556736 | -0.479196 | 35.70 deg | 0.750 | 1.000 | locally_covered |
| human_push_1_window_02 | 0 | n/a | n/a | -0.526268 | -0.393350 | 37.52 deg | 0.850 | 1.000 | locally_covered |
| human_push_2_window_00 | 0 | n/a | n/a | -0.541474 | -0.315968 | 40.22 deg | 0.750 | 0.938 | locally_covered |
| human_push_2_window_01 | 0 | n/a | n/a | -0.566813 | -0.471406 | 37.95 deg | 0.900 | 1.000 | locally_covered |
| human_push_2_window_02 | 0 | n/a | n/a | -0.475021 | -0.301938 | 40.15 deg | 0.900 | 1.000 | locally_covered |

## Limitations

- Static expert velocities are set to zero when constructing backward observations.
- Dynamic expert velocities are reconstructed by finite differences at 50 Hz.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- Human-push files do not include synchronized skateboard state; initialization aligns the right support foot to the push-start deck reference and preserves the left push foot's expert-relative position and velocity.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
- Without-prior score-angle plots use the searched constant-latent CEM anchor as their reference.
