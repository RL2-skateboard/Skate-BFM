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
| human_push_1_window_00 | 24 | -0.532793 | 0.000 | -0.496919 | -0.497779 | 13.80 deg | 0.250 | 0.000 | fragile |
| human_push_1_window_01 | 24 | -0.576890 | 0.000 | -0.567508 | -0.518339 | 13.09 deg | 0.000 | 0.000 | not_covered |
| human_push_1_window_02 | 24 | -0.579753 | 0.000 | -0.541270 | -0.527105 | 14.00 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_00 | 24 | -0.583071 | 0.000 | -0.568968 | -0.547325 | 12.15 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_01 | 24 | -0.561036 | 0.000 | -0.571243 | -0.524113 | 13.42 deg | 0.000 | 0.000 | not_covered |
| human_push_2_window_02 | 24 | -0.557242 | 0.000 | -0.359733 | -0.479503 | 14.74 deg | 0.850 | 0.000 | locally_covered |

## Limitations

- Static expert velocities are set to zero when constructing backward observations.
- Dynamic expert velocities are reconstructed by finite differences at 50 Hz.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- Human-push files do not include synchronized skateboard state; initialization aligns the expert feet to the HUSKY deck while preserving expert pose and velocity.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
- Dynamic score-angle plots use trajectory midpoints for display; CEM constraints and reported maximum angles use every original latent step.
