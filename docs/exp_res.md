# Experiment Results

# H1 Expert-Guided BFM0 Behavior Coverage

## h1_bfm0_motion_without_prior

### Experiment metadata

- Experiment name: h1_bfm0_motion_without_prior
- Experiment type: H1 Expert-Guided BFM0 Behavior Coverage
- Study date: 2026-07-31
- Prior mode: `without_prior`
- Start time: 2026-08-02T09:36:45.467744+08:00
- End time: 2026-08-02T09:45:11.335094+08:00
- Duration: 505.840 seconds
- Git commit: `6d0adcdcd87d90d6890514fb9c0c2f86d9208e56`
- Checkpoint: `model/bfm-zero-official`
- Device: `cuda`
- Result directory: `docs/res/h1_bfm0_motion_without_prior`

### Expert dataset status

| Dataset | Format confirmed | Used as BFM input | Used as search target | Limitation |
|---|---:|---:|---:|---|
| push_start_pose | true | False | True | Used only for evaluation; excluded from latent proposal generation |
| steer_start_pose | true | False | True | Used only for evaluation; excluded from latent proposal generation |
| human_push_1 | true | False | True | Used only for evaluation; excluded from latent proposal generation |
| human_push_2 | true | False | True | Used only for evaluation; excluded from latent proposal generation |

### Configuration

- Prior mode: without_prior
- Global sphere samples: 256
- CEM population: 64
- CEM iterations: 6
- Horizon: 0.5
- Seeds: [0, 1, 2]
- Robust trials: 20
- Geodesic angles: [5, 10, 20, 40, 80]
- Samples per angle: 16
- Action gain: 1.0
- CEM maximum angle from search reference: 180.0 degrees
- CEM temporal noise correlation: 0.0
- Static rollouts start from their reconstructed expert pose
- Human-push rollouts start from the push expert pose because the motion files do not include skateboard state

### Main results

| Expert target | Latent steps | Encoded score | Encoded robust | Global best | CEM best | CEM max angle | CEM robust | Coverage type |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| push_start_pose | 0 | n/a | n/a | -0.662178 | -0.523622 | 41.24 deg | 0.000 | not_covered |
| steer_start_pose | 0 | n/a | n/a | -0.368672 | -0.287402 | 40.77 deg | 0.000 | fragile |
| human_push_1_window_00 | 0 | n/a | n/a | -0.604842 | -0.526947 | 33.94 deg | 0.000 | not_covered |
| human_push_1_window_01 | 0 | n/a | n/a | -0.592723 | -0.524287 | 43.12 deg | 0.000 | not_covered |
| human_push_1_window_02 | 0 | n/a | n/a | -0.584074 | -0.323566 | 37.15 deg | 0.850 | locally_covered |
| human_push_2_window_00 | 0 | n/a | n/a | -0.575069 | -0.465430 | 37.27 deg | 0.350 | fragile |
| human_push_2_window_01 | 0 | n/a | n/a | -0.615353 | -0.360626 | 37.13 deg | 0.950 | locally_covered |
| human_push_2_window_02 | 0 | n/a | n/a | -0.572806 | -0.311128 | 34.20 deg | 1.000 | locally_covered |

### Latent-space results

| Metric | Result |
|---|---:|
| Push-steer angular distance (degrees) | 90.02168273925781 |
| Global stable proposal rate | 1.0 |
| Latent-behavior Spearman correlation | 0.038124072015193224 |

### Main findings

No expert backward-map latent is used for proposal generation. Expert poses and motions are used only as common evaluation targets and rollout initial conditions.

Broad CEM improved the global-best score for 8/8 targets. Its distance from the global-best initialization ranged 33.94-43.12 degrees.

The search first evaluates uniformly sampled constant latent directions on the frozen BFM0 sphere, then refines each target from its global best.

5/8 targets met the formal coverage criteria. Coverage failure means the selected frozen-BFM0 behavior was not maintained under the adapted actor, coupled skateboard physics, and current thresholds.

The per-target classifications above require the final static pose or full dynamic window to meet its threshold; a transient intermediate pose is not counted as success.

### Latent-space visualizations

![2D latent t-SNE](res/h1_bfm0_motion_without_prior/plots/latent_tsne_2d.png)

![Qualitative latent sphere t-SNE](res/h1_bfm0_motion_without_prior/plots/latent_sphere_tsne.png)

![Score versus geodesic angle](res/h1_bfm0_motion_without_prior/plots/score_vs_geodesic_angle.png)

![Geodesic support curve](res/h1_bfm0_motion_without_prior/plots/geodesic_support_curve.png)

### Videos

- [Push pose: global best](res/h1_bfm0_motion_without_prior/videos/push_pose_global_best.mp4)
- [Steer pose: global best](res/h1_bfm0_motion_without_prior/videos/steer_pose_global_best.mp4)
- [Push pose: CEM best](res/h1_bfm0_motion_without_prior/videos/push_pose_cem_best.mp4)
- [Steer pose: CEM best](res/h1_bfm0_motion_without_prior/videos/steer_pose_cem_best.mp4)
- [Human push 1 window 00: CEM best](res/h1_bfm0_motion_without_prior/videos/human_push_1_window_00_cem_best.mp4)
- [Push-steer midpoint](res/h1_bfm0_motion_without_prior/videos/push_steer_midpoint_blend.mp4)
- [All generated videos](res/h1_bfm0_motion_without_prior/videos/)

### Limitations

- Static expert velocities are set to zero when constructing backward observations.
- Dynamic expert velocities are reconstructed by finite differences at 50 Hz.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
- Dynamic score-angle plots use trajectory midpoints for display; CEM constraints and reported maximum angles use every original latent step.
