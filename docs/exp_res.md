# Experiment Results

## Comparison

| Study date | Experiment | Latent proposal | Stable coverage | Main result |
|---|---|---|---:|---|
| 2026-07-31 | `h1_bfm0_motion_without_prior` | 256 global sphere samples, then broad CEM from global best | 3/8 | Frozen BFM0 contains robust constant-latent behaviors for three dynamic windows |
| 2026-08-01 | `h1_bfm0_motion_with_prior` | Time-aligned expert `z_t`, then trajectory-local CEM | 0/8 | Expert trajectories produce clear time-varying motion but are not robustly maintained in HUSKY |

The without-prior run found three robust local regions, with success rates
`0.85`, `0.95`, and `1.00`. The with-prior run preserved the official
next-frame latent sequence and remained within `12.10-14.15` degrees of its
expert reference, but every target had zero robust success.

This comparison measures frozen BFM0 motion capacity under the current HUSKY
adapter. It does not establish that the without-prior motions semantically
match human pushing: dynamic scoring currently uses only the common 23 joint
positions, without root trajectory, foot contact, or synchronized skateboard
state.

# H1 Frozen BFM0 Motion Coverage

## h1_bfm0_motion_without_prior

### Experiment metadata

- Experiment name: h1_bfm0_motion_without_prior
- Experiment type: H1 Frozen BFM0 Motion Coverage
- Study date: 2026-07-31
- Prior mode: `without_prior`
- Start time: 2026-08-02T09:48:03.669309+08:00
- End time: 2026-08-02T09:56:34.888070+08:00
- Duration: 511.193 seconds
- Git commit: `4609369cfc149727542e065610fb5805b5c4c3de`
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

3/8 targets met the formal coverage criteria. Coverage failure means the selected frozen-BFM0 behavior was not maintained under the adapted actor, coupled skateboard physics, and current thresholds.

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
- Without-prior score-angle plots use the searched constant-latent CEM anchor as their reference.

# H1 Frozen BFM0 Motion Coverage

## h1_bfm0_motion_with_prior

### Experiment metadata

- Experiment name: h1_bfm0_motion_with_prior
- Experiment type: H1 Frozen BFM0 Motion Coverage
- Study date: 2026-08-01
- Prior mode: `with_prior`
- Start time: 2026-08-02T10:02:37.178675+08:00
- End time: 2026-08-02T10:11:12.593365+08:00
- Duration: 515.387 seconds
- Git commit: `cfe70c78d05833c1d1d113c2f32921310c682f84`
- Checkpoint: `model/bfm-zero-official`
- Device: `cuda`
- Result directory: `docs/res/h1_bfm0_motion_with_prior`

### Expert dataset status

| Dataset | Format confirmed | Used as BFM input | Used as search target | Limitation |
|---|---:|---:|---:|---|
| push_start_pose | true | True | True | Official backward observation reconstructed from confirmed fields |
| steer_start_pose | true | True | True | Official backward observation reconstructed from confirmed fields |
| human_push_1 | true | True | True | Official backward observation reconstructed from confirmed fields |
| human_push_2 | true | True | True | Official backward observation reconstructed from confirmed fields |

### Configuration

- Prior mode: with_prior
- Global sphere samples: 256
- CEM population: 64
- CEM iterations: 6
- Horizon: 0.5
- Seeds: [0, 1, 2]
- Robust trials: 20
- Geodesic angles: [5, 10, 20, 40, 80]
- Samples per angle: 16
- Action gain: 1.0
- CEM maximum angle from search reference: 40.0 degrees
- CEM temporal noise correlation: 0.9
- Static rollouts start from their reconstructed expert pose
- Human-push rollouts start from the push expert pose because the motion files do not include skateboard state

### Main results

| Expert target | Latent steps | Encoded score | Encoded robust | Global best | CEM best | CEM max angle | CEM robust | Coverage type |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| push_start_pose | 1 | -0.727388 | 0.000 | -0.662178 | -0.687604 | 13.85 deg | 0.000 | not_covered |
| steer_start_pose | 1 | -0.608622 | 0.000 | -0.368672 | -0.499037 | 14.15 deg | 0.000 | not_covered |
| human_push_1_window_00 | 24 | -0.590235 | 0.000 | -0.604842 | -0.560348 | 13.17 deg | 0.000 | not_covered |
| human_push_1_window_01 | 24 | -0.565269 | 0.000 | -0.592723 | -0.541290 | 12.10 deg | 0.000 | not_covered |
| human_push_1_window_02 | 24 | -0.575500 | 0.000 | -0.584074 | -0.546026 | 13.40 deg | 0.000 | not_covered |
| human_push_2_window_00 | 24 | -0.581676 | 0.000 | -0.575069 | -0.543916 | 13.47 deg | 0.000 | not_covered |
| human_push_2_window_01 | 24 | -0.590527 | 0.000 | -0.615353 | -0.561113 | 14.04 deg | 0.000 | not_covered |
| human_push_2_window_02 | 24 | -0.560243 | 0.000 | -0.572806 | -0.529089 | 13.09 deg | 0.000 | not_covered |

### Latent-space results

| Metric | Result |
|---|---:|
| Push-steer angular distance (degrees) | 79.63737487792969 |
| Global stable proposal rate | 1.0 |
| Latent-behavior Spearman correlation | 0.038124072015193224 |

### Main findings

Static goal latents and time-aligned dynamic latent trajectories are produced by the frozen official backward map from reconstructed expert observations. Dynamic execution follows the official next-frame pattern `z_t = project_z(backward_map(obs[t + 1]))`.

Trajectory-local CEM improved the encoded score for 8/8 targets. The per-target maximum trajectory angle ranged 12.10-14.15 degrees.

Random constant-latent sampling is reported only as a baseline. Dynamic CEM starts from the complete encoded expert trajectory and uses time-correlated perturbations under a per-step angular cap.

0/8 targets met the formal coverage criteria. Coverage failure means the selected frozen-BFM0 behavior was not maintained under the adapted actor, coupled skateboard physics, and current thresholds.

The per-target classifications above require the final static pose or full dynamic window to meet its threshold; a transient intermediate pose is not counted as success.

### Latent-space visualizations

![2D latent t-SNE](res/h1_bfm0_motion_with_prior/plots/latent_tsne_2d.png)

![Qualitative latent sphere t-SNE](res/h1_bfm0_motion_with_prior/plots/latent_sphere_tsne.png)

![Score versus geodesic angle](res/h1_bfm0_motion_with_prior/plots/score_vs_geodesic_angle.png)

![Geodesic support curve](res/h1_bfm0_motion_with_prior/plots/geodesic_support_curve.png)

### Videos

- [Push pose: encoded expert goal](res/h1_bfm0_motion_with_prior/videos/push_pose_encoded_anchor.mp4)
- [Steer pose: encoded expert goal](res/h1_bfm0_motion_with_prior/videos/steer_pose_encoded_anchor.mp4)
- [Human push 1 window 00: encoded trajectory](res/h1_bfm0_motion_with_prior/videos/human_push_1_window_00_encoded_trajectory.mp4)
- [Push pose: CEM best](res/h1_bfm0_motion_with_prior/videos/push_pose_cem_best.mp4)
- [Steer pose: CEM best](res/h1_bfm0_motion_with_prior/videos/steer_pose_cem_best.mp4)
- [Human push 1 window 00: CEM best](res/h1_bfm0_motion_with_prior/videos/human_push_1_window_00_cem_best.mp4)
- [Push-steer midpoint](res/h1_bfm0_motion_with_prior/videos/push_steer_midpoint_blend.mp4)
- [All generated videos](res/h1_bfm0_motion_with_prior/videos/)

### Limitations

- Static expert velocities are set to zero when constructing backward observations.
- Dynamic expert velocities are reconstructed by finite differences at 50 Hz.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
- Dynamic score-angle plots use trajectory midpoints for display; CEM constraints and reported maximum angles use every original latent step.
