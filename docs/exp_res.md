# Experiment Results

# H1 Expert-Guided BFM0 Behavior Coverage

## h1_bfm_coverage_bfmzero_official

### Experiment metadata

- Experiment name: h1_bfm_coverage_bfmzero_official
- Experiment type: H1 Expert-Guided BFM0 Behavior Coverage
- Start time: 2026-08-01T22:16:22.940662+08:00
- End time: 2026-08-01T22:24:48.596577+08:00
- Duration: 505.630 seconds
- Git commit: `486cf93c9e2e278a088e54d91275394d8c770eb1`
- Checkpoint: `model/bfm-zero-official`
- Device: `cuda`
- Result directory: `docs/res/h1_bfm_coverage_bfmzero_official`

### Expert dataset status

| Dataset | Format confirmed | Used as BFM input | Used as search target | Limitation |
|---|---:|---:|---:|---|
| push_start_pose | true | True | True | Official backward observation reconstructed from confirmed fields |
| steer_start_pose | true | True | True | Official backward observation reconstructed from confirmed fields |
| human_push_1 | true | True | True | Official backward observation reconstructed from confirmed fields |
| human_push_2 | true | True | True | Official backward observation reconstructed from confirmed fields |

### Configuration

- Global sphere samples: 256
- CEM population: 64
- CEM iterations: 6
- Horizon: 0.5
- Seeds: [0, 1, 2]
- Robust trials: 20
- Geodesic angles: [5, 10, 20, 40, 80]
- Samples per angle: 16
- Action gain: 1.0
- CEM maximum per-step angle from encoded expert latent: 40.0 degrees
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

Trajectory-local CEM improved the encoded score for 8/8 targets. Selected CEM latents remained at most 12.10-14.15 degrees from their matching expert latent at every step.

0/8 targets met the formal coverage criteria. A failed rollout does not imply failed expert encoding; it means the encoded behavior was not maintained under the adapted actor, coupled skateboard physics, and current thresholds.

The per-target classifications above require the final static pose or full dynamic window to meet its threshold; a transient intermediate pose is not counted as success.

Random constant-latent sampling is reported only as a baseline. Dynamic CEM starts from the complete encoded expert trajectory and uses time-correlated perturbations under a per-step angular cap.

### Latent-space visualizations

![2D latent t-SNE](res/h1_bfm_coverage_bfmzero_official/plots/latent_tsne_2d.png)

![Qualitative latent sphere t-SNE](res/h1_bfm_coverage_bfmzero_official/plots/latent_sphere_tsne.png)

![Score versus geodesic angle](res/h1_bfm_coverage_bfmzero_official/plots/score_vs_geodesic_angle.png)

![Geodesic support curve](res/h1_bfm_coverage_bfmzero_official/plots/geodesic_support_curve.png)

### Videos

- [Push pose: encoded expert anchor](res/h1_bfm_coverage_bfmzero_official/videos/push_pose_encoded_anchor.mp4)
- [Push pose: CEM best](res/h1_bfm_coverage_bfmzero_official/videos/push_pose_cem_best.mp4)
- [Steer pose: encoded expert anchor](res/h1_bfm_coverage_bfmzero_official/videos/steer_pose_encoded_anchor.mp4)
- [Steer pose: CEM best](res/h1_bfm_coverage_bfmzero_official/videos/steer_pose_cem_best.mp4)
- [Push-steer midpoint](res/h1_bfm_coverage_bfmzero_official/videos/push_steer_midpoint_blend.mp4)
- [All generated videos](res/h1_bfm_coverage_bfmzero_official/videos/)

### Limitations

- Static expert velocities are set to zero when constructing backward observations.
- Dynamic expert velocities are reconstructed by finite differences at 50 Hz.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
- Dynamic score-angle plots use trajectory midpoints for display; CEM constraints and reported maximum angles use every original latent step.
