# Experiment Results

# H1 Expert-Guided BFM0 Behavior Coverage

## h1_bfm_coverage_bfmzero_official

### Experiment metadata

- Experiment name: h1_bfm_coverage_bfmzero_official
- Experiment type: H1 Expert-Guided BFM0 Behavior Coverage
- Start time: 2026-08-01T19:13:13.868933+08:00
- End time: 2026-08-01T19:21:54.773042+08:00
- Duration: 520.884 seconds
- Git commit: `32a4432b0d15d5f28ede46bbb2f61cb2e3121dd2`
- Checkpoint: `model/bfm-zero-official`
- Checkpoint SHA-256: `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`
- Device: `cuda`
- Result directory: `docs/res/h1_bfm_coverage_bfmzero_official`

### Expert dataset status

| Dataset | Format confirmed | Used as BFM input | Used as search target | Limitation |
|---|---:|---:|---:|---|
| push_start_pose | true | false | True | Incomplete BFM0 observation |
| steer_start_pose | true | false | True | Incomplete BFM0 observation |
| human_push_1 | true | false | True | Incomplete BFM0 observation |
| human_push_2 | true | false | True | Incomplete BFM0 observation |

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
- Push and human-push initial pose: pinned upstream push keyframe
- Steer initial pose: IK fit to the official steer pose with both feet on board
- Static-pose success criterion: final pose error at or below `0.5`

### Main results

| Expert target | Encoded anchor | Global best | CEM best | Robust success | Coverage type |
|---|---:|---:|---:|---:|---|
| push_start_pose | n/a | -0.154998 | 0.017661 | 0.000 | fragile |
| steer_start_pose | n/a | -0.914587 | -0.553239 | 0.000 | not_covered |
| human_push_1_window_00 | n/a | -0.390123 | -0.311494 | 0.800 | locally_covered |
| human_push_1_window_01 | n/a | -0.315596 | -0.315596 | 0.500 | fragile |
| human_push_1_window_02 | n/a | -0.321111 | -0.321111 | 0.650 | fragile |
| human_push_2_window_00 | n/a | -0.347950 | -0.317644 | 0.950 | locally_covered |
| human_push_2_window_01 | n/a | -0.366666 | -0.324695 | 0.650 | fragile |
| human_push_2_window_02 | n/a | -0.340893 | -0.304406 | 0.900 | locally_covered |

### Latent-space results

| Metric | Result |
|---|---:|
| Push-steer angular distance (degrees) | 86.69477844238281 |
| Push local support at 10 degrees | 0.000 |
| Push local support at 20 degrees | 0.000 |
| Steer local support at 10 degrees | 0.000 |
| Steer local support at 20 degrees | 0.000 |
| Global stable proposal rate | 0.994140625 |
| Latent-behavior Spearman correlation | 0.015096519531489674 |

### Main findings

- No encoded zero-shot conclusion is available because the expert files do not
  contain complete BFM0 observations. All anchors are search results.
- Push is not naturally covered under the final-pose criterion. CEM improves
  final pose error from `0.544354` to `0.371695`, but the result fails all 20
  initial-state perturbation trials and is therefore `fragile`.
- Steer begins from a valid two-feet-on-board pose with initial error `0.029762`.
  The best CEM latent reaches final error `0.583001`, above the `0.5` threshold,
  and is `not_covered`.
- CEM finds corresponding behavior for all six human-push windows. Three are
  `locally_covered`; three remain `fragile` under perturbation.
- Neither static anchor has successful geodesic support at 5 to 80 degrees. The
  dynamic windows have strong support near their searched anchors that narrows
  as angular distance increases.
- Push-steer SLERP has no falls, but its push and steer scores are non-monotonic.
  This run does not support a smooth push-to-steer interpolation claim.
- The `0.994141` stable global proposal rate shows that most short rollouts avoid
  a fall, but the static target failures show that stability alone does not
  imply useful skateboard behavior coverage.

### Latent-space visualizations

#### 2D latent t-SNE

![2D latent t-SNE](res/h1_bfm_coverage_bfmzero_official/plots/latent_tsne_2d.png)

#### Qualitative latent sphere t-SNE

![Qualitative latent sphere t-SNE](res/h1_bfm_coverage_bfmzero_official/plots/latent_sphere_tsne.png)

#### Score versus geodesic angle

![Score versus geodesic angle](res/h1_bfm_coverage_bfmzero_official/plots/score_vs_geodesic_angle.png)

#### Geodesic support curve

![Geodesic support curve](res/h1_bfm_coverage_bfmzero_official/plots/geodesic_support_curve.png)

### Videos

- [Push pose: global best](res/h1_bfm_coverage_bfmzero_official/videos/push_pose_global_best.mp4)
- [Push pose: CEM best](res/h1_bfm_coverage_bfmzero_official/videos/push_pose_cem_best.mp4)
- [Steer pose: global best](res/h1_bfm_coverage_bfmzero_official/videos/steer_pose_global_best.mp4)
- [Steer pose: CEM best](res/h1_bfm_coverage_bfmzero_official/videos/steer_pose_cem_best.mp4)
- [Human push 1, window 00](res/h1_bfm_coverage_bfmzero_official/videos/human_push_1_window_00_cem_best.mp4)
- [Human push 1, window 01](res/h1_bfm_coverage_bfmzero_official/videos/human_push_1_window_01_cem_best.mp4)
- [Human push 1, window 02](res/h1_bfm_coverage_bfmzero_official/videos/human_push_1_window_02_cem_best.mp4)
- [Human push 2, window 00](res/h1_bfm_coverage_bfmzero_official/videos/human_push_2_window_00_cem_best.mp4)
- [Human push 2, window 01](res/h1_bfm_coverage_bfmzero_official/videos/human_push_2_window_01_cem_best.mp4)
- [Human push 2, window 02](res/h1_bfm_coverage_bfmzero_official/videos/human_push_2_window_02_cem_best.mp4)
- [Push-steer push anchor](res/h1_bfm_coverage_bfmzero_official/videos/push_steer_push_anchor.mp4)
- [Push-steer quarter blend](res/h1_bfm_coverage_bfmzero_official/videos/push_steer_quarter_blend.mp4)
- [Push-steer midpoint blend](res/h1_bfm_coverage_bfmzero_official/videos/push_steer_midpoint_blend.mp4)
- [Push-steer three-quarter blend](res/h1_bfm_coverage_bfmzero_official/videos/push_steer_three_quarter_blend.mp4)
- [Push-steer steer anchor](res/h1_bfm_coverage_bfmzero_official/videos/push_steer_steer_anchor.mp4)
- [Representative failure](res/h1_bfm_coverage_bfmzero_official/videos/failure_latent_0025.mp4)

### Limitations

- Expert arrays do not provide complete BFM0 observations; encoded anchors are disabled.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
