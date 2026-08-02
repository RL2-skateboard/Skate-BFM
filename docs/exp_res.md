# Experiment Results

## Comparison

| Study date | Experiment | Latent proposal | Stable coverage | Main result |
|---|---|---|---:|---|
| 2026-07-31 | `h1_bfm0_motion_without_prior` | Target-conditioned retrieval: 256 global sphere samples, then broad CEM from each target's global best | 6/8 | Retrieved stable short-horizon joint-motion candidates for all six human-push windows without expert latent proposals |
| 2026-08-01 | `h1_bfm0_motion_with_prior` | Time-aligned expert `z_t`, then trajectory-local CEM | 0/8 | The encoded expert neighborhoods were not robustly maintained by the frozen actor in HUSKY |

Both experiments were rerun after correcting the human-push world alignment.
For every dynamic window, the right ankle starts at the static push reference
in board coordinates, approximately `(0.069, -0.002, 0.048)` meters. The left
ankle starts outside the deck width at `y=0.269-0.564` meters. In the first
window of each source motion it is near ground level at board-relative
`z=-0.057/-0.046` meters; later windows preserve the source swing-foot height.
The complete root trajectory and initial velocity receive the same rigid
transform.

The without-prior run is goal-directed retrieval over the frozen BFM0 latent
space. Expert data defines target trajectories, initial states, and scores but
does not propose latents. Broad CEM found locally robust constant-latent
candidates for all six dynamic windows, with robust success rates
`0.75-0.90`; neither static target was stably covered.

The with-prior run uses 24 time-aligned next-frame expert latents for every
dynamic window. Trajectory-local CEM improved all encoded scores while staying
within `12.17-14.15` degrees of the expert trajectory, but no target achieved
robust coverage.

The `6/8` result proves only that short-horizon motions satisfying the current
23-joint trajectory thresholds can be retrieved from the tested frozen model.
Video inspection shows that some candidates separate from the skateboard
during the rollout. Because scoring does not include root trajectory, foot
contact, or synchronized skateboard state, these candidates are not six
complete push skills. Failed finite searches also do not establish the
motion-library limit of BFM0.

# H1 Frozen BFM0 Motion Coverage

## h1_bfm0_motion_without_prior

### Experiment metadata

- Experiment name: h1_bfm0_motion_without_prior
- Experiment type: H1 Frozen BFM0 Motion Coverage
- Study date: 2026-07-31
- Prior mode: `without_prior`
- Start time: 2026-08-02T12:43:55.487680+08:00
- End time: 2026-08-02T12:53:31.422477+08:00
- Duration: 575.908 seconds
- Git commit: `44e4328330d90818d3e57502e0f2d2d082388fbf`
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
- Human-push rollouts start from each window's first expert qpos/qvel; the right support foot is aligned to the push-start deck reference while the left push foot preserves its expert offset

### Main results

| Expert target | Latent steps | Encoded score | Encoded robust | Global best | CEM best | CEM max angle | CEM robust | Coverage type |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| push_start_pose | 0 | n/a | n/a | -0.662178 | -0.523622 | 41.24 deg | 0.000 | not_covered |
| steer_start_pose | 0 | n/a | n/a | -0.368672 | -0.287402 | 40.77 deg | 0.000 | fragile |
| human_push_1_window_00 | 0 | n/a | n/a | -0.506456 | -0.235852 | 40.79 deg | 0.750 | locally_covered |
| human_push_1_window_01 | 0 | n/a | n/a | -0.556736 | -0.479196 | 35.70 deg | 0.750 | locally_covered |
| human_push_1_window_02 | 0 | n/a | n/a | -0.526268 | -0.393350 | 37.52 deg | 0.850 | locally_covered |
| human_push_2_window_00 | 0 | n/a | n/a | -0.541474 | -0.315968 | 40.22 deg | 0.750 | locally_covered |
| human_push_2_window_01 | 0 | n/a | n/a | -0.566813 | -0.471406 | 37.95 deg | 0.900 | locally_covered |
| human_push_2_window_02 | 0 | n/a | n/a | -0.475021 | -0.301938 | 40.15 deg | 0.900 | locally_covered |

### Latent-space results

| Metric | Result |
|---|---:|
| Push-steer angular distance (degrees) | 90.02168273925781 |
| Global stable proposal rate | 0.98974609375 |
| Latent-behavior Spearman correlation | 0.038124072015193224 |

### Main findings

No expert backward-map latent is used for proposal generation. Expert poses and motions are used only as common evaluation targets and rollout initial conditions.

Broad CEM improved the global-best score for 8/8 targets. Its distance from the global-best initialization ranged 35.70-41.24 degrees.

The search first evaluates uniformly sampled constant latent directions on the frozen BFM0 sphere, then refines each target from its global best.

6/8 targets met the formal coverage criteria. Coverage failure means the selected frozen-BFM0 behavior was not maintained under the adapted actor, coupled skateboard physics, and current thresholds.

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
- [Human push 1 window 01: CEM best](res/h1_bfm0_motion_without_prior/videos/human_push_1_window_01_cem_best.mp4)
- [Human push 1 window 02: CEM best](res/h1_bfm0_motion_without_prior/videos/human_push_1_window_02_cem_best.mp4)
- [Human push 2 window 00: CEM best](res/h1_bfm0_motion_without_prior/videos/human_push_2_window_00_cem_best.mp4)
- [Human push 2 window 01: CEM best](res/h1_bfm0_motion_without_prior/videos/human_push_2_window_01_cem_best.mp4)
- [Human push 2 window 02: CEM best](res/h1_bfm0_motion_without_prior/videos/human_push_2_window_02_cem_best.mp4)
- [Push-steer midpoint](res/h1_bfm0_motion_without_prior/videos/push_steer_midpoint_blend.mp4)
- [All generated videos](res/h1_bfm0_motion_without_prior/videos/)

### Limitations

- Static expert velocities are set to zero when constructing backward observations.
- Dynamic expert velocities are reconstructed by finite differences at 50 Hz.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- Human-push files do not include synchronized skateboard state; initialization aligns the right support foot to the push-start deck reference and preserves the left push foot's expert-relative position and velocity.
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
- Start time: 2026-08-02T12:55:27.095285+08:00
- End time: 2026-08-02T13:05:04.721771+08:00
- Duration: 577.582 seconds
- Git commit: `24dd9ddab94a0cf5f8239a1eca851e3c01d30107`
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
- Human-push rollouts start from each window's first expert qpos/qvel; the right support foot is aligned to the push-start deck reference while the left push foot preserves its expert offset

### Main results

| Expert target | Latent steps | Encoded score | Encoded robust | Global best | CEM best | CEM max angle | CEM robust | Coverage type |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| push_start_pose | 1 | -0.727388 | 0.000 | -0.662178 | -0.687604 | 13.85 deg | 0.000 | not_covered |
| steer_start_pose | 1 | -0.608622 | 0.000 | -0.368672 | -0.499037 | 14.15 deg | 0.000 | not_covered |
| human_push_1_window_00 | 24 | -0.530916 | 0.000 | -0.506456 | -0.500381 | 13.96 deg | 0.000 | not_covered |
| human_push_1_window_01 | 24 | -0.571654 | 0.000 | -0.556736 | -0.530974 | 12.17 deg | 0.000 | not_covered |
| human_push_1_window_02 | 24 | -0.568220 | 0.000 | -0.526268 | -0.519531 | 13.37 deg | 0.000 | not_covered |
| human_push_2_window_00 | 24 | -0.586573 | 0.000 | -0.541474 | -0.534871 | 13.15 deg | 0.000 | not_covered |
| human_push_2_window_01 | 24 | -0.565684 | 0.000 | -0.566813 | -0.517439 | 13.42 deg | 0.000 | not_covered |
| human_push_2_window_02 | 24 | -0.546455 | 0.000 | -0.475021 | -0.508025 | 13.18 deg | 0.000 | not_covered |

### Latent-space results

| Metric | Result |
|---|---:|
| Push-steer angular distance (degrees) | 79.63737487792969 |
| Global stable proposal rate | 0.98974609375 |
| Latent-behavior Spearman correlation | 0.038124072015193224 |

### Main findings

Static goal latents and time-aligned dynamic latent trajectories are produced by the frozen official backward map from reconstructed expert observations. Dynamic execution follows the official next-frame pattern `z_t = project_z(backward_map(obs[t + 1]))`.

Trajectory-local CEM improved the encoded score for 8/8 targets. The per-target maximum trajectory angle ranged 12.17-14.15 degrees.

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
- [Human push 2 window 02: encoded trajectory](res/h1_bfm0_motion_with_prior/videos/human_push_2_window_02_encoded_trajectory.mp4)
- [Push pose: CEM best](res/h1_bfm0_motion_with_prior/videos/push_pose_cem_best.mp4)
- [Steer pose: CEM best](res/h1_bfm0_motion_with_prior/videos/steer_pose_cem_best.mp4)
- [Human push 1 window 00: CEM best](res/h1_bfm0_motion_with_prior/videos/human_push_1_window_00_cem_best.mp4)
- [Human push 2 window 02: CEM best](res/h1_bfm0_motion_with_prior/videos/human_push_2_window_02_cem_best.mp4)
- [Push-steer midpoint](res/h1_bfm0_motion_with_prior/videos/push_steer_midpoint_blend.mp4)
- [All generated videos](res/h1_bfm0_motion_with_prior/videos/)

### Limitations

- Static expert velocities are set to zero when constructing backward observations.
- Dynamic expert velocities are reconstructed by finite differences at 50 Hz.
- Static scoring uses all 30 confirmed robot bodies relative to the skateboard.
- Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.
- Human-push files do not include synchronized skateboard state; initialization aligns the right support foot to the push-start deck reference and preserves the left push foot's expert-relative position and velocity.
- The current short-horizon experiment does not validate complete skateboarding.
- Foot contact metrics are not included in H1 coverage.
- t-SNE sphere plots are qualitative; quantitative distances use original latents.
- Dynamic score-angle plots use trajectory midpoints for display; CEM constraints and reported maximum angles use every original latent step.
