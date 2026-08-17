# H1 Experiment Results

H1 has two matched sub-experiments. Both use the frozen official BFM0, the
same HUSKY simulator, target states, horizon, seeds, robustness noise, and
coverage thresholds.

## Summary

| Sub-experiment | Latent source | Robust coverage | Interpretation |
|---|---|---:|---|
| `without_prior` | 256 global sphere samples -> broad CEM | 6/8 | Local short-horizon retrieval found for six dynamic targets |
| `with_prior` | Official backward-map expert latent -> local CEM | 0/8 | Encoded expert neighborhoods were not robustly executable |

**Caption.** `6/8` and `0/8` are target-level robust-coverage counts, not
complete skateboarding success rates.

## H1-without-prior

### Protocol

| Parameter | Value |
|---|---:|
| Global latent samples | 256 |
| CEM population / iterations | 64 / 6 |
| CEM max angle | 180 deg |
| Temporal correlation | 0.0 |
| Robustness trials | 20 |
| Seeds | 0, 1, 2 |
| Horizon | 0.5 s |

**Caption.** The search proposes latents from a global normalized sphere scan;
expert data supplies target states and scoring only.

### Target Results

| Target | Global best | CEM best | CEM max angle | Robust rate | Status |
|---|---:|---:|---:|---:|---|
| `push_start_pose` | -0.662178 | -0.523622 | 41.24 deg | 0.000 | not covered |
| `steer_start_pose` | -0.368672 | -0.287402 | 40.77 deg | 0.000 | fragile |
| `human_push_1_window_00` | -0.506456 | -0.235852 | 40.79 deg | 0.750 | covered |
| `human_push_1_window_01` | -0.556736 | -0.479196 | 35.70 deg | 0.750 | covered |
| `human_push_1_window_02` | -0.526268 | -0.393350 | 37.52 deg | 0.850 | covered |
| `human_push_2_window_00` | -0.541474 | -0.315968 | 40.22 deg | 0.750 | covered |
| `human_push_2_window_01` | -0.566813 | -0.471406 | 37.95 deg | 0.900 | covered |
| `human_push_2_window_02` | -0.475021 | -0.301938 | 40.15 deg | 0.900 | covered |

**Caption.** Scores are the evaluator's target-motion scores; larger is
better. `Robust rate` is the fraction of 20 noisy trials satisfying the
configured pose/motion thresholds.

### Plots

![Without-prior latent t-SNE](res/h1_bfm0_motion_without_prior/plots/latent_tsne_2d.png)

*Caption: 2D t-SNE view of sampled latent/behavior points; qualitative
visualization, not the distance metric used for decisions.*

![Without-prior latent sphere](res/h1_bfm0_motion_without_prior/plots/latent_sphere_tsne.png)

*Caption: qualitative t-SNE view of the normalized global latent scan.*

![Without-prior score by geodesic angle](res/h1_bfm0_motion_without_prior/plots/score_vs_geodesic_angle.png)

*Caption: target score versus geodesic distance from the search reference;
angle is measured by `d_geo` in [`exp.md`](exp.md).*

![Without-prior geodesic support](res/h1_bfm0_motion_without_prior/plots/geodesic_support_curve.png)

*Caption: fraction of candidates meeting the local support criterion at each
geodesic angle.*

### Videos

- [Push CEM best](res/h1_bfm0_motion_without_prior/videos/push_pose_cem_best.mp4)
  *Caption: selected CEM rollout from the push static-pose reset.*
- [Steer CEM best](res/h1_bfm0_motion_without_prior/videos/steer_pose_cem_best.mp4)
  *Caption: selected CEM rollout from the steer static-pose reset.*
- [Human push 1 window 00](res/h1_bfm0_motion_without_prior/videos/human_push_1_window_00_cem_best.mp4)
  *Caption: selected CEM rollout for the first human-push target window.*
- [Human push 2 window 02](res/h1_bfm0_motion_without_prior/videos/human_push_2_window_02_cem_best.mp4)
  *Caption: selected CEM rollout for the second human-push target file.*
- [All without-prior videos](res/h1_bfm0_motion_without_prior/videos/)
  *Caption: complete content-named video set for this sub-experiment.*

### Result

The finite global scan plus CEM found local candidates for six dynamic
human-push windows. The two static poses were not robustly covered. Video
inspection shows that some candidates separate from the board, so the result
is motion retrieval evidence rather than a complete push-skill result.

## H1-with-prior

### Protocol

| Parameter | Value |
|---|---:|
| Expert latent source | `z_t = project_z(B(normalize(o[t+1])))` |
| Dynamic latent steps | 24 |
| CEM population / iterations | 64 / 6 |
| CEM max angle | 40 deg |
| Temporal correlation | 0.9 |
| Robustness trials | 20 |
| Seeds | 0, 1, 2 |
| Horizon | 0.5 s |

**Caption.** The prior is an encoded expert latent trajectory; CEM perturbs
that trajectory locally with temporally correlated noise.

### Target Results

| Target | Encoded score | CEM best | CEM max angle | Robust rate | Status |
|---|---:|---:|---:|---:|---|
| `push_start_pose` | -0.727388 | -0.687604 | 13.85 deg | 0.000 | not covered |
| `steer_start_pose` | -0.608622 | -0.499037 | 14.15 deg | 0.000 | not covered |
| `human_push_1_window_00` | -0.530916 | -0.500381 | 13.96 deg | 0.000 | not covered |
| `human_push_1_window_01` | -0.571654 | -0.530974 | 12.17 deg | 0.000 | not covered |
| `human_push_1_window_02` | -0.568220 | -0.519531 | 13.37 deg | 0.000 | not covered |
| `human_push_2_window_00` | -0.586573 | -0.534871 | 13.15 deg | 0.000 | not covered |
| `human_push_2_window_01` | -0.565684 | -0.517439 | 13.42 deg | 0.000 | not covered |
| `human_push_2_window_02` | -0.546455 | -0.508025 | 13.18 deg | 0.000 | not covered |

**Caption.** CEM improved every encoded score, but none of the eight targets
reached the required robust success rate of `0.70`.

### Plots

![With-prior latent t-SNE](res/h1_bfm0_motion_with_prior/plots/latent_tsne_2d.png)

*Caption: 2D t-SNE view of the encoded expert and sampled latent points.*

![With-prior latent sphere](res/h1_bfm0_motion_with_prior/plots/latent_sphere_tsne.png)

*Caption: qualitative normalized latent-space visualization for the prior
trajectory and local samples.*

![With-prior score by geodesic angle](res/h1_bfm0_motion_with_prior/plots/score_vs_geodesic_angle.png)

*Caption: encoded/CEM score as a function of geodesic perturbation angle.*

![With-prior geodesic support](res/h1_bfm0_motion_with_prior/plots/geodesic_support_curve.png)

*Caption: local support rate around the encoded expert latent trajectory.*

### Videos

- [Push encoded anchor](res/h1_bfm0_motion_with_prior/videos/push_pose_encoded_anchor.mp4)
  *Caption: rollout from the encoded static push latent.*
- [Push CEM best](res/h1_bfm0_motion_with_prior/videos/push_pose_cem_best.mp4)
  *Caption: local-CEM refinement of the push latent.*
- [Human push 1 window 00 encoded trajectory](res/h1_bfm0_motion_with_prior/videos/human_push_1_window_00_encoded_trajectory.mp4)
  *Caption: rollout using the time-aligned expert latent trajectory.*
- [Human push 1 window 00 CEM best](res/h1_bfm0_motion_with_prior/videos/human_push_1_window_00_cem_best.mp4)
  *Caption: local-CEM refinement of that dynamic trajectory.*
- [All with-prior videos](res/h1_bfm0_motion_with_prior/videos/)
  *Caption: complete content-named video set for this sub-experiment.*

### Result

The backward-map prior is numerically reconstructible and local CEM improves
the encoded score, but the frozen actor does not robustly maintain any tested
target in the coupled HUSKY rollout.

## Files and Reproduction

- Without prior:
  [`docs/res/h1_bfm0_motion_without_prior/`](res/h1_bfm0_motion_without_prior/)
- With prior:
  [`docs/res/h1_bfm0_motion_with_prior/`](res/h1_bfm0_motion_with_prior/)
- Dataset schema and provenance:
  `dataset_schema.json`, `metadata.json`, `checkpoint_compatibility.json`
  inside each result directory.
- Both experiments use the configuration snapshot in each result directory;
  the current template is [`configs/h1_bfm_coverage.yaml`](../configs/h1_bfm_coverage.yaml).

## Interpretation Boundary

- [x] Frozen BFM0 motion-coverage experiment completed.
- [x] Both prior conditions are recorded separately.
- [x] Curves, tables, plots, and representative videos are indexed.
- [ ] H1 establishes a complete skateboarding controller.
- [ ] H1 proves the full BFM0 latent space has been exhausted.
- [ ] H1 validates a trained Skate-BFM motion library.
