# Experiment Log

## 2026-08-01

- Added MuJoCo visualization to the existing BFM0-HUSKY smoke command.

## 2026-07-30

- Wrapped the HUSKY simulator under `husky_sim/`.
- Connected the basic BFM0 interface to the HUSKY 23DoF runtime.
- Configured the `skatebfm` Conda environment.
- Organized the repository structure and documentation.

## 2026-08-01

### h1_bfm_coverage_smoke_20260801_114246

- Experiment type: H1 Expert-Guided BFM0 Behavior Coverage
- Run type: smoke
- Start time: 2026-08-01T11:42:47.264393+08:00
- End time: 2026-08-01T11:42:49.075809+08:00
- Duration: 1.805 seconds
- Checkpoint: `/tmp/h1_bfm0_temporary_smoke.pt`
- Git commit: `bb49f5ae36b8e8137db094724f8da46cf86e56e8`
- Configuration:
  - global latents: 4
  - CEM population: 4
  - CEM iterations: 1
  - horizon: 0.04 seconds
  - seeds: [0]
- Enabled expert targets: push_start_pose
- Unsupported expert targets: encoded expert anchors (incomplete BFM0 observations)
- Result directory: `docs/res/h1_bfm_coverage_smoke_20260801_114246/`
- Ruff: passed
- Pytest: 17 passed
- Main status: completed
- Known limitations: This run uses a temporary random checkpoint and supports pipeline validation only.; Expert arrays do not provide complete BFM0 observations; encoded anchors are disabled.; Static scoring uses all 30 confirmed robot bodies relative to the skateboard.; Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.; The current short-horizon experiment does not validate complete skateboarding.; Foot contact metrics are not included in H1 coverage.; t-SNE sphere plots are qualitative; quantitative distances use original latents.

## 2026-08-01

### h1_bfm_coverage_smoke_videos

- Experiment type: H1 Expert-Guided BFM0 Behavior Coverage
- Run type: smoke
- Start time: 2026-08-01T17:44:02.856857+08:00
- End time: 2026-08-01T17:44:05.026879+08:00
- Duration: 2.164 seconds
- Checkpoint: `/tmp/h1_bfm0_temporary_smoke.pt`
- Git commit: `56141c55bda4397300ca0d09e94f3051605b7b44`
- Configuration:
  - global latents: 4
  - CEM population: 4
  - CEM iterations: 1
  - horizon: 0.04 seconds
  - seeds: [0]
- Enabled expert targets: push_start_pose
- Unsupported expert targets: encoded expert anchors (incomplete BFM0 observations)
- Result directory: `docs/res/h1_bfm_coverage_smoke_videos/`
- Ruff: passed
- Pytest: 18 passed
- Main status: completed
- Known limitations: This run uses a temporary random checkpoint and supports pipeline validation only.; Expert arrays do not provide complete BFM0 observations; encoded anchors are disabled.; Static scoring uses all 30 confirmed robot bodies relative to the skateboard.; Dynamic scoring uses only the confirmed common 23 joint positions at 50 Hz.; The current short-horizon experiment does not validate complete skateboarding.; Foot contact metrics are not included in H1 coverage.; t-SNE sphere plots are qualitative; quantitative distances use original latents.
