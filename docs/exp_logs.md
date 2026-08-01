# Experiment Log

## 2026-08-01

- Integrated and strictly loaded the official pretrained BFM-Zero checkpoint.
- Calibrated the official BFM0 observation and action interfaces for HUSKY.
- Completed the first formal H1 behavior-coverage experiment.
- Corrected steer evaluation to start with both feet on the skateboard.
- Corrected static-pose success to require the final pose to meet the threshold.
- Generated the latent-space plots and representative MuJoCo videos.
- Made H1 smoke runs temporary so they leave no experiment records or artifacts.

## 2026-07-30

- Wrapped the HUSKY simulator under `husky_sim/`.
- Connected the basic BFM0 interface to the HUSKY 23DoF runtime.
- Configured the `skatebfm` Conda environment.
- Organized the repository structure and documentation.
