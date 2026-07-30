# Experiment Log

## 2026-07-30

- Initialized the clean Skate-BFM repository under the RL2-skateboard
  organization.
- Defined a concise project layout for configuration, documentation, BFM0
  components, HUSKY simulation, integration adapters, and tests.
- Pinned the HUSKY simulator as a Git submodule at upstream revision
  `d93833e80deff7f927c0b80ef9c435d8b5c488fe` and recorded its license and
  provenance.
- Added a lightweight headless MuJoCo runtime around the official HUSKY scene.
- Implemented a compact BFM0 forward-backward model interface for policy,
  goal-latent, and reward-latent integration tests.
- Implemented name-based BFM0 G1-29DoF to HUSKY G1-23DoF action mapping.
- Implemented HUSKY-to-BFM0 state and temporal-history conversion.
- Added the `skatebfm` Conda specification, setup script, baseline
  configuration, unit tests, and end-to-end smoke command.
- Created and validated the local `skatebfm` environment with Python 3.12.13,
  PyTorch 2.5.1, CUDA 12.4, and MuJoCo 3.11.0.
- Passed five unit tests and a 20-step headless rollout in the official HUSKY
  MuJoCo scene.
- Added the repository overview, research hypothesis, pipeline figure, and
  initial experiment-results record.
