# Motion Library

This directory stores generated motion-library checkpoints and related model
outputs from the `train` branch.

Each completed training run is grouped by its completion timestamp:

```text
YYYY-MM-DD_HHMMSS/
├── checkpoint_20000/
├── checkpoint_50000/
└── checkpoint_100000/
```

Current local run:

```text
2026-08-15_143013/
├── checkpoint_20000/
├── checkpoint_50000/
└── checkpoint_100000/
```

The checkpoints reload correctly and contain finite model weights. Frozen
evaluation currently shows early falls on HUSKY, so these artifacts are
diagnostic training outputs rather than validated skate policies.

Model binaries and generated outputs are ignored by Git. Keep reproducible
training configuration, dataset provenance, and evaluation results in tracked
project files.
