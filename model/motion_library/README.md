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

Model binaries and generated outputs are ignored by Git. Keep reproducible
training configuration, dataset provenance, and evaluation results in tracked
project files.
