# Motion Library

This directory stores generated motion-library checkpoints and related model
outputs from the `train` branch.

Each completed training run is grouped by a stable model name:

```text
<model_name>/
├── checkpoint_20000/
├── checkpoint_50000/
└── checkpoint_100000/
```

Published model:

```text
m2.6-phase-100k-seed4728/
└── checkpoint_100000/
```

The complete 100k checkpoint is hosted at
[`Yak9Ce3teeh/skate-bfm`](https://huggingface.co/Yak9Ce3teeh/skate-bfm/tree/main/motion_library/m2.6-phase-100k-seed4728).
Restore it from the repository root:

```bash
hf download Yak9Ce3teeh/skate-bfm \
  --include "motion_library/m2.6-phase-100k-seed4728/**" \
  --local-dir model
```

The checkpoints reload correctly and contain finite model weights. Frozen
evaluation currently shows early falls on HUSKY, so these artifacts are
diagnostic training outputs rather than validated skate policies.

Model binaries and generated outputs are ignored by Git. Keep reproducible
training configuration, dataset provenance, and evaluation results in tracked
project files.
