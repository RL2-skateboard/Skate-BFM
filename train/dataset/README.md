# Training Data

All Skate-BFM training data is stored under this directory and ignored by
Git. Restore binary artifacts from their authoritative sources before
training.

```text
base/
  lafan_29dof_10s-clipped.pkl
sim_collected/
  raw/
  phase/
    motion_library/
    qc/
  continuous/
    motion_library/
    qc/
```

- `base/` contains the official 862-motion BFM-Zero LAFAN training source.
- `sim_collected/raw/` contains synchronized HUSKY source rollouts.
- `sim_collected/phase/` contains phase-pure MotionLib records and QC.
- `sim_collected/continuous/` contains fixed-window MotionLib records and QC.

Download the Skate data:

```bash
hf download Yak9Ce3teeh/skate-sim-dataset \
  --repo-type dataset \
  --local-dir train/dataset/sim_collected
```

Download the Base/LAFAN training source:

```bash
mkdir -p train/dataset/base
curl -L \
  https://media.githubusercontent.com/media/LeCAR-Lab/BFM-Zero/main/humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  -o train/dataset/base/lafan_29dof_10s-clipped.pkl
```

Expected Base/LAFAN SHA256:
`7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010`.
