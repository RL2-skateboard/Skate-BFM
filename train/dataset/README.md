# Training Data

All Skate-BFM training data is stored under this directory and ignored by
Git. Restore binary artifacts from their authoritative sources before
training.

```text
base/
  lafan_29dof_10s-clipped.pkl
sim_collected/
  train/
    raw/
    phase/
      motion_library/
      qc/
    continuous/
      motion_library/
      qc/
  val/
    raw/
    phase/
      motion_library/
      qc/
    continuous/
      motion_library/
      qc/
  test/
    raw/
    phase/
      motion_library/
      qc/
    continuous/
      motion_library/
      qc/
```

- `base/` contains the official 862-motion BFM-Zero LAFAN training source.
- `sim_collected/train/` is the only split used by formal training.
- `sim_collected/val/` is source-disjoint and reserved for model selection.
- `sim_collected/test/` is source-disjoint from Train and Val and reserved for
  final held-out reporting.
- Every split contains synchronized HUSKY raw rollouts, phase-pure MotionLib
  records, and 500-frame Continuous MotionLib records with QC artifacts.

Download the complete Skate dataset:

```bash
hf download Yak9Ce3teeh/skate-sim-dataset \
  --repo-type dataset \
  --local-dir train/dataset/sim_collected
```

Formal training requires `train/raw/` plus either `train/phase/` or
`train/continuous/`. Do not add `val/` or `test/` to the training sampler.

Download the Base/LAFAN training source:

```bash
mkdir -p train/dataset/base
curl -L \
  https://media.githubusercontent.com/media/LeCAR-Lab/BFM-Zero/main/humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  -o train/dataset/base/lafan_29dof_10s-clipped.pkl
```

Expected Base/LAFAN SHA256:
`7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010`.
