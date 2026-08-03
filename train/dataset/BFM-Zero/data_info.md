# BFM-Zero LAFAN Motion Data

## Scope

This directory contains only the two official BFM-Zero LAFAN motion datasets.
No code, checkpoint, configuration, environment, training output, HUSKY data,
AMASS data, CMU data, or BMLHandball data was added here. No derived data has
been generated.

The source files were already present on the server and were reused without
downloading, deleting, or overwriting them. The source repository is
`LeCAR-Lab/BFM-Zero` at local revision
`318cf44a3262e5bdec5944f82f1a5f509b95d09b`.

## Dataset Placement

| Purpose | File |
|---|---|
| BFM-Zero training | `train/lafan_29dof_10s-clipped.pkl` |
| BFM-Zero tracking/evaluation | `evaluation/lafan_29dof.pkl` |

The original server paths were:

- `model/bfm-zero-source/humanoidverse/data/lafan_29dof_10s-clipped.pkl`
- `model/bfm-zero-source/humanoidverse/data/lafan_29dof.pkl`

The copied files are byte-identical to those source paths. The binary dataset
files are local ignored artifacts; this metadata file is the tracked record.

## File Integrity

| File | Size | SHA256 | LFS pointer | Valid joblib load |
|---|---:|---|---:|---:|
| `train/lafan_29dof_10s-clipped.pkl` | 205,117,479 bytes | `7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010` | No | Yes |
| `evaluation/lafan_29dof.pkl` | 209,659,488 bytes | `f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11` | No | Yes |

The files have a binary joblib/pickle header and do not begin with the Git
LFS pointer marker. They must be loaded with the official-compatible
`joblib.load` path rather than bare `pickle.load`.

## Schema

Both files have top-level type `dict`, mapping motion names to motion records.
Only schema summaries and a small key sample were inspected; the complete
motion dictionaries were not printed.

### Training Dataset

- File: `lafan_29dof_10s-clipped.pkl`
- Motion count: `862`
- Top-level key sample:
  `fallAndGetUp1_subject4_clip0`,
  `fallAndGetUp1_subject4_clip1`,
  `fallAndGetUp1_subject4_clip2`,
  `fallAndGetUp1_subject4_clip3`,
  `fallAndGetUp1_subject4_clip4`,
  `fallAndGetUp1_subject4_clip5`,
  `fallAndGetUp1_subject4_clip6`,
  `fallAndGetUp1_subject4_clip7`
- Per-motion keys:
  `root_trans_offset`, `pose_aa`, `dof`, `root_rot`, `smpl_joints`, `fps`,
  `motion_name`
- Per-motion shapes:
  - `root_trans_offset`: `(300, 3)`
  - `pose_aa`: `(300, 30, 3)`
  - `dof`: `(300, 29)`
  - `root_rot`: `(300, 4)`
  - `smpl_joints`: `(300, 24, 3)`
- FPS: `30` for all 862 motions
- DoF count: `29` for every motion
- Frame count: `300` for every motion
- NaN/Inf: none detected in numeric arrays

### Evaluation Dataset

- File: `lafan_29dof.pkl`
- Motion count: `40`
- Top-level key sample:
  `fallAndGetUp1_subject4`,
  `dance1_subject3`,
  `dance2_subject4`,
  `fightAndSports1_subject1`,
  `jumps1_subject1`,
  `fight1_subject3`,
  `run1_subject5`,
  `dance1_subject2`
- Per-motion keys:
  `root_trans_offset`, `pose_aa`, `dof`, `root_rot`, `smpl_joints`, `fps`
- Per-motion shapes:
  - `root_trans_offset`: `(T, 3)`
  - `pose_aa`: `(T, 30, 3)`
  - `dof`: `(T, 29)`
  - `root_rot`: `(T, 4)`
  - `smpl_joints`: `(T, 24, 3)`
- FPS: `30` for all 40 motions
- DoF count: `29` for every motion
- Frame-count distribution:
  `3066 x1`, `3945 x3`, `4918 x3`, `5047 x3`, `6771 x5`,
  `7135 x2`, `7146 x3`, `7334 x3`, `7345 x2`, `7347 x3`,
  `7355 x2`, `7399 x5`, `7840 x3`, `8194 x2`
- NaN/Inf: none detected in numeric arrays

## Exclusions

- AMASS: no matching AMASS dataset path was found during the server scan.
- CMU: no matching CMU motion dataset path was found during the server scan.
- BMLHandball: no matching BMLHandball dataset path was found during the
  server scan.
- HUSKY: not included in this directory.
- Derived data: none generated.
