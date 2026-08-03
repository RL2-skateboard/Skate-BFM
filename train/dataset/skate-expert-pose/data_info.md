# HUSKY Skate Expert Pose and Motion Data

## Scope

This directory classifies the unmodified pose and reference-motion arrays
provided by the pinned official HUSKY project at submodule revision
`d93833e80deff7f927c0b80ef9c435d8b5c488fe`.

The complete `husky_sim/upstream/dataset/` tree contains four files. No other
official motion, reference-pose, transition-pose, or transition-motion data
file was found. The organized `.npy` entries are relative symbolic links to
the official files; no source file was moved, deleted, overwritten, merged,
interpolated, redirected, or converted.

## Classification

| Official source | Organized path | Classification | Shape | Dtype | Frames | Size | SHA256 | Static pose | Dynamic motion | NaN/Inf |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|
| `husky_sim/upstream/dataset/skate_push/human_push_1.npy` | `dynamic_motion/push/human_push_1.npy` | Dynamic push reference motion | `(180, 36)` | `float64` | 180 | 51,968 bytes | `0e4c0a95cf671681d089bd3c726e499eb4fdf5f69243fae81dd1d1ff3d0e48fb` | No | Yes | No |
| `husky_sim/upstream/dataset/skate_push/human_push_2.npy` | `dynamic_motion/push/human_push_2.npy` | Dynamic push reference motion | `(221, 36)` | `float64` | 221 | 63,776 bytes | `ff9dcf23cf1af62055dba76bced4ac453b845a79e6c397a2e394495be191861e` | No | Yes | No |
| `husky_sim/upstream/dataset/ref_pose/push_start_pose_b.npy` | `canonical_pose/push/push_start_pose_b.npy` | Canonical push pose | `(30, 7)` | `float32` | 1 | 968 bytes | `c9efdf242b3e728af531134ed7488ac01b1a025dbe3cd410827b91a0f140d9fc` | Yes | No | No |
| `husky_sim/upstream/dataset/ref_pose/steer_start_pose_b.npy` | `canonical_pose/steer/steer_start_pose_b.npy` | Canonical steer pose | `(30, 7)` | `float32` | 1 | 968 bytes | `f38e00f1a53f0339d3e071fb587d2047b652cb1d9f8e07400fcc76bebe6845b0` | Yes | No | No |

`Frames = 1` for each canonical pose means one pose containing 30 robot-body
rows. It does not mean a 30-frame trajectory.

## Read-Only Inspection

### `human_push_1.npy`

- Sequence type: continuous reference motion
- Shape: `(180, 36)`
- Global numeric range: `[-3.85087939453125, 1.3483298839497726]`
- Columns `0:3` range: `[-3.85087939453125, 0.7978767581256585]`
- Columns `3:7` range: `[-0.6744936100458029, 0.7980652842041387]`
- Columns `7:36` range: `[-1.3562294786453013, 1.3483298839497726]`
- Finite values: all

### `human_push_2.npy`

- Sequence type: continuous reference motion
- Shape: `(221, 36)`
- Global numeric range: `[-4.9010966796875, 1.5499179137274302]`
- Columns `0:3` range: `[-4.9010966796875, 0.8104192922503226]`
- Columns `3:7` range: `[-0.11832425081565001, 0.7572891816307254]`
- Columns `7:36` range: `[-1.2805104676762145, 1.5499179137274302]`
- Finite values: all

The official `G1_AMPLoader` identifies columns `0:3` as base position,
columns `3:7` as base quaternion in `wxyz` order, and columns `7:36` as
29-DoF position in MuJoCo joint order. The current loader implementation
copies `7:35` into its internal buffer; this report records that source
behavior but does not alter the official arrays.

### `push_start_pose_b.npy`

- Sequence type: one canonical body pose
- Shape: `(30, 7)`
- Global numeric range: `[-0.5627859234809875, 0.9994242191314697]`
- Body-position columns `0:3` range:
  `[-0.18518255650997162, 0.9199886322021484]`
- Body-quaternion columns `3:7` range:
  `[-0.5627859234809875, 0.9994242191314697]`
- Finite values: all

### `steer_start_pose_b.npy`

- Sequence type: one canonical body pose
- Shape: `(30, 7)`
- Global numeric range: `[-0.5251820683479309, 0.9322994947433472]`
- Body-position columns `0:3` range:
  `[-0.1567169725894928, 0.9322994947433472]`
- Body-quaternion columns `3:7` range:
  `[-0.5251820683479309, 0.7886719107627869]`
- Finite values: all

## Official Usage Evidence

- `husky_sim/upstream/src/mjlab_husky/rl/config.py` sets
  `amp_motion_files = "dataset/skate_push"`.
- `husky_sim/upstream/rsl_rl/utils/motion_loader_g1.py` loads every array in
  that directory as a motion trajectory and uses a 50 Hz frame duration.
- `husky_sim/upstream/src/mjlab_husky/envs/g1_skate_rl_env.py` loads
  `push_start_pose_b.npy` and `steer_start_pose_b.npy`, splits each row into
  body position and quaternion, and repeats the single pose across
  environments.
- The same environment constructs push-to-steer and steer-to-push transition
  targets online using current body state, Bézier position interpolation, and
  quaternion Slerp. No independent official transition pose or motion file is
  provided.

## Interpretation and Exclusions

- The dynamic push motions can be used as dynamic expert references for
  Skate-BFM.
- The canonical push pose is only a static pose anchor.
- The canonical steer pose is not a dynamic steer expert motion.
- `DYNAMIC_STEER_EXPERT_NOT_FOUND`
- No independently recorded dynamic transition motion was found.
- `transition_pose/` is currently empty.
- `unknown/` is currently empty; all four official files were classified from
  source usage and array structure.
- The current data contains no online FB rollout.
- The current data contains no teacher pseudo-expert.
- The current data has not been converted into BFM-compatible observations.
- No action, contact label, interpolation, derived trajectory, or merged file
  was generated.
