# M2.6 Phase 20k / 50k / 100k Checkpoint Comparison

- Git HEAD: `275426f`
- Benchmark: 80 fixed Test cases, 20 each for `push`, `steer`, `push2steer`, and `steer2push`.
- Exact case identity hash: `ff7a290d384d00ae34ff2ccd8c5765227df711bca0fab5e668532234aa12e3ad`.
- Sampling: rollout-balanced without replacement, seed 4728. All checkpoints replay the identical case order and source provenance.
- Frozen analysis: no training, backward pass, optimizer update, normalizer update, physics-evaluation rerun, or evaluator modification.

## Training Diagnostics

| Checkpoint | Env transitions | Native updates | Episodes | Falls | Tracking ratio | Tracking length mean | z norm mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20k | 20,000 | 1,900 | 17 | 1 | 0.0227 | 45.500 | 16.000 |
| 50k | 50,000 | 4,900 | 236 | 212 | 0.0612 | 33.188 | 16.000 |
| 100k | 100,000 | 9,900 | 1,917 | 1,893 | 0.1886 | 38.293 | 16.000 |

Each checkpoint entry in [`comparison.json`](comparison.json) preserves actual `training_diagnostics.jsonl` metric names, the closest update block, and preceding five-block mean/std/min/max. These numerical diagnostics do not prove closed-loop skill quality.

## Frozen Test Evaluation

### push

| Checkpoint | Complete | Completion ratio | Joint MAE (rad) | Joint velocity MAE (rad/s) | Root orientation (deg) | Board XY (m) | Board heading (deg) | Coupling XY (m) | Feet on board | Root tilt mean/max (deg) | Termination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20k | 95.00% | 0.986 | 0.485 | 1.605 | 18.92 | 0.290 | 12.80 | 0.186 | 0.660 | 13.66 / 26.35 | 5.00% |
| 50k | 15.00% | 0.580 | 0.599 | 2.966 | 29.67 | 0.146 | 3.28 | 0.159 | 0.673 | 36.53 / 101.76 | 85.00% |
| 100k | 25.00% | 0.760 | 0.536 | 2.015 | 21.69 | 0.129 | 5.62 | 0.181 | 0.848 | 30.66 / 77.11 | 75.00% |

Errors are lower-is-better; completion and feet-on-board are higher-is-better. Scalars are evaluator aggregates over this fixed behavior subset, not full-Test generalization estimates.

### steer

| Checkpoint | Complete | Completion ratio | Joint MAE (rad) | Joint velocity MAE (rad/s) | Root orientation (deg) | Board XY (m) | Board heading (deg) | Coupling XY (m) | Feet on board | Root tilt mean/max (deg) | Termination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20k | 90.00% | 0.965 | 0.431 | 1.513 | 23.79 | 0.750 | 15.98 | 0.783 | 0.362 | 10.35 / 24.67 | 10.00% |
| 50k | 20.00% | 0.730 | 0.456 | 1.913 | 22.76 | 0.243 | 10.63 | 0.360 | 0.694 | 21.89 / 82.97 | 80.00% |
| 100k | 75.00% | 0.894 | 0.431 | 1.168 | 28.89 | 0.160 | 13.57 | 0.151 | 0.922 | 21.77 / 53.20 | 25.00% |

Errors are lower-is-better; completion and feet-on-board are higher-is-better. Scalars are evaluator aggregates over this fixed behavior subset, not full-Test generalization estimates.

### push2steer

| Checkpoint | Complete | Completion ratio | Joint MAE (rad) | Joint velocity MAE (rad/s) | Root orientation (deg) | Board XY (m) | Board heading (deg) | Coupling XY (m) | Feet on board | Root tilt mean/max (deg) | Termination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20k | 80.00% | 0.893 | 0.456 | 1.495 | 41.35 | 1.353 | 20.11 | 0.754 | 0.406 | 11.73 / 32.95 | 20.00% |
| 50k | 0.00% | 0.121 | 0.707 | 4.146 | 32.72 | 0.043 | 0.81 | 0.069 | 0.395 | 48.97 / 92.55 | 100.00% |
| 100k | 5.00% | 0.301 | 0.526 | 2.162 | 24.73 | 0.264 | 3.68 | 0.147 | 0.904 | 31.49 / 86.00 | 95.00% |

Errors are lower-is-better; completion and feet-on-board are higher-is-better. Scalars are evaluator aggregates over this fixed behavior subset, not full-Test generalization estimates.

### steer2push

| Checkpoint | Complete | Completion ratio | Joint MAE (rad) | Joint velocity MAE (rad/s) | Root orientation (deg) | Board XY (m) | Board heading (deg) | Coupling XY (m) | Feet on board | Root tilt mean/max (deg) | Termination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20k | 100.00% | 1.000 | 0.451 | 1.461 | 40.70 | 2.208 | 18.88 | 1.652 | 0.179 | 11.27 / 21.80 | 0.00% |
| 50k | 0.00% | 0.222 | 0.503 | 2.383 | 25.45 | 0.112 | 5.63 | 0.184 | 0.728 | 27.47 / 104.16 | 100.00% |
| 100k | 0.00% | 0.434 | 0.461 | 1.582 | 29.76 | 0.195 | 15.72 | 0.194 | 0.856 | 25.38 / 91.92 | 100.00% |

Errors are lower-is-better; completion and feet-on-board are higher-is-better. Scalars are evaluator aggregates over this fixed behavior subset, not full-Test generalization estimates.

## Transition Sections

For transition cases, these are unweighted means of available case-level section means. `PRE`, `TRANSITION`, and `POST` are labeled at canonical Raw frame `reset_raw_frame+t+1`.

### push2steer

| Checkpoint | Section | Cases | Joint MAE (rad) | Root orientation (deg) | Board XY (m) | Coupling XY (m) |
|---|---|---:|---:|---:|---:|---:|
| 20k | pre | 20 | 0.486 | 10.45 | 0.036 | 0.074 |
| 20k | transition | 20 | 0.503 | 41.87 | 0.344 | 0.216 |
| 20k | post | 19 | 0.427 | 58.34 | 1.780 | 0.885 |
| 50k | pre | 20 | 0.707 | 32.72 | 0.043 | 0.069 |
| 50k | transition | 0 | - | - | - | - |
| 50k | post | 0 | - | - | - | - |
| 100k | pre | 20 | 0.501 | 13.66 | 0.053 | 0.101 |
| 100k | transition | 19 | 0.644 | 66.44 | 0.386 | 0.253 |
| 100k | post | 4 | 0.556 | 59.61 | 1.618 | 0.482 |

### steer2push

| Checkpoint | Section | Cases | Joint MAE (rad) | Root orientation (deg) | Board XY (m) | Coupling XY (m) |
|---|---|---:|---:|---:|---:|---:|
| 20k | pre | 20 | 0.457 | 13.50 | 0.311 | 0.370 |
| 20k | transition | 20 | 0.496 | 40.04 | 0.777 | 0.862 |
| 20k | post | 20 | 0.447 | 48.83 | 1.976 | 1.673 |
| 50k | pre | 20 | 0.449 | 18.43 | 0.087 | 0.151 |
| 50k | transition | 14 | 0.884 | 85.75 | 0.259 | 0.402 |
| 50k | post | 2 | 0.815 | 54.76 | 0.325 | 0.427 |
| 100k | pre | 20 | 0.383 | 8.78 | 0.037 | 0.064 |
| 100k | transition | 20 | 0.498 | 31.60 | 0.106 | 0.158 |
| 100k | post | 18 | 0.546 | 59.12 | 0.422 | 0.379 |

## Paired Case Comparison

tie when abs(right-left) <= 1e-9 + 1e-6*max(abs(left),abs(right)); otherwise higher completion is improve and lower error is improve.

### 20k_vs_50k

| Behavior | Metric | Right improved | Right worsened | Tie | Right - left mean |
|---|---|---:|---:|---:|---:|
| push | completion_ratio (higher) | 1 | 16 | 3 | -0.406500 |
| push | joint_position_mae_rad (lower) | 2 | 18 | 0 | 0.114137 |
| push | board_xy_displacement_error_m (lower) | 16 | 4 | 0 | -0.143464 |
| push | coupling_xy_error_m (lower) | 12 | 8 | 0 | -0.027660 |
| steer | completion_ratio (higher) | 2 | 14 | 4 | -0.235500 |
| steer | joint_position_mae_rad (lower) | 9 | 11 | 0 | 0.025588 |
| steer | board_xy_displacement_error_m (lower) | 20 | 0 | 0 | -0.506788 |
| steer | coupling_xy_error_m (lower) | 17 | 3 | 0 | -0.422709 |
| push2steer | completion_ratio (higher) | 0 | 20 | 0 | -0.772200 |
| push2steer | joint_position_mae_rad (lower) | 0 | 20 | 0 | 0.251744 |
| push2steer | board_xy_displacement_error_m (lower) | 20 | 0 | 0 | -1.310036 |
| push2steer | coupling_xy_error_m (lower) | 19 | 1 | 0 | -0.685137 |
| steer2push | completion_ratio (higher) | 0 | 20 | 0 | -0.777600 |
| steer2push | joint_position_mae_rad (lower) | 2 | 18 | 0 | 0.052472 |
| steer2push | board_xy_displacement_error_m (lower) | 20 | 0 | 0 | -2.095863 |
| steer2push | coupling_xy_error_m (lower) | 20 | 0 | 0 | -1.467945 |

### 50k_vs_100k

| Behavior | Metric | Right improved | Right worsened | Tie | Right - left mean |
|---|---|---:|---:|---:|---:|
| push | completion_ratio (higher) | 15 | 2 | 3 | 0.181000 |
| push | joint_position_mae_rad (lower) | 20 | 0 | 0 | -0.063080 |
| push | board_xy_displacement_error_m (lower) | 13 | 7 | 0 | -0.017239 |
| push | coupling_xy_error_m (lower) | 7 | 13 | 0 | 0.022263 |
| steer | completion_ratio (higher) | 13 | 3 | 4 | 0.164000 |
| steer | joint_position_mae_rad (lower) | 12 | 8 | 0 | -0.025384 |
| steer | board_xy_displacement_error_m (lower) | 14 | 6 | 0 | -0.082387 |
| steer | coupling_xy_error_m (lower) | 18 | 2 | 0 | -0.208950 |
| push2steer | completion_ratio (higher) | 20 | 0 | 0 | 0.179800 |
| push2steer | joint_position_mae_rad (lower) | 20 | 0 | 0 | -0.181229 |
| push2steer | board_xy_displacement_error_m (lower) | 1 | 19 | 0 | 0.221324 |
| push2steer | coupling_xy_error_m (lower) | 3 | 17 | 0 | 0.077767 |
| steer2push | completion_ratio (higher) | 19 | 1 | 0 | 0.211200 |
| steer2push | joint_position_mae_rad (lower) | 15 | 5 | 0 | -0.042384 |
| steer2push | board_xy_displacement_error_m (lower) | 5 | 15 | 0 | 0.082876 |
| steer2push | coupling_xy_error_m (lower) | 7 | 13 | 0 | 0.010345 |

### 20k_vs_100k

| Behavior | Metric | Right improved | Right worsened | Tie | Right - left mean |
|---|---|---:|---:|---:|---:|
| push | completion_ratio (higher) | 0 | 15 | 5 | -0.225500 |
| push | joint_position_mae_rad (lower) | 3 | 17 | 0 | 0.051057 |
| push | board_xy_displacement_error_m (lower) | 16 | 4 | 0 | -0.160703 |
| push | coupling_xy_error_m (lower) | 9 | 11 | 0 | -0.005398 |
| steer | completion_ratio (higher) | 2 | 5 | 13 | -0.071500 |
| steer | joint_position_mae_rad (lower) | 12 | 8 | 0 | 0.000204 |
| steer | board_xy_displacement_error_m (lower) | 20 | 0 | 0 | -0.589174 |
| steer | coupling_xy_error_m (lower) | 19 | 1 | 0 | -0.631659 |
| push2steer | completion_ratio (higher) | 1 | 18 | 1 | -0.592400 |
| push2steer | joint_position_mae_rad (lower) | 2 | 18 | 0 | 0.070515 |
| push2steer | board_xy_displacement_error_m (lower) | 18 | 2 | 0 | -1.088713 |
| push2steer | coupling_xy_error_m (lower) | 19 | 1 | 0 | -0.607370 |
| steer2push | completion_ratio (higher) | 0 | 20 | 0 | -0.566400 |
| steer2push | joint_position_mae_rad (lower) | 5 | 15 | 0 | 0.010088 |
| steer2push | board_xy_displacement_error_m (lower) | 20 | 0 | 0 | -2.012988 |
| steer2push | coupling_xy_error_m (lower) | 20 | 0 | 0 | -1.457600 |

## Latent-Space Reconstruction

- Reference prior: official model.sample_z(4096), seed=4728; standard-normal draw followed by project_z=sqrt(z_dim)*normalize because norm_z=True.
- Reference norm: mean `16.000000`, std `0.000001`, min/max `15.999998` / `16.000002`.
- Shared PCA fingerprint: `e7982e69994c7334d2e0515d6ff448cc982898ffbb7ab5009732a4d515cbd729`; explained variance ratio: 0.10177, 0.06914, 0.05542.
- Method: `u=z/||z||`, shared 3D PCA over reference and all checkpoint/phase directions, then `v=PCA(u)/||PCA(u)||` for a unit sphere. This is a shared direction visualization, not lossless 256D geometry or a behavior-quality metric.

| Checkpoint | Visited z count: push / steer / push2steer / steer2push | z norm mean +/- std |
|---|---:|---:|
| 20k | 5688 / 5959 / 1183 / 540 | 16.000000 +/- 0.000000 |
| 50k | 1807 / 2412 / 0 / 118 | 16.000000 +/- 0.000000 |
| 100k | 3407 / 2974 / 288 / 313 | 16.000000 +/- 0.000000 |

- [Combined latent comparison](latent_space_compare.png) and [shared projection metadata](latent_projection.json).
- [20k latent space](../20k-s20b_test_phase_eval/latent_space.png) and [metadata](../20k-s20b_test_phase_eval/latent_space.json).
- [50k latent space](../50k-s20b_test_phase_eval/latent_space.png) and [metadata](../50k-s20b_test_phase_eval/latent_space.json).
- [100k latent space](../100k-s20b_test_phase_eval/latent_space.png) and [metadata](../100k-s20b_test_phase_eval/latent_space.json).

## Interpretation

- The 20k, 50k, and 100k checkpoints are comparable only on this fixed 80-case held-out Test subset. Case identity and source physics provenance are exactly equal.
- Finite losses, bounded latent norms, and checkpoint reloads establish numerical/protocol integrity, not successful Skate behavior.
- Completion and physical tracking errors differ by skill and checkpoint. The paired table is the authoritative direction-of-change summary because it compares identical cases.
- Latent figures show checkpoint-specific tracking latents from the exact evaluator bridge. They describe accessed directions, not action-space coverage or causal skill separation.
