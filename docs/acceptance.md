# skate-sim acceptance report

- Overall status: **PASS**
- Requested stage: `all`
- Requested backend: `all`

## Stage 01 / mujoco

Status: **PASS**

| Check | Result |
| --- | --- |
| `step_advances_time` | PASS |
| `wrench_changes_position` | PASS |
| `reset_restores_time` | PASS |
| `finite_state` | PASS |

## Stage 01 / mjlab

Status: **PASS**

| Check | Result |
| --- | --- |
| `cuda_physics` | PASS |
| `step_advances_time` | PASS |
| `wrench_changes_position` | PASS |
| `reset_restores_time` | PASS |
| `finite_state` | PASS |

```json
{
  "num_envs": 64,
  "device": "cuda:0"
}
```
## Stage 02 / mujoco

Status: **PASS**

| Check | Result |
| --- | --- |
| `style_standard` | PASS |
| `style_cruiser` | PASS |
| `style_longboard` | PASS |
| `style_street` | PASS |
| `style_downhill` | PASS |

```json
{
  "standard": {
    "nq": 13,
    "nv": 12,
    "ncon": 8,
    "hash": "73293e591509"
  },
  "cruiser": {
    "nq": 13,
    "nv": 12,
    "ncon": 8,
    "hash": "8adf07c01ebc"
  },
  "longboard": {
    "nq": 13,
    "nv": 12,
    "ncon": 8,
    "hash": "2fb2d5a7fa0d"
  },
  "street": {
    "nq": 13,
    "nv": 12,
    "ncon": 8,
    "hash": "94b88340c03c"
  },
  "downhill": {
    "nq": 13,
    "nv": 12,
    "ncon": 8,
    "hash": "0a2d6d02fe8d"
  }
}
```

## Stage 03 / mujoco

Status: **PASS**

| Check | Result |
| --- | --- |
| `truck_finite` | PASS |
| `truck_lean_response` | PASS |
| `reduced_finite` | PASS |
| `reduced_lean_response` | PASS |

```json
{
  "truck": {
    "deck_roll_rad": 0.04134932620308689,
    "steer_rad": {
      "front": -0.040896043785414934,
      "rear": 0.04089604378541441
    }
  },
  "reduced": {
    "deck_roll_rad": 0.009433505133750061,
    "steer_rad": {
      "front": -0.01471327977342364,
      "rear": 0.01471327977342364
    }
  }
}
```

## Stage 04 / mujoco

Status: **PASS**

| Check | Result |
| --- | --- |
| `contact_exists` | PASS |
| `normal_force_positive` | PASS |
| `pull_reaches_contact` | PASS |
| `finite_state` | PASS |

```json
{
  "diagnostics": {
    "contacts": 14,
    "normal_force_N": 15.848193721158294,
    "tangential_force_N": 9.999386055098306,
    "relative_slip_speed_m_s": 0.0034229386625247715
  }
}
```

## Stage 05 / mujoco

Status: **PASS**

| Check | Result |
| --- | --- |
| `29_joint_mapping` | PASS |
| `finite_state` | PASS |
| `pose_torque_path` | PASS |
| `bfm_checkpoint` | PASS |
| `bfm_nonzero_action` | PASS |
| `bfm_torque_injected` | PASS |

```json
{
  "pose": {
    "qfrc_applied_norm": 7.892138574975951e-07,
    "physics_steps": 20
  },
  "policy": {
    "obs_shape": [
      721
    ],
    "policy_counter": 2
  }
}
```

## Stage 06 / mujoco

Status: **PASS**

| Check | Result |
| --- | --- |
| `separate_finite` | PASS |
| `separate_penetration` | PASS |
| `push_start_finite` | PASS |
| `push_start_penetration` | PASS |
| `steer_start_finite` | PASS |
| `steer_start_penetration` | PASS |
| `rider_policy` | PASS |

```json
{
  "separate": {
    "penetration_min_m": 0.0,
    "left_deck": 0,
    "right_deck": 0
  },
  "push_start": {
    "penetration_min_m": -0.00020000007160971955,
    "left_deck": 0,
    "right_deck": 4
  },
  "steer_start": {
    "penetration_min_m": -0.0001999999990290567,
    "left_deck": 4,
    "right_deck": 4
  }
}
```

## Stage 07 / mjlab

Status: **PASS**

| Check | Result |
| --- | --- |
| `cuda_physics` | PASS |
| `step_advances_time` | PASS |
| `partial_reset_untouched` | PASS |
| `partial_reset_target_restored` | PASS |
| `finite_state` | PASS |
| `rider_pose` | PASS |
| `rider_policy` | PASS |

```json
{
  "num_envs": 64,
  "device": "cuda:0",
  "step_time": 0.02200000174343586,
  "pre_reset_time": 0.020000001415610313,
  "rider_pose": {
    "qfrc_applied_norm": 89.74133089940524,
    "state_change": 0.17077340185642242
  },
  "rider_policy": {
    "policy_counter": 2,
    "raw_action_norm": 0.91979587071096,
    "qfrc_applied_norm": 85.52830942581065
  }
}
```

## Stage 09 / mujoco

Status: **PASS**

| Check | Result |
| --- | --- |
| `parameter_ranges` | PASS |
| `parameter_response` | PASS |
| `finite_state` | PASS |
| `each_board_parameter_applies` | PASS |
| `same_seed_reproducible` | PASS |
| `synthetic_calibration` | PASS |
| `each_policy_parameter_applies` | PASS |

Disclosure: synthetic calibration only; no real-data sim2real claim

```json
{
  "nominal_roll": -0.003176641362980832,
  "random_roll": -0.00891804975594812,
  "parameters": {
    "deck_mass_scale": 1.1035224378393704,
    "truck_mass_scale": 0.9321946182338214,
    "wheel_mass_scale": 1.0115489094731385,
    "truck_stiffness_scale": 0.9208479649985628,
    "truck_damping_scale": 0.828965178137061,
    "wheel_spin_damping_scale": 0.9429120187945992,
    "bearing_torque_scale": 0.9640822495468042,
    "friction_scale": 1.0668058650993884,
    "pd_kp_scale": 1.0945543482053146,
    "pd_kd_scale": 0.8061158626386846,
    "deck_com_offset_x": -0.00983836344829443,
    "deck_com_offset_y": 0.002082917091278603,
    "observation_noise_std": 0.0008373669468714318,
    "control_delay_steps": 2.0
  },
  "reset_count": 100,
  "parameter_effects": {
    "deck_mass_scale": true,
    "truck_mass_scale": true,
    "wheel_mass_scale": true,
    "truck_stiffness_scale": true,
    "truck_damping_scale": true,
    "wheel_spin_damping_scale": true,
    "bearing_torque_scale": true,
    "friction_scale": true,
    "deck_com_offset_x": true,
    "deck_com_offset_y": true
  },
  "nominal_sim_time": 0.16000000000000011,
  "random_sim_time": 0.16000000000000011,
  "calibration": {
    "method": "cem",
    "train_loss": 0.0002504271770419263,
    "heldout_loss": 9.156457415364969e-05,
    "baseline_heldout_loss": 0.02460175449189152,
    "truth_parameter_rmse": 0.17395434842847657,
    "experiment": "release",
    "heldout_trials": [
      "synthetic-004",
      "synthetic-005"
    ]
  },
  "policy_parameter_effects": {
    "observation_noise_std": true,
    "control_delay_steps": true,
    "pd_kp_kd_scale": true
  }
}
```

## Stage 09 / mjlab

Status: **PASS**

| Check | Result |
| --- | --- |
| `batch_parameter_diversity` | PASS |
| `partial_reset_state_isolated` | PASS |
| `partial_reset_parameters_isolated` | PASS |
| `runtime_parameter_response` | PASS |
| `finite_state` | PASS |

Disclosure: mjlab randomized drag/loss terms are explicit runtime forces; no real-data claim

```json
{
  "num_envs": 64,
  "view_env": 0,
  "parameter_summary": {
    "deck_mass_scale": {
      "min": 0.8515735994048839,
      "max": 1.1452021855184629,
      "mean": 1.0135097302388962
    },
    "truck_mass_scale": {
      "min": 0.9016540214388901,
      "max": 1.0989203957917486,
      "mean": 1.0013635596783854
    },
    "wheel_mass_scale": {
      "min": 0.90018029830166,
      "max": 1.0856590008655638,
      "mean": 0.9968247778149926
    },
    "truck_stiffness_scale": {
      "min": 0.709206558088778,
      "max": 1.2945073686460715,
      "mean": 0.9675020580028941
    },
    "truck_damping_scale": {
      "min": 0.7122330287307427,
      "max": 1.2748960508915412,
      "mean": 1.0303226591737598
    },
    "wheel_spin_damping_scale": {
      "min": 0.7583904206792833,
      "max": 1.248761512958377,
      "mean": 1.0243649494753504
    },
    "bearing_torque_scale": {
      "min": 0.7636782474216466,
      "max": 1.2459149915222947,
      "mean": 1.0166468494096248
    },
    "friction_scale": {
      "min": 0.7033180940120863,
      "max": 1.273032016078411,
      "mean": 0.9981898148772341
    },
    "pd_kp_scale": {
      "min": 0.8030711594055874,
      "max": 1.1987588470603445,
      "mean": 0.9989412594069409
    },
    "pd_kd_scale": {
      "min": 0.8061158626386846,
      "max": 1.1979844972167346,
      "mean": 1.0245097913750696
    },
    "deck_com_offset_x": {
      "min": -0.019502014647047646,
      "max": 0.01914188870634125,
      "mean": -0.00018701772905236512
    },
    "deck_com_offset_y": {
      "min": -0.009397977963099782,
      "max": 0.00979489069459791,
      "mean": -0.0004401350217693752
    },
    "observation_noise_std": {
      "min": 4.073676003176652e-05,
      "max": 0.009966108359617271,
      "mean": 0.005126415146341578
    },
    "control_delay_steps": {
      "min": 0.0,
      "max": 2.0,
      "mean": 0.984375
    }
  },
  "runtime_parameter_effects": {
    "truck_stiffness_damping": true,
    "wheel_loss": true,
    "com_gravity_torque": true
  },
  "reset_ids": [
    1,
    63
  ]
}
```
