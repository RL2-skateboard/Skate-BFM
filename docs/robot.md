# Robot contract

Stage 05 uses the 29-DoF contract in `src/skate_sim/robot/contract.py` and the
public deploy configuration in `configs/bfm/motivo_newG1.yaml`.

The contract provides:

- exactly 29 unique joints, including all six wrist DoF;
- 2 ms physics and 20 ms control timing, decimation 10;
- per-physics-substep PD torque computation and clipping;
- resettable history and last-action state;
- real q/dq, base angular velocity and body-frame gravity fields.

The public BFM-Zero ONNX policy is stored outside git:

```text
checkpoints/bfm-zero/model/exported/FBcprAuxModel.onnx
```

Its verified interface is:

```text
actor_obs: [1, 721] float32
action:    [1, 29] float32
```

The adapter follows the public deploy structure: 93 current observation fields,
372 newest-first field-major history fields and a 256-dimensional latent. Current
fields are real q minus default, dq, body-frame projected gravity, base angular
velocity and previous action; history is reset-padded and updated once per 20 ms
policy boundary. The reward latent is `move-ego-0-0` from
`reward_locomotion.pkl`. Official context modes are also
available: `--policy-task tracking` uses public `zs_walking.pkl`; `--policy-task
goal --latent-name NAME` uses a public goal entry. The public goal list has no
T-pose entry, so a T-pose test uses the independent `--controller pose` PD
baseline, not BFM. Official source commits:

```text
deploy:           de9e248ce8b315508aafaecc8f15def3e057fccc
minimal_inference: df9c4a9633aa155c727419e82c906010b1e1845f
```

The checkpoint is CC BY-NC 4.0 material from BFM-Zero and is not committed to
this project. A 30-second MuJoCo ground smoke loop has run with non-zero ONNX
actions. This validates loading, shape, ordering, timing and recording; it does
not claim stable locomotion or skateboarding skill in a new task.

Downloaded artifacts and SHA-256:

```text
FBcprAuxModel.onnx 209097902c45621eebab2edb81070c31895fcdd558f0cf6f7f5a360fd747ab74
config.json         3bc2c5c0c260cc89f9c4af20add424304f84899717505f5b383c33d22a23b16e
model_config.json   aee5e94fd089979cb31d1f1a5fa4905e926398231d46dc8eb2047e25b8d2935b
```

The ONNX runtime dependency is installed in `skate-sim-mjlab` as
`onnxruntime`. The official source is downloaded from the public BFM-Zero
Hugging Face model repository using the deploy/minimal-inference source
versions listed above.

For every control update the action path is:

```text
raw_action -> clip[-1,1] -> action_rescale(5)
            -> per-joint action_scale -> default_joint_pos
            -> position-limit clip -> q_target
```

Every 2 ms physics step computes
`tau = clip(kp * (q_target - q) - kd * dq, -effort_limit, effort_limit)`
from the current state and writes those 29 values to `qfrc_applied`. The
official free-base G1 starts at the source pelvis height with MuJoCo `wxyz`
quaternion `(1,0,0,0)`. All 29 joints are looked up by official name, including
wrist roll, pitch and yaw.

Rider contact entities use the official `left_ankle_roll_link` and
`right_ankle_roll_link` bodies. Their descendant `dummy_lf_*`/`dummy_rf_*`
finite sphere geoms are the deploy foot contact proxies; ankle-pitch meshes are
not treated as sole bodies. The rider board is the procedural `deck` body with
`deck_strip_*` collision geoms, and `ground` is the single rider ground plane.

Policy command:

```bash
python play.py --backend mujoco --scene robot --controller policy \
  --policy-task reward --latent-name move-ego-0-0 \
  --checkpoint checkpoints/bfm-zero --duration 30 \
  --record runs/05_bfm_reward
```
