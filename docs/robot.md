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
velocity and previous policy command; history is updated once per 20 ms policy
boundary. State history contains the current frame and three earlier frames,
newest first. Previous command is `5 * clip(previous_raw_action, -1, 1)`, even
when actuator delay has not yet applied that command. Initial history slots
are zero-padded. Episode reset clears history, command, delay queue and the
per-environment tracking cursor, then returns a fresh observation preview
without advancing any of those states. This explicit episode-reset behavior
is a simulation contract; the pinned upstream deploy reset callback is empty.
The reward latent is `move-ego-0-0` from
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
this project. The 2026-11-02 correction and validation are recorded in
[Stage 05](stages/05.md). Earlier claims of 30-second fixed-file runs were not
supported by those original event logs; use the new complete NPZ records.

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

Gravity uses `R_body_to_world.T @ [0, 0, -1]`. MuJoCo/Warp `cvel` angular
velocity is transformed by that same inverse body rotation before applying
the deploy scale `0.25`. Sensor noise is confined to current measured-state
fields and their history; latent and remembered commands are not perturbed.

Tracking follows the pinned official `walking.yaml`: a future window of up to
three context vectors, discounted by `gamma=0.8`, averaged and normalized to
the first context's norm. The window is truncated, not wrapped, at the sequence
end. The simulation automatically starts once on launch/reset. With the public
13,064-frame sequence it computes index 2000 once, then holds the index-0
window until reset. Each batch environment owns its cursor and active flag.
Goal mode uses the explicitly named public entry without substitution.

Independent numerical checks use the pinned official configuration bytes in
`src/skate_sim/validation/reference05/`, with recorded source URLs and SHA-256.
The selected ONNX actor consumes 721 inputs; the training encoder/critic's
privileged features are not appended to this actor input. Deploy history timing
is tested independently of the different timing in a reference training base
class; the checkpoint's exact training path is not inferred from that class.

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
