# Skate-BFM Training Experiments

This document explains the experiment design. Numerical results and retained
artifacts are listed in [`train_res.md`](train_res.md); dated work is listed
in [`train_log.md`](train_log.md).

## 1. Objective and Current State

Skate-BFM adapts the frozen official BFM-Zero initialization to HUSKY
skateboard dynamics. The training path must:

1. collect synchronized robot-board expert rollouts;
2. convert HUSKY 23DoF motion into the BFM 29DoF MotionLib contract;
3. mix Base/LAFAN and Skate expert sequences;
4. collect HUSKY online replay with the BFM Actor;
5. run the vendored BFM-Zero `FBcprAuxAgent.update()` unchanged;
6. evaluate frozen checkpoints under matched resets.

Current conclusion:

- [x] Data collection, conversion, replay, rewards, termination, native update,
  checkpoint save/reload, and frozen evaluation execute correctly.
- [x] Source physics, robot-board reset, and active 23DoF action mapping are
  explicitly validated.
- [x] The Phase 100k run completed numerically.
- Stable Skate behavior is **not established**: every 100k checkpoint was less
  stable than official BFM0 under the matched 32-episode evaluation.
- Short retraining is **not yet justified**: the post-alignment P0 audit still
  shows large waist saturation and no meaningful tracking-z advantage.

All reported adaptation runs start from the official BFM0
`model.safetensors` SHA256:

```text
33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361
```

## 2. Environment Matrix

| Use | Simulator / model | Rate and execution | Policy / model | Physics |
|---|---|---|---|---|
| Expert collection | HUSKY official MuJoCo scene `husky_sim/upstream/test_scene/mjlab_scene.xml` | 50 Hz policy; 3000 frames = 60 s maximum | `husky_sim/upstream/ckpts/test.onnx`, CPU | Official HUSKY play-time randomization sampled once per rollout |
| Formal online training | `HuskyLiteEnv`, same HUSKY XML | `control_dt=0.02 s`; four independent MuJoCo envs stepped sequentially; batched Actor on GPU | fresh official BFM0, then updated Actor | Current contract restores the exact source rollout physics; no newly sampled online randomization |
| Expert MotionLib | vendored BFM-Zero runtime in `train/scripts/isaac_env/` | 50 Hz Skate MotionLib; sequence length 8 | official MotionLib and expert loader | Kinematic expert loading, no HUSKY rollout |
| Frozen evaluation | `HuskyBfmOnlineEnv`, same HUSKY XML | deterministic mean Actor unless explicitly stochastic | selected checkpoint | same reset and physics contract as training |

The formal Phase 100k run predates the source-physics fix: it injected source
`qpos/qvel` into nominal physics. D1.1 corrected this for the next run and for
the current evaluator; the old 100k result is not retroactively relabeled.

## 3. Data Collection

### 3.1 Command Grid and Rollout Plan

The authoritative configuration is
[`scripts/data_collection/rollout_config.json`](scripts/data_collection/rollout_config.json).

```text
v = {0.50, 0.75, 1.00, 1.25, 1.50}
h = {-0.7, -0.6, ..., 0.0, ..., +0.6, +0.7}
command cells = 5 * 15 = 75
baseline rollouts = 10 rounds * 15 rollouts = 150
baseline repetitions per command cell = 2
parallel workers = 2
target raw duration = 150 min
extra capacity = 4 rounds * 15 = 60 rollouts
maximum nominal capacity = 210 min
plan seed = 20260804
dataset split = train
overwrite = false
```

`v` is the HUSKY forward command. During steer, `h>0` is labeled left,
`h=0` forward, and `h<0` right. The command list is permuted inside each round
using the plan seed; it does not change the command values.

Collection stops when cumulative raw time reaches 150 minutes. A confirmed
fall ends one rollout but is not a worker failure; replacement jobs use the
extra rounds until the raw-duration target is reached.

### 3.2 HUSKY Phase Rule

The labels follow the official HUSKY six-second cycle:

```text
phase_value = (frame mod 300) / 300

push:        [0.00, 0.40) = 120 frames = 2.4 s
push2steer:  [0.40, 0.50) =  30 frames = 0.6 s
steer:       [0.50, 0.95) = 135 frames = 2.7 s
steer2push:  [0.95, 1.00) =  15 frames = 0.3 s
```

`steer` is split into `steer_left`, `steer_forward`, or `steer_right` from
the command sign. Board yaw is recorded as a diagnostic but does not decide
the steer class.

Fall detection is shared by collection and online training:

```text
fall_candidate =
    (root_tilt > 70 deg)
    OR (root_height < 0.45 m AND illegal_body_contact)

fall = fall_candidate persists for 0.2 s = 10 frames at 50 Hz
```

Temporary feet-off-board and board separation alone are not falls. When a
fall is confirmed, the confirmation frames are relabeled `fall` and the
rollout stops.

### 3.3 Per-rollout Physics Randomization

One seed is derived for each rollout and the following values are sampled once:

| Quantity | Sampling range |
|---|---|
| Robot torso COM offset `(x,y,z)` | `[-0.025,0.025]`, `[-0.025,0.025]`, `[-0.03,0.03]` m |
| Skateboard COM offset `(x,y,z)` | `[-0.02,0.02]`, `[-0.02,0.02]`, `[-0.01,0.01]` m |
| Robot sliding-friction scale | `[0.3,1.6]` per robot geom |
| Deck sliding-friction scale | `[0.8,2.0]` |
| Foot sliding friction | `[0.3,1.8]` per foot geom |
| Wheel rolling-friction scale | `[0.8,1.6]` per wheel geom |
| Initial robot joint offset | `[-0.01,0.01]` rad per joint |

No external push or observation corruption is added. The sampled values and
seed are stored in each rollout metadata file.

### 3.4 Recorded Raw State

Every 50 Hz row contains full `qpos/qvel`, 23D source action, robot root,
23D joint pose/velocity, robot body pose/velocity, skateboard root and joint
state, command `(v,h)`, `phase_id`, `phase_value`, fall/reset flags, and board
heading delta. Arrays must be frame-aligned, finite, and consistent with
metadata `nq/nv`, joint order, quaternion order, XML, and physics seed.

The source timing contract is:

```text
state[t] contains the effect of action[t]
the policy output recorded next is action[t+1]
action[t+1] produces state[t+1]
```

This alignment is required when a source transition action is compared with a
BFM action.

## 4. Dataset Construction

### 4.1 Phase Dataset

[`convert_phase.py`](scripts/data_collection/convert_phase.py) scans each
rollout in order:

1. locate maximal contiguous runs with one `phase_id`;
2. discard every `fall` run;
3. remove `0.15 s` (8 frames at 50 Hz) before the first fall;
4. split at reset frames and discard the reset row;
5. reject segments shorter than `seq_length + 1 = 9` frames;
6. require every retained row to have the same non-fall phase;
7. preserve the exact source start/end frame and physics provenance.

Motion duration follows the MotionLib transition convention:

```text
duration(segment with T frames) = (T - 1) / 50
```

Actual Phase result:

| Phase | Motions | Frames | Duration (s) | Min / median / max frames |
|---|---:|---:|---:|---:|
| push | 1,522 | 182,296 | 3,615.48 | 29 / 120 / 120 |
| push2steer | 1,516 | 45,471 | 879.10 | 21 / 30 / 30 |
| steer_left | 685 | 91,205 | 1,810.40 | 11 / 135 / 135 |
| steer_forward | 109 | 14,670 | 291.22 | 90 / 135 / 135 |
| steer_right | 717 | 96,314 | 1,911.94 | 23 / 135 / 135 |
| steer2push | 1,489 | 22,335 | 416.92 | 15 / 15 / 15 |

The converter discarded 330 fall frames, 240 pre-fall margin frames, and six
too-short segments. No rollout, alignment, conversion, or finite-value error
was rejected.

### 4.2 Continuous Dataset

[`convert_continuous.py`](scripts/data_collection/convert_continuous.py) uses
the same 158 raw rollouts. It:

1. removes fall rows and the same `0.15 s` pre-fall margin;
2. splits valid intervals at reset rows;
3. partitions each interval into non-overlapping 500-frame windows;
4. drops the remainder shorter than 500 frames;
5. allows normal phase transitions inside a window;
6. rejects any window crossing fall or reset.

```text
clip length = 500 frames = 10.0 s
stride = 500 frames
overlap = 0
```

The result is 890 clips, 445,000 frames, and 148.333 minutes. All 890 clips
contain normal phase transitions; the mean is 7.317 phase runs per clip.
Discarded rows are 7,310 tail, 245 pre-fall margin, and 330 fall frames.

### 4.3 HUSKY Pose to BFM MotionLib

For each shared joint `j`, name mapping is exact:

```text
q_bfm[j,t] = q_husky[index(j),t]
```

The six BFM wrist joints absent from HUSKY are:

```text
q_bfm[wrist,t] = 0
```

For root quaternion `q_root` recorded as `wxyz`:

```text
q_xyzw = reorder(normalize(q_root))
r_root = Log_SO(3)(q_xyzw)

pose_aa[t,0] = r_root
pose_aa[t,j+1] = q_bfm[j,t] * axis_bfm[j]
root_trans_offset[t] = root_position[t]
```

The output follows the official record fields:

```text
root_trans_offset [T,3]
pose_aa           [T,30,3]
dof               [T,29]
root_rot          [T,4]  # xyzw
smpl_joints       [T,24,3] = 0 placeholder
fps               = 50
```

Raw 23D action, board state, phase, command, source frame range, and physics
seed are retained as extra aligned fields; they are not silently converted
into BFM normalized actions.

### 4.4 Dataset Validation

- [x] Phase and Continuous records load through official `MotionLibRobot`.
- [x] Official expert loader produces finite current/next `state [16,64]`.
- [x] Seq8 sampling passes for both datasets.
- [x] Every retained motion resolves to its original raw rollout and frame
  range.
- [x] Phase QC renders 10 random examples for each of six phases; Continuous
  QC renders 10 random clips, seed `20260813`.
- A held-out validation/test collection is pending; `dataset_split=train` is
  the only formal raw split collected so far.

## 5. BFM Observation, Action, and Expert Mapping

### 5.1 Online Observation

HUSKY 23D joint arrays are expanded by name to 29D with zero wrists. The
official online BFM state is:

```text
s_t = [
    q_29(t) - q_default,       # 29
    qdot_29(t),                # 29
    gravity_body(t),           # 3
    0.25 * omega_body(t)       # 3
] in R^64
```

The Actor also receives:

```text
last_action = 5 * a_bfm(t-1) in R^29
history_actor in R^372
```

History contains four frames of action (29), angular velocity (3), joint
position (29), joint velocity (29), and projected gravity (3):

```text
4 * (29 + 3 + 29 + 29 + 3) = 372
```

Formal reset uses zero `last_action` and zero history. After each step, only
the current online state/action enters history; source expert actions are not
injected.

The 463D privileged state is built in the robot heading frame:

```text
p = [
    root_height,                       # 1
    relative body positions,           # 30 * 3
    body tangent and normal vectors,   # 31 * 6
    local body linear velocities,      # 31 * 3
    local body angular velocities      # 31 * 3
] in R^463
```

### 5.2 MotionLib Expert Latent

For one eight-frame expert sequence, the official backward map is evaluated
on normalized next observations:

```text
b_t = B(N(o_expert,t+1))
z_expert = project_z((1/8) * sum_{t=0}^{7} b_t)
||z_expert||_2 = sqrt(256) = 16
```

The same `z_expert` is repeated over the eight sequence rows. Each native
update samples 64 complete Base/LAFAN sequences and 64 complete Skate
sequences, giving:

```text
128 sequences * 8 rows = batch size 1024
```

The Base source contains 862 LAFAN motions. The formal Skate source is selected
as either 6,038 Phase motions or 890 Continuous clips; the completed 100k run
used Phase.

Known upstream asymmetry: MotionLib expert `last_action` is zero, and expert
root angular velocity is not multiplied by the online `0.25` scale. This was
measured in D1.3 and remains a documented compatibility issue.

### 5.3 Online BFM Action to HUSKY Control

The Actor emits normalized `a_bfm in [-1,1]^29`. Six wrist dimensions are
forced to zero, then the 23 shared joints are selected by name. The physical
target used by the online HUSKY controller is:

```text
q_target_bfm[j] =
    q_default_bfm[j] + 5 * scale_bfm[j] * a_bfm[j]
```

Replay stores the projected 29D BFM action; MuJoCo executes the mapped 23D
action. Shared joint values are not reordered by numeric index assumptions.

### 5.4 Source Expert Action to BFM Action

The collected ONNX source policy uses:

```text
q_target_src[j] =
    q_default_src[j] + scale_src[j] * a_src[j]
```

Equating physical targets gives the exact diagnostic bridge:

```text
a_bfm_eq[j] =
    (q_default_src[j] + scale_src[j] * a_src[j]
     - q_default_bfm[j])
    / (5 * scale_bfm[j])
```

```text
EXACT:     |a_bfm_eq[j]| <= 1
PROJECTED: a_bfm_bridge[j] = clip(a_bfm_eq[j], -1, 1)
```

The bridge reconstructs representable physical targets with RMSE
`1.97e-17 rad` and maximum error `4.44e-16 rad`. Across 452,885 source frames,
component coverage is `99.600839%`, but full-frame exact coverage is only
`92.1503%`; the remaining tail is almost entirely left/right hip pitch.

This bridge is used for diagnostics and expert-action translation only. It is
not fed into the formal Actor rollout or history.

## 6. Online Reset and Latent Lifecycle

### 6.1 Physical Reset

The reset sampler first chooses a MotionLib record uniformly, then a local
frame uniformly within that record:

```text
m ~ Uniform(motions)
t ~ Uniform({0, ..., frames(m)-1})
```

It loads `qpos[t]`, `qvel[t]`, robot and skateboard together, validates source
XML/joint/quaternion metadata, restores the recorded physics realization, runs
`mj_setConst`, and injects the exact state. Missing or ambiguous provenance
fails closed.

### 6.2 Current Formal Latent Rule

With four environments and seed 4728, expert rollout slots are `[0,3]`;
slots `[1,2]` are free. Every slot owns a background latent refreshed every
100 episode steps:

```text
z_background =
    z_buffer.sample(), if the buffer is non-empty
    model.sample_z(),  otherwise
```

For an expert slot reset at motion frame `t`, tracking latent step `k` is:

```text
z_track[k] =
  project_z(
    mean_{i=k+1}^{min(k+8, trajectory_end)}
      B(N(o_expert,t+i))
  )
```

Tracking lasts at most 250 steps and never crosses a Phase segment. After it
ends, the slot returns to its maintained background latent. Replay stores the
effective latent actually passed to Actor.

The historical Phase 100k run used random background z only. The mixed,
reset-aligned rule was added after that failed run and has only been tested in
frozen preflight, not retrained.

## 7. Native BFM-Zero Optimization

The project calls the vendored
[`FBcprAuxAgent.update()`](scripts/isaac_env/humanoidverse/agents/fb_cpr_aux/agent.py);
it does not reimplement component optimizers.

### 7.1 Latent Mixture

Each update samples:

```text
20% z_goal   = project_z(B(next online goal))
60% z_expert = encoded Base/Skate expert sequence
20% z_random = uniform model latent
```

With probability `0.8`, a replay transition latent is relabeled by this
mixture before the update. The z-buffer capacity is 8,192.

### 7.2 FB Objective

For batch embedding matrices:

```text
M = F(s,a,z) B(g)^T
M_target = F_target(s',a',z) B_target(g)^T
D = M - gamma * M_target

L_FB =
    0.5 * sum_offdiag(D^2) / number_offdiag
    - E * mean_diag(D)
    + 100 * L_ortho(B)

L_ortho(B) =
    0.5 * sum_offdiag((B B^T)^2) / number_offdiag
    - mean_diag(B B^T)
```

`E=2` is the number of parallel F maps and `gamma=0.98`; the optional Q term
is disabled (`q_loss_coef=0`). F/B target networks use Polyak coefficient
`tau_FB=0.01`.

### 7.3 Discriminator, Critics, and Actor

The discriminator separates expert `(o,z_expert)` from online `(o,z)`:

```text
L_D = -log(sigmoid(D_expert))
      + softplus(D_online)
      + 10 * gradient_penalty
```

The main and auxiliary critic targets are:

```text
y_D   = r_discriminator + gamma * (Qbar_mean - 0.5 * Qbar_uncertainty)
y_aux = r_aux_normalized + gamma * (Qaux_mean - 0.5 * Qaux_uncertainty)
```

The Actor objective is:

```text
Q_FB = <F(s, pi(s,z), z), z>
w = stop_gradient(mean(|Q_FB|))

L_actor =
    -mean(Q_FB)
    -0.05 * w * mean(Q_discriminator)
    -0.02 * w * mean(Q_aux)
```

### 7.4 Optimizer and Model Parameters

| Module | Architecture and inputs |
|---|---|
| Forward F | residual MLP, 6 hidden layers of 2,048, two parallel maps; state, privileged state, last action, history |
| Backward B | one hidden layer of 256; state and privileged state |
| Actor | residual MLP, 6 hidden layers of 2,048; state, last action, history |
| Main / auxiliary critic | residual MLP, 6 hidden layers of 2,048, two parallel maps; same inputs as F |
| Discriminator | 3 hidden layers of 1,024; state and privileged state |

| Component | Learning rate |
|---|---:|
| Forward map F | `3e-4` |
| Backward map B | `1e-5` |
| Actor | `3e-4` |
| Discriminator | `1e-5` |
| Main critic | `3e-4` |
| Auxiliary critic | `3e-4` |

Other fixed parameters:

```text
latent dimension = 256; latent norm = 16
sequence length = 8; batch size = 1024
actor std = 0.05; sampled std clip = 0.3
critic target tau = 0.005
FB target tau = 0.01
weight decay = 0
AMP = false; compile = false; cudagraphs = false
```

### 7.5 Auxiliary Reward

The environment records positive raw penalties; the agent forms:

```text
r_aux = sum_i weight_i * penalty_i
```

| Penalty | Definition | Weight |
|---|---|---:|
| action rate | `sum((a_t-a_{t-1})^2)` on executed 23D action | `-0.1` |
| feet orientation | contact-gated world-horizontal foot-normal error | `-0.4` |
| ankle roll | left/right ankle-roll square | `-4.0` |
| DoF position limit | distance outside 95% HUSKY joint range | `-10.0` |
| slippage | foot tangential velocity relative to contacted ground/board | `-2.0` |
| undesired contact | illegal body-ground/board contact above 1 N | `-1.0` |
| torque square | `sum(qfrc_actuator^2)` | `0.0` |
| torque limit | excess above 95% runtime HUSKY torque limit | `0.0` |

Position/torque authority is the actual HUSKY `MjModel`; reward semantics come
from vendored BFM-Zero. No forward-speed, command, balance, or skateboard
displacement reward has been added.

### 7.6 Formal Schedule

```text
online transitions = 100,000
parallel HUSKY envs = 4
replay capacity = 100,000 on CPU
warmup = 1,024 stochastic pretrained-Actor transitions
first update = transition 1,500
update interval = 500 transitions
native updates per block = 50
blocks = 198
total native updates = 9,900
checkpoints = 20k, 50k, 100k
episode horizon = 1,024 transitions
```

## 8. Experiment Sequence and Conclusions

### E1. Expert Integration and Sampling

Environment: official MotionLib loader plus one HUSKY sample, no optimizer.

- [x] Base-only and Skate-only MotionLib loading passed.
- [x] Complete Seq8 Base/Skate 50/50 sampling passed.
- [x] Frozen B produced finite `[1024,256]` expert latent rows with norm 16.
- Conclusion: the data schema can enter BFM training; this does not validate
  physical Skate behavior.

### E2. B/F-only Boundary

Environment: 1,024 seen-dynamics HUSKY transitions and independent
1/10/100-update runs from official BFM0. Evaluation used protocol seed
`20260810`, keyframe-0 robot/board reset, `dt=0.02 s`, and a 128-step horizon:

| Split | Dynamics seed | Command `(v,h)` |
|---|---:|---:|
| seen-1 | 22001 | `(0.75,-0.35)` |
| seen-2 | 22002 | `(1.25,+0.35)` |
| unseen-1 | 23001 | `(0.75,+0.35)` |
| unseen-2 | 23002 | `(1.25,-0.35)` |

- Base-only and correctly configured Base+Skate 50/50 treatments were
  compared while Actor, discriminator, QD, and Qaux were frozen.
- Training consumed only seen-dynamics replay; unseen and evaluation
  transitions in training were both zero.
- At 100 updates, Base+Skate Top-5 changed `0.6875 -> 0.8281` and mean rank
  `5.2813 -> 3.8125`.
- Conclusion: promising representation adaptation under this evaluator, but
  no Base retention or physical task-success claim.

### E3. Reward, Termination, and Native-update Readiness

Environment: fixed 1,024-transition HUSKY replay, one Phase-limited Skate
expert, official BFM0 initialization.

- [x] Eight auxiliary reward fields were finite `[1024,1]`.
- [x] Fall termination produced 14 terminal, one horizon-truncated, and 1,009
  normal transitions with zero overlap.
- [x] Full native updates at 1, 10, and 100 iterations remained finite; all
  six Adam optimizers reached the expected step.
- Conclusion: the native update dependencies are executable, but fixed-replay
  numerical stability is not policy-quality evidence.

### E4. 20k Closed-loop Baseline

Environment: one nominal HUSKY env, random model latents, 20,000 transitions,
1,900 native updates, no domain randomization.

- [x] 19,592 normal, 389 falls, and 19 horizon truncations were recorded.
- [x] 10k/20k checkpoints reloaded with complete optimizer state.
- All 60 fixed target-conditioned evaluation episodes fell before 128 steps.
- Conclusion: training executed, but physical performance was inconclusive.

### E5. Phase 100k Formal Run

Environment: four nominal HUSKY envs, Phase expert resets, random latent
refresh every 100 transitions, source `qpos/qvel` but pre-D1 nominal physics.

- [x] 100,000 transitions, 198 blocks, and 9,900 updates completed.
- [x] All 537 model tensors and 846,227,305 values were finite.
- Block-final FB loss changed from `325,215.56` at step 1,500 to
  `-10,048.30` at 100k, while critic and auxiliary-critic losses remained
  finite; the negative value is allowed by the FB diagonal objective.
- Every 3,109 completed training episode ended in fall; no episode reached the
  1,024-step horizon.
- Frozen mean survival was `1.264 s` for official BFM0 versus
  `0.604/0.519/0.643 s` for 20k/50k/100k.
- Conclusion: numerical integrity passed; behavioral adaptation failed.

### E6. Post-failure Alignment Audits

Environment: no training unless explicitly stated; official checkpoint and
historical 100k checkpoint evaluated separately.

- D1.1 restored exact recorded source physics with exact raw qpos/qvel.
- D1.2 enforced the shared 23DoF action subspace and zero wrists throughout
  Actor, replay, F, critics, and history.
- D1.3 found waist position/velocity drift of approximately 20-27 normalized
  standard deviations within 20 steps and the expert/online angular-velocity
  scale asymmetry.
- D2.1 constructed same-reset tracking z. On the old 100k checkpoint,
  step-20 root tilt improved `47.8 -> 25.2 deg` and saturation
  `17.1% -> 0.29%`; official BFM0 waist divergence remained.
- D2.3/D2.4 identified hip-pitch control-range mismatch and moderate temporal
  context loss; Phase strict five-action context coverage was `64.8366%`,
  with `steer2push=12.1379%`.
- D2.5 retained exact affine translation plus explicit projection. A
  diagnostic 4,096-transition hip-tail refinement changed normalized
  next-state RMSE `0.4689 -> 0.4482`, but held-out ratio `0.9610` was too weak
  and inconsistent to adopt a learned correction.
- Conclusion: source physics and interfaces are now explicit; hip control
  range, temporal context, upstream observation asymmetry, and frozen policy
  takeover remain unresolved behavioral factors.

### E7. M2.6-P0 Frozen Closed-loop Preflight

Environment: fresh official BFM0, Phase dataset, 512 matched resets, seed
4728, 51-step horizon, exact source physics, no update/backward/optimizer.

```text
FORMAL_D2.2:
  expert slots [0,3] use aligned tracking z
  free slots [1,2] use background z

PURE_RANDOM_Z:
  all slots use background random z
```

- [x] qpos/qvel reset errors were zero; physics mismatch count was zero.
- [x] Model, normalizer, and buffer hashes were unchanged.
- [x] No failures occurred by step 20 in either condition.
- `FORMAL_D2.2` versus random mean survival was `49.816` versus `49.705`
  steps; failure by step 50 was `26.17%` versus `26.76%`.
- Waist action saturation `|a|>=0.95` was `82.33%` versus `82.69%`.
- The reset population contained 421 fully exact and 91 projected source
  action contexts. Under `FORMAL_D2.2`, Actor/expert first-action cosine was
  `0.435` for exact contexts and `0.185` for projected contexts; first
  physical-target jump mean/p95 was `2.129/3.080`.
- Board/robot-relative reset variables had weak survival association; the
  largest absolute reported Spearman coefficient was `0.136`.
- Conclusion: structural contract passed, but aligned tracking-z did not
  materially improve frozen behavior. The source-to-BFM takeover action is
  less aligned in projected contexts, but no single board variable explains
  failure. Classification is `BEHAVIORAL_DIAGNOSTIC_REQUIRED`.

## 9. Verified and Pending Conclusions

Verified:

- [x] Formal raw data is synchronized and reproducible from recorded seeds.
- [x] Phase and Continuous split rules are deterministic and provenance-safe.
- [x] HUSKY pose can be represented in the BFM29 MotionLib schema with six
  fixed wrists.
- [x] Most source actions have an exact BFM physical-target equivalent, with
  explicit projection for the hip-pitch tail.
- [x] Native FB-CPR-Aux optimization and checkpointing run without numerical
  corruption.
- [x] The existing Phase 100k checkpoint is behaviorally worse than official
  BFM0 under the matched frozen evaluation.

Pending:

- a separate validation/test raw collection;
- a resolved expert/online angular-velocity observation contract;
- a decision for hip-pitch range and low-level controller mismatch;
- periodic frozen-policy curves and parameter/gradient norm tracking;
- a successful short retrain under the post-D1/D2 contracts;
- evidence that the learned model is a stable, diverse Skate motion library.

## 10. Source Index

- Collection: [`scripts/data_collection/rollout_split.py`](scripts/data_collection/rollout_split.py)
- Phase conversion: [`scripts/data_collection/convert_phase.py`](scripts/data_collection/convert_phase.py)
- Continuous conversion: [`scripts/data_collection/convert_continuous.py`](scripts/data_collection/convert_continuous.py)
- Formal trainer: [`scripts/train_skate_bfm.py`](scripts/train_skate_bfm.py)
- Frozen evaluator: [`scripts/evaluator.py`](scripts/evaluator.py)
- Observation/action bridge: [`../src/skate_bfm/integration/`](../src/skate_bfm/integration/)
- HUSKY runtime: [`../husky_sim/src/skate_husky/`](../husky_sim/src/skate_husky/)
- Formal Skate datasets:
  [Hugging Face](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset)
