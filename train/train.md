# Skate-BFM Training Experiments

This document contains three conducted/current project experiments and one
planned next experiment. Every audit, preflight, smoke run, and diagnostic
belongs to its parent experiment; it is not a separate experiment.

Numerical results are in [`train_res.md`](train_res.md), and dated work is in
[`train_log.md`](train_log.md).

## Experiment 1: Training Workspace and BFM-HUSKY Integration

### Experiment Goal

Create the `train` branch and a self-contained training workspace that connects
the official BFM-Zero model/runtime to the HUSKY skateboard MuJoCo
environment. This experiment establishes the software and tensor contracts
needed by later data collection and training; it does not train a Skate model.

### Environment and Inputs

| Item | Setting |
|---|---|
| Project environment | Conda `skatebfm`, Python 3.12 |
| BFM runtime | vendored under `train/scripts/isaac_env/` |
| HUSKY runtime | `husky_sim/src/skate_husky/` |
| HUSKY scene | `husky_sim/upstream/test_scene/mjlab_scene.xml` |
| Official BFM0 checkpoint | `model/bfm-zero-official/` |
| BFM0 checkpoint SHA256 | `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361` |
| BFM action width | 29 |
| HUSKY physical action width | 23 |
| BFM latent width | 256 |

**Caption.** Action width is the number of commanded joint coordinates: BFM
uses 29 degrees of freedom (DoFs), while HUSKY has 23 physical actuators.
Latent width is the dimension of the unitless skill vector. `SHA256` is the
checkpoint-content hash used to verify that the same model file is loaded.
The runtime and scene rows identify the code and MuJoCo XML actually executed.

The training runtime is project-owned but keeps the official BFM-Zero agent,
MotionLib, model, and configuration structure. Formal code does not depend on
`model/bfm-zero-source/`.

### Experiment Process and Method

The integration boundary was built in four steps:

1. vendor the complete BFM-Zero HumanoidVerse/Isaac runtime under
   `train/scripts/isaac_env/`;
2. expose the HUSKY MuJoCo state and controller through `HuskyLiteEnv`;
3. map BFM observations/actions to HUSKY by joint name;
4. verify headless stepping, the MuJoCo viewer, MotionLib loading, and native
   BFM forward/backward/update interfaces.

Relevant implementation:

- [`../src/skate_bfm/integration/`](../src/skate_bfm/integration/)
- [`scripts/isaac_env/`](scripts/isaac_env/)
- [`../husky_sim/src/skate_husky/`](../husky_sim/src/skate_husky/)

The action-space relation is:

```text
a_bfm in R^29
    -> zero six unavailable wrist coordinates
    -> select 23 shared joints by name
    -> a_husky in R^23
```

The shared coordinates are copied exactly:

```text
a_husky[j] = a_bfm[index_of_same_joint_name(j)]
```

The six unavailable joints are left/right wrist roll, pitch, and yaw. They are
zero in replay, online history, physical execution, and model-side action
projection.

The base online observation contract is:

```text
state              [64]
privileged_state   [463]
last_action        [29]
history_actor      [372]
latent z           [256]
stored action      [29]
executed action    [23]
```

The 64D state is:

```text
s_t = [
    q_29(t) - q_default,       # 29
    qdot_29(t),                # 29
    gravity_body(t),           # 3
    0.25 * omega_body(t)       # 3
]
```

The 372D Actor history contains four frames of action, angular velocity,
joint position, joint velocity, and projected gravity:

```text
4 * (29 + 3 + 29 + 29 + 3) = 372
```

The 463D privileged state uses the robot heading frame:

```text
privileged_state = [
    root_height,                       # 1
    relative body positions,           # 30 * 3
    body tangent and normal vectors,   # 31 * 6
    local body linear velocities,      # 31 * 3
    local body angular velocities      # 31 * 3
]
```

### Problems and Solutions

| Problem | Method used |
|---|---|
| BFM0 has 29 actions but HUSKY has 23 actuators | Use one name-based mapping and force the six absent wrists to zero. |
| The original BFM source directory will not remain a runtime dependency | Vendor the complete required BFM runtime under `train/scripts/isaac_env/`. |
| HUSKY and BFM use different observation structures | Construct the official 64/463/29/372 observation dictionary from HUSKY state. |
| Viewer and headless execution need the same state/action boundary | Wrap both modes with `HuskyLiteEnv` and one integration adapter. |

**Caption.** The numbers 64/463/29/372 are unitless tensor widths for state,
privileged state, last action, and Actor history. `29D` and `23D` below mean
29- and 23-coordinate action vectors, respectively.

### Verified and Unverified Conclusions

Verified:

- [x] The `train` branch and independent training workspace were created.
- [x] Official BFM0 loads strictly with 537 model tensors.
- [x] BFM29-to-HUSKY23 name mapping preserves all shared actions exactly.
- [x] Headless MuJoCo stepping and the interactive viewer work.
- [x] Base and Skate MotionLib sources load through the official expert loader.
- [x] Native BFM forward/backward and update interfaces accept the integrated
  tensor schema.

Unverified:

- [ ] This integration alone produces stable Skate behavior.
- [ ] The six fixed wrists and current HUSKY controller preserve every
  physical behavior represented by the original BFM0 model.

## Experiment 2: HUSKY Expert Dataset Collection and Construction

### Experiment Goal

Collect synchronized HUSKY robot-skateboard expert trajectories, then build
two BFM-compatible datasets from the same raw source:

- **Phase:** one motion per phase-pure contiguous segment;
- **Continuous:** non-overlapping 10-second clips that retain normal phase
  transitions.

This experiment produces data only. It does not optimize BFM0.

### Environment and Inputs

| Item | Setting |
|---|---|
| Simulator | official HUSKY MuJoCo test scene |
| Source policy | `husky_sim/upstream/ckpts/test.onnx` |
| Source policy device | CPU |
| Record frequency | 50 Hz |
| Maximum episode length | 3,000 frames = 60 s |
| Parallel workers | 2 |
| Dataset split | `train` |
| Plan seed | `20260804` |
| Target raw duration | 150 min |
| Local dataset root | `train/dataset/sim_collected/` |
| Output | Hugging Face `Yak9Ce3teeh/skate-sim-dataset` |

**Caption.** `Hz` means recorded control frames per second; at 50 Hz one frame
spans 0.02 s. An episode is one uninterrupted rollout, capped at 3,000 frames
(60 s). The seed fixes command scheduling and randomization for
reproducibility; `workers` are parallel collection processes, `split=train`
is the dataset partition label, and `min` denotes accumulated simulation
minutes. The local root stores `raw`, `phase`, and `continuous` as sibling
directories; Hugging Face is the authoritative published copy.

The collection command grid is:

```text
v = {0.50, 0.75, 1.00, 1.25, 1.50}
h = {-0.7, -0.6, ..., 0.0, ..., +0.6, +0.7}
```

This gives 75 `(v,h)` cells. Ten rounds with 15 episodes produce 150 baseline
episodes, so every cell is sampled twice. Four extra rounds provide at most
60 fall replacements and 210 nominal minutes of total capacity.

For **collection only**, HUSKY's play-time physics realization is sampled once
per source episode:

| Quantity | Range |
|---|---|
| Robot torso COM `(x,y,z)` | `+/-0.025`, `+/-0.025`, `+/-0.03` m |
| Skateboard COM `(x,y,z)` | `+/-0.02`, `+/-0.02`, `+/-0.01` m |
| Robot friction scale | `[0.3,1.6]` per geom |
| Deck friction scale | `[0.8,2.0]` |
| Foot sliding friction | `[0.3,1.8]` per foot geom |
| Wheel rolling-friction scale | `[0.8,1.6]` per wheel |
| Initial joint offset | `[-0.01,0.01]` rad per joint |

**Caption.** `COM` is center of mass; `(x,y,z)` offsets are sampled in meters
around the nominal model. Friction scales are unitless multipliers of nominal
MuJoCo coefficients: robot/foot rows affect body-ground sliding, deck friction
affects foot-deck contact, and wheel rolling friction affects wheel-ground
motion. Joint offsets are in radians, with positive/negative values applied
relative to the nominal pose. Each interval is the uniform per-episode sample
range, not an observed confidence interval.

The seed and every sampled value are stored with the episode. No external push
or observation corruption is added.

### Experiment Process and Method

#### Raw recording

Every row stores synchronized robot and skateboard `qpos/qvel`, source action,
robot root/joints/bodies, skateboard root/joints, command `(v,h)`, phase,
fall/reset flags, and board heading. All arrays must have the same first
dimension and contain no NaN/Inf.

The source action timing is:

```text
action[t] is the policy output already applied before state[t]
action[t+1] produces state[t+1]
```

#### Phase labeling

The official six-second HUSKY cycle has 300 frames:

```text
push:        [0.00, 0.40) = 120 frames
push2steer:  [0.40, 0.50) =  30 frames
steer:       [0.50, 0.95) = 135 frames
steer2push:  [0.95, 1.00) =  15 frames
```

During steer, `h>0` is left, `h=0` is forward, and `h<0` is right. Board yaw
is recorded for analysis but does not override the command label.

Fall is:

```text
candidate =
    root_tilt > 70 deg
    OR (root_height < 0.45 m AND illegal_body_contact)

fall = candidate persists for 0.2 s = 10 frames
```

Feet temporarily leaving the board or the board separating from the robot is
not sufficient to label a fall.

#### Phase splitting

`convert_phase.py`:

1. finds maximal runs with one phase ID;
2. removes every fall run and 0.15 s (8 frames) before the first fall;
3. splits at reset rows and discards the reset row;
4. rejects segments shorter than 9 frames (`Seq8 + one next frame`);
5. preserves source episode/frame/command/physics provenance.

For a segment with `T` frames:

```text
MotionLib duration = (T - 1) / 50
```

#### Continuous splitting

`convert_continuous.py` removes the same fall/reset boundaries, then divides
each valid interval into:

```text
clip length = 500 frames = 10.0 s
stride = 500 frames
overlap = 0
```

The remainder shorter than 500 frames is discarded. A clip may cross ordinary
phase transitions but cannot cross fall or reset.

#### HUSKY23 pose to BFM29 MotionLib

For each shared joint:

```text
q_bfm[j,t] = q_husky[index_of_same_joint_name(j),t]
```

The six missing wrists are zero. For root quaternion recorded as `wxyz`:

```text
q_xyzw = reorder(normalize(q_root))
r_root = Log_SO(3)(q_xyzw)

pose_aa[t,0] = r_root
pose_aa[t,j+1] = q_bfm[j,t] * axis_bfm[j]
root_trans_offset[t] = root_position[t]
```

The official BFM record contains:

```text
root_trans_offset [T,3]
pose_aa           [T,30,3]
dof               [T,29]
root_rot          [T,4]  # xyzw
smpl_joints       [T,24,3] = 0 placeholder
fps               = 50
```

Raw source action, board state, phase, command, source frame range, and physics
seed remain aligned extra fields.

Relevant implementation and data:

- [`scripts/data_collection/`](scripts/data_collection/)
- [`dataset/`](dataset/)
- [Hugging Face raw/Phase/Continuous dataset](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset)

The linked Phase QC videos replay the robot and skateboard for 10 samples per
phase; Continuous QC shows 10 fixed 500-frame clips. They verify dataset
conversion and visual integrity, not trained-policy performance.

### Problems and Solutions

| Problem | Method used |
|---|---|
| One episode contains push, transitions, and steer | Use official phase clock and contiguous phase-run segmentation. |
| Falls contaminate the preceding motion tail | Remove the confirmed fall and a 0.15 s pre-fall margin. |
| Short phase fragments cannot provide Seq8 next-state samples | Require at least 9 frames. |
| Phase-only records lose longer temporal transitions | Build Continuous 500-frame clips from the same raw data. |
| HUSKY has 23 joints but MotionLib expects 29 | Map shared joints by name and set six wrists to zero. |
| Processed data could lose source identity | Store source NPZ, frame range, command, physics seed, phase, board state, and action. |

**Caption.** `Seq8` is an eight-transition expert sequence and therefore needs
nine state frames. `NPZ` is the raw synchronized NumPy archive; frame ranges
and seeds identify the exact source interval and physics realization.

### Verified and Unverified Conclusions

Verified:

- [x] 158 source episodes produced 452,885 raw frames and 150.962 minutes.
- [x] Phase conversion produced 6,038 motions, 452,291 frames, and 148.751
  minutes.
- [x] Continuous conversion produced 890 clips, 445,000 frames, and 148.333
  minutes.
- [x] No complete source episode was rejected.
- [x] Both datasets load through official `MotionLibRobot` and pass Seq8
  expert sampling.
- [x] Phase QC rendered 10 samples per phase; Continuous QC rendered 10 clips.
- [x] Raw, Phase, Continuous, manifests, and QC are published on Hugging Face.

Unverified:

- [ ] A held-out validation/test split; only `dataset_split=train` was
  formally collected.
- [ ] Equal physical quality across every command cell and phase.
- [ ] Dynamic Skate skill quality; data validity does not prove that every
  expert segment is equally useful for BFM training.

## Experiment 3: BFM + Skate Expert Training and Semantics Alignment

### Experiment Goal

Train BFM0 with Base/LAFAN and Skate expert data, evaluate the resulting
closed-loop policy, then diagnose and resolve the semantics mismatch exposed
by the failed training result. Reward audits, rollout preflights, reset/latent
audits, action audits, and frozen evaluations are all internal steps of this
single experiment.

### Environment and Inputs

| Item | Formal setting |
|---|---|
| Initialization | fresh official BFM0 checkpoint |
| Base expert | `train/dataset/base/`: 862 LAFAN motions |
| Skate expert | `train/dataset/sim_collected/`: Phase 6,038 motions; Continuous 890 clips |
| Completed formal run | Phase dataset |
| Online simulator | four independent `HuskyBfmOnlineEnv` instances |
| HUSKY integration | MuJoCo `0.002 s * 10 = 0.02 s` control step |
| Official BFM reference | IsaacSim `0.005 s * 4 = 0.02 s` control step |
| BFM robot contract | `g1_29dof_hard_waist` |
| Actor/model device | CUDA |
| Replay | CPU `DictBuffer`, capacity 100,000 |
| Episode horizon | 1,024 transitions |
| Stored/executed action | BFM29 / HUSKY23 |
| Expert batch | 64 Base Seq8 + 64 Skate Seq8 = 1,024 rows |
| Online reset | uniform motion, then uniform local frame |

**Caption.** A transition is one `(state, action, next state)` sample at the
0.02 s control step. Episode horizon is the maximum transitions before
time-limit truncation. `Seq8` contains eight consecutive transitions; the
1,024 expert rows are 128 complete sequences times eight. BFM29 is the stored
29-DoF action, while only its 23 physical HUSKY coordinates are executed.
`Motions` count MotionLib trajectories, `clips` count fixed continuous
segments, replay capacity is the maximum stored online transitions, and the
uniform reset gives each selected motion and then each local frame equal
sampling probability.

The **current intended online rollout setting** restores the exact source
physics realization associated with the selected expert frame. It does not
sample an additional random physics realization online. This differs from:

- Experiment 2, where random physics is sampled to collect diverse source
  episodes;
- the historical Phase 100k run, which injected source `qpos/qvel` into
  nominal physics before the mismatch was found.

Formal schedule:

```text
transitions = 100,000
warmup = 1,024 stochastic pretrained-Actor transitions
first update = 1,500
update interval = 500 transitions
updates per block = 50
update blocks = 198
total native updates = 9,900
checkpoints = 20k, 50k, 100k
seed = 4728
```

### Experiment Process and Method

#### Expert sequence and latent

One native batch contains 128 complete eight-frame sequences:

```text
64 Base sequences + 64 Skate sequences
128 * 8 = 1024 rows
```

For each sequence:

```text
b_t = B(N(o_expert,t+1))
z_expert = project_z((1/8) * sum_{t=0}^{7} b_t)
||z_expert||_2 = sqrt(256) = 16
```

The expert latent is repeated over the eight rows.

#### Online reset and observation history

```text
motion m ~ Uniform(all selected Skate motions)
frame t  ~ Uniform({0, ..., frames(m)-1})
```

Robot and skateboard `qpos/qvel` are restored from the same raw frame. Source
XML, joint order, quaternion convention, frame range, physics seed, and
physics fields must match or reset fails closed.

Reset observation uses:

```text
last_action = 0
history_actor = 0
```

After stepping, history contains only the current Actor's online data. Expert
source actions are not inserted into Actor history.

#### Current online latent setting

With four environments, expert-role slots are `[0,3]`; free slots are `[1,2]`.
All slots keep a background latent, refreshed every 100 episode steps from the
z-buffer when available and otherwise from `model.sample_z()`.

For an expert-role reset at source frame `t`:

```text
z_track[k] =
  project_z(
    mean_{i=k+1}^{min(k+8, trajectory_end)}
      B(N(o_expert,t+i))
  )
```

Tracking lasts at most 250 steps and cannot cross the selected Phase segment.
After tracking ends, the slot returns to its background latent. The historical
100k run used unrelated random z; this aligned rule was added after failure
and has not yet been used in a retraining run.

#### Native FB-CPR-Aux optimization

The project calls the vendored `FBcprAuxAgent.update()` rather than
reimplementing it.

Update-time latent mixture:

```text
20% z_goal   = project_z(B(next online goal))
60% z_expert = Base/Skate expert encoding
20% z_random = model.sample_z()
```

With probability 0.8, online replay z is relabeled by this mixture.

Forward-backward objective:

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

`E=2`, `gamma=0.98`, `q_loss_coef=0`, and F/B target `tau=0.01`.

Discriminator:

```text
L_D = -log(sigmoid(D_expert))
      + softplus(D_online)
      + 10 * gradient_penalty
```

Critic targets:

```text
y_D   = r_discriminator + gamma * (Qbar_mean - 0.5 * Qbar_uncertainty)
y_aux = r_aux_normalized + gamma * (Qaux_mean - 0.5 * Qaux_uncertainty)
```

Actor:

```text
Q_FB = <F(s, pi(s,z), z), z>
w = stop_gradient(mean(|Q_FB|))

L_actor =
    -mean(Q_FB)
    -0.05 * w * mean(Q_discriminator)
    -0.02 * w * mean(Q_aux)
```

Optimizer settings:

| Component | Learning rate |
|---|---:|
| F | `3e-4` |
| B | `1e-5` |
| Actor | `3e-4` |
| Discriminator | `1e-5` |
| Main critic | `3e-4` |
| Auxiliary critic | `3e-4` |

**Caption.** Learning rate is the unitless optimizer step-size coefficient.
`F` and `B` are the forward and backward representation networks; `Actor` is
the action policy; `Discriminator` separates expert and online samples; the
main and auxiliary critics estimate discriminator and auxiliary returns.
Smaller rates make more conservative parameter updates, but neither smaller
nor larger is intrinsically better without stability and behavioral evidence.

Actor standard deviation is `0.05`, action-noise clip is `0.3`, critic target
`tau=0.005`, weight decay is zero, AMP/compile/cudagraphs are disabled, and
the z-buffer capacity is 8,192.

Auxiliary penalty:

```text
r_aux = sum_i weight_i * penalty_i
```

| Penalty | Weight |
|---|---:|
| action rate | `-0.1` |
| feet orientation | `-0.4` |
| ankle roll | `-4.0` |
| DoF position limit | `-10.0` |
| surface-relative slippage | `-2.0` |
| undesired contact | `-1.0` |
| torque square | `0.0` |
| torque limit | `0.0` |

**Caption.** Each unitless weight multiplies its penalty before summation into
`r_aux`; a more negative value penalizes that violation more strongly. Action
rate is step-to-step action change; feet orientation and ankle roll measure
pose deviation; DoF-limit penalty measures joint-limit excess; slippage is
foot velocity relative to the board; undesired contact flags disallowed body
contact; torque square measures effort; torque limit measures actuator-limit
excess. Zero disables the two torque terms.

No forward-speed, command, balance, steering, or board-displacement reward was
added. Fall termination uses the same persistent 70-degree/0.45 m detector as
Experiment 2.

#### Training and failure

The native update path was first checked on one fixed 1,024-transition replay
for 1, 10, and 100 updates. A 20k closed-loop baseline then verified growing
replay and checkpoints. The formal Phase run completed 100k transitions and
9,900 native updates, but:

```text
completed training episodes = 3,109
fall terminations = 3,109
horizon completions = 0
```

Frozen matched evaluation:

```text
official BFM0 mean survival = 1.264 s
20k checkpoint             = 0.604 s
50k checkpoint             = 0.519 s
100k checkpoint            = 0.643 s
```

This established a behavioral failure despite finite optimization and valid
checkpoint reloads.

#### Semantics mismatch diagnosis and methods

The following are process steps inside Experiment 3:

| Observed mismatch | Method and result |
|---|---|
| Source expert state used randomized physics, while training reset used nominal physics | Restore the recorded COM/friction/joint-offset realization before exact robot-board `qpos/qvel`; reset error and physics mismatch became zero. |
| BFM gradients/replay included six nonphysical wrists | Project all model-side and online actions to the shared 23DoF subspace; wrists remain exactly zero. |
| MotionLib and online observation scales differ | Distribution audit found expert root angular velocity scale 1.0 versus online 0.25 and 20-27 sigma waist drift; documented but not yet resolved. |
| Random z was unrelated to the selected expert reset | Build same-reset tracking z from future expert observations and use it only in expert-role rollout slots. |
| Source ONNX action and BFM normalized action have different physical target semantics | Derive an exact per-joint physical-target bridge and explicitly mark out-of-range components as projected. |
| The first bridge used ordinary-G1 target gains instead of the formal hard-waist gains | Resolve the actual BFM training override and restore hard-waist `q0`, `Kp`, `Kd`, effort limits, and normalized target gains in the HUSKY runtime. |
| Phase boundaries can remove required previous-action context | Recompute representability under the authoritative hard-waist contract; Phase strict-five coverage is 15.3561%, with `steer2push=3.0445%`. |

**Caption.** `qpos` and `qvel` are MuJoCo generalized position and velocity;
their reset errors should be zero. `sigma` is standard deviations from the
reference observation distribution, so 20-27 sigma is severe drift. Coverage
is the percentage of samples representable without clipping; higher is
better. `Tracking z` is a latent encoded from future states of the same reset
motion, while wrist projection restricts learning/execution to the 23 physical
HUSKY DoFs.

#### Skate expert action matching research

This is the central action-semantics study inside Experiment 3. The question
was not whether a 23D vector can be copied into a 29D vector; that part is
already solved by joint names. The question was whether the source HUSKY
policy action and the BFM normalized action produce the same physical PD
target in the same HUSKY controller.

The source ONNX policy uses the HUSKY control convention:

```text
q_target_src[j] = q0_src[j] + s_src[j] * a_src[j]
```

The BFM Actor uses the BFM-Zero convention:

```text
a_env[j] = clip(5 * a_bfm[j], -5, 5)

G_bfm[j] = 5 * 0.25 * effort_hard-waist[j] / Kp_hard-waist[j]

q_target_bfm[j] = q0_hard-waist[j] + G_bfm[j] * a_bfm[j]
```

Therefore the only exact per-joint physical-target bridge is:

```text
a_bfm_eq[j] =
    (q0_src[j] + s_src[j] * a_src[j] - q0_hard-waist[j])
    / G_bfm[j]
```

The matching procedure was:

1. compare raw source action, raw source action divided by 5, and the affine
   physical-target inverse;
2. reconstruct the source target with `q0_src + s_src*a_src` and the BFM target
   with `q0_hard-waist + G_bfm*a_bfm`;
3. classify each component as `EXACT` when `|a_bfm_eq| <= 1`, otherwise use an
   explicit diagnostic `PROJECTED = clip(a_bfm_eq,-1,1)`;
4. audit all 452,885 collected frames and every joint;
5. compare source and BFM target transitions in MuJoCo;
6. compare target-only, full hard-waist control, and full control with
   BFM-like timing against the saved official IsaacSim one-step response;
7. compare the translated source action with the fresh Actor's first action at
   the same reset, without executing the translated action.

The first D2.5 bridge incorrectly used target scales approximating ordinary
`g1_29dof`. The formal BFM0 entrypoint actually resolves
`robot=g1/g1_29dof_hard_waist`, with:

```text
normalize_from = 1
normalize_to   = 5
action_clip    = 5
action_scale   = 0.25
control_type   = P
```

The current online path now applies the corresponding hard-waist `q0`,
`G_bfm`, `Kp`, `Kd`, and effort limits to the 23 shared MuJoCo actuators.
MuJoCo `gainprm[0]=Kp`, `biasprm[1]=-Kp`,
`biasprm[2]=-Kd`, and `forcerange=[-effort,+effort]`. Robot actuators are
matched one-to-one by joint name; skateboard actuators are unchanged.

Measured mapping results after the authoritative correction:

```text
BFM-equivalent range       = [-9.588672, 9.545807]
exact component coverage   = 95.758968%
exact full-frame coverage  = 37.012707%
projected frames           = 285,260 / 452,885
```

The largest violation counts are:

| Joint | Out-of-range components |
|---|---:|
| waist pitch | 221,184 |
| waist roll | 157,780 |
| waist yaw | 59,893 |
| right hip pitch | 2,694 |
| left hip pitch | 128 |

**Caption.** `BFM-equivalent range` is the normalized action required to
reconstruct each HUSKY source PD target. `Exact component coverage` counts
joint-frame values inside `[-1,1]`; `exact full-frame coverage` requires all
23 physical joints in a frame to be inside that range. A projected frame
clips at least one coordinate and therefore cannot exactly reproduce the
source target. Higher coverage and fewer violations are better.

The temporal part was audited separately because
`action[t]` is the action already applied before `state[t]`, while the source
transition to `state[t+1]` is `action[t+1]`. Under the formal reset
distribution, authoritative strict-five coverage was:

| Dataset | Strict five-action context |
|---|---:|
| Phase | 15.3561% |
| Continuous | 25.7769% |
| Phase `steer2push` | 3.0445% |
| Continuous `steer2push` | 3.0640% |

**Caption.** `Strict five-action context` requires the current source action
and its four history actions to map inside the BFM range. Percentages are
computed over formal MotionLib records; higher is better. The low transition
coverage shows that source actions remain diagnostic and cannot be inserted as
universal BFM expert actions.

The low-level controller restoration was tested on 163 identical probes:
one zero action, 138 single-joint actions, 16 seeded random actions, and eight
fresh frozen-Actor outputs. Of these, 153 had a valid no-contact one-step
comparison. Conditions were:

```text
A: authoritative target + old HUSKY actuator + 0.002*10
B: authoritative target + hard-waist actuator + 0.002*10
C: authoritative target + hard-waist actuator + 0.005*4
```

Against official IsaacSim, B reduced `dq`, `dqdot`, and torque RMSE relative
to A by `20.65%`, `18.80%`, and `48.85%`. Hip `dqdot` improved `43.94%` and
waist `dqdot` improved `82.36%`. C improved `dqdot` but worsened `dq` and
torque relative to B, so production timing remains `0.002*10`.

In the pre-D2.7 P0 matched reset, using the superseded target contract, 421
contexts were fully exact and 91 contained a projected component. This was a
historical diagnostic only; it is not a post-hard-waist result:

| Context | Actor/expert cosine | Actor/expert L2 |
|---|---:|---:|
| Fully exact | 0.4351 | 1.0477 |
| Contains projected component | 0.1853 | 1.6058 |

**Caption.** Cosine similarity measures action-vector direction on `[-1,1]`;
higher is more aligned. `L2` is the Euclidean distance between normalized
23-DoF Actor and translated expert actions; lower is closer. These compare
first actions only and do not measure rollout success.

This is why the source action cannot currently be treated as a universally
exact BFM expert action, even though most individual components are
representable.

#### Post-alignment frozen preflight (pre-D2.7)

The final pre-D2.7 current-stack check used fresh official BFM0, 512 matched
Phase resets, exact source physics, seed 4728, and a 51-step horizon. It
compared:

```text
formal setting: expert slots use aligned tracking z; free slots use background z
control:        all slots use background random z
```

| Metric | Formal aligned/mixed | Pure random |
|---|---:|---:|
| Mean survival | 49.816 | 49.705 |
| Failure by step 20 | 0.00% | 0.00% |
| Failure by step 50 | 26.17% | 26.76% |
| Root tilt p95 | 66.71 deg | 66.93 deg |
| Waist action saturation | 82.33% | 82.69% |

**Caption.** Survival is measured in control steps out of a 51-step horizon,
so higher is better. Failure rates are the percentages terminated by the
named step, so lower is better. Root tilt is torso inclination in degrees;
`p95` is the 95th percentile. Saturation is the percentage of normalized waist
actions at the range boundary, where lower generally leaves more control
margin.

The historical reset population contained 421 fully exact and 91 projected
source-action contexts. Formal Actor/expert first-action cosine was `0.435`
for exact contexts and `0.185` for projected contexts. No board-relative reset
variable strongly predicted failure; the largest absolute survival Spearman
coefficient was `0.136`. These numbers were not rerun after D2.7 and do not
establish current post-restoration behavior.

### Problems and Solutions

| Problem | Current treatment |
|---|---|
| Formal 100k training is numerically valid but behavior is worse | Freeze further long training and use matched frozen evaluation. |
| Physical reset did not reproduce source dynamics | Restore exact source physics and exact robot-board state. |
| Nonphysical wrist actions contaminated the BFM action interface | Apply one 29D active-subspace projection everywhere. |
| Expert reset and random latent represented unrelated motion | Add same-reset future-expert tracking z for expert rollout slots. |
| Source and BFM action coordinates differed | Use the hard-waist physical-target inverse plus explicit `PROJECTED` fallback. |
| The first bridge used the wrong ordinary-G1 target gain | Restore the formal hard-waist target and actuator contract, then recompute all coverage. |
| Waist and hip targets exceed the normalized BFM range | Record representability and projection; do not silently enlarge `[-1,1]`. |
| Observation scale and history semantics are still asymmetric | Keep as explicit unresolved items; do not claim successful retraining. |

**Caption.** `100k` denotes 100,000 online transitions. A projected action has
at least one coordinate clipped to `[-1,1]`; this preserves the BFM action
contract but no longer exactly reproduces the source physical target. The
table pairs each diagnosed failure mode with the currently retained treatment;
it is not a new experiment or a claim that every mismatch is resolved.

### Verified and Unverified Conclusions

Verified:

- [x] Base/Skate 50/50 complete-sequence sampling enters the native BFM update.
- [x] Rewards, termination, replay, six optimizers, model state, and checkpoint
  reload are finite and structurally valid.
- [x] The Phase 100k run completed 100,000 transitions and 9,900 updates.
- [x] The Phase 100k trained checkpoints are less stable than official BFM0
  under the matched frozen evaluation.
- [x] Source-physics restoration and robot-board qpos/qvel reset are exact.
- [x] Shared 23DoF action mapping is exact and six wrists remain zero.
- [x] Official hard-waist `q0`, `Kp`, `Kd`, effort, and normalized target gain
  match their authoritative configuration within `1e-6`.
- [x] Restoring the complete hard-waist controller improves one-step MuJoCo
  response relative to target-only correction.
- [x] The authoritative source-to-BFM inverse is exact for 95.758968% of
  components and 37.012707% of complete frames; violations are dominated by
  waist pitch, roll, and yaw.
- [x] The pre-D2.7 structural preflight passed without changing model or
  normalizer hashes.

Unverified:

- [ ] The expert/online angular-velocity scale asymmetry has been resolved.
- [ ] Remaining IsaacSim/MuJoCo plant differences have been fully explained.
- [ ] Aligned tracking z materially improves frozen official BFM0 behavior.
- [ ] A post-alignment short retraining improves survival or Skate behavior.
- [ ] The trained model is a stable and diverse Skate motion library.
- [ ] Held-out validation/test generalization.

Relevant implementation:

- [`scripts/train_skate_bfm.py`](scripts/train_skate_bfm.py)
- [`scripts/evaluator.py`](scripts/evaluator.py)
- [Published `m2.6-phase-100k-seed4728` checkpoint](https://huggingface.co/Yak9Ce3teeh/skate-bfm/tree/main/motion_library/m2.6-phase-100k-seed4728)
- [`../src/skate_bfm/integration/actions.py`](../src/skate_bfm/integration/actions.py)
- [`../src/skate_bfm/integration/online.py`](../src/skate_bfm/integration/online.py)

## Experiment 4 (Planned): BFB/RFB Dynamics-Conditioned Training

### Experiment Goal

After the Experiment 3 state, action, reset, and controller semantics are
controlled, integrate **Belief-FB (BFB)** and **Rotation-FB (RFB)** into
Skate-BFM. The experiment will test whether conditioning the FB policy on
recent transition dynamics improves adaptation across HUSKY physics
realizations, and whether dynamics-centered latent sampling improves useful
Skate behavior coverage.

This is a planned experiment. No BFB/RFB implementation, training result, or
behavioral conclusion is claimed here.

### Environment and Inputs

| Item | Planned setting |
|---|---|
| Algorithm source | [`maxsbob/BeliefConditionedFB`](https://github.com/maxsbob/BeliefConditionedFB), revision `30e7487` |
| Source files | `agents/dynamics_fb.py`, `agents/dynamics_rfb.py`, `utils/networks.py` |
| Training base | current Skate-BFM FB-CPR-Aux implementation |
| Initialization | fresh official BFM0 checkpoint for every comparison arm |
| Expert data | Base/LAFAN plus Skate Phase and Continuous datasets |
| Online simulator | HUSKY MuJoCo with source-matched reset physics |
| Dynamics context | a fixed-length sequence of `(s_t, a_t, s_{t+1})` |
| Train/evaluation split | disjoint physics signatures and rollout IDs |
| Action contract | BFM29 storage, 23 physical HUSKY DoFs, six wrists zero |

**Caption.** A physics signature is the recorded set of randomized mass/COM,
friction, joint-offset, damping, and actuator fields for one rollout. A
dynamics context is a recent transition window used to infer the active
physics. Disjoint signatures and rollout IDs prevent frames from the same
physical rollout entering both training and held-out evaluation. The exact
context length, BFB/RFB mixing ratio, vMF concentration, training budget, and
seed set will be frozen before the first run rather than inferred from the
current BFM0 configuration.

Experiment 4 may start only after these Experiment 3 preconditions pass:

- source-matched robot, skateboard, and physics reset;
- one consistent expert/online observation scale;
- the 23DoF physical action contract at every model boundary;
- explicit handling of non-representable hip-pitch targets;
- a frozen evaluator that reproduces the official BFM0 and Experiment 3
  baselines.

### Experiment Process and Method

#### Dynamics context

For a context window

```text
C_t = {(s_i, a_i, s_{i+1})}_{i=t-L+1}^{t}
```

the dynamics transformer produces a Gaussian belief:

```text
(mu_h, log_sigma_h) = T_phi(C_t)
h = mu_h + epsilon * exp(log_sigma_h),  epsilon ~ N(0, I)
```

The context encoder is trained through next-state prediction:

```text
s_hat_{i+1} = P_psi(s_i, a_i, h)
L_ctx = mean_i ||s_hat_{i+1} - s_{i+1}||_2^2
```

`h` is the inferred dynamics embedding. `L` is context length in control
steps, and `L_ctx` is normalized next-state mean squared error, where lower is
better. Context windows must not cross rollout, fall, or Phase boundaries.
At episode start, evaluation must report the warm-up interval separately from
the interval with a complete context.

#### BFB integration

BFB keeps one dynamics-independent backward representation and conditions the
forward representation on dynamics:

```text
B = B(s')
F_h = F(s, a, z, stop_gradient(h))
Q_z(s, a | h) = F_h^T z
```

The existing FB residual is changed only at the forward term:

```text
M_h        = F(s,  a,  z, stop_gradient(h))^T B(g)
M_h_target = F_target(s', a', z, stop_gradient(h))^T B_target(g)
D_h        = M_h - gamma * M_h_target
```

The current FB diagonal, off-diagonal, and `B` orthogonality losses then
operate on `D_h`. `B` remains shared across dynamics and receives no `h`.
During FB/Actor updates, `h` is stop-gradient; `T_phi` and `P_psi` are updated
through `L_ctx` in a separate update path. The Actor receives the same `h`
used by the forward value when required by the source method.

#### RFB integration

RFB uses the inferred dynamics direction as the center of latent exploration.
For latent dimension `d`:

```text
h_hat = h / ||h||_2
u ~ vMF(e_1, kappa)
v = (e_1 - h_hat) / ||e_1 - h_hat||_2
H(h_hat) = I - 2 v v^T
z_rfb = sqrt(d) * H(h_hat) u
```

The Householder transform `H` maps the north-pole-centered von
Mises-Fisher sample to the dynamics direction. `kappa` is the unitless vMF
concentration: larger values concentrate samples more tightly around `h_hat`.
The source mixed sampler is retained:

```text
z_goal = project_z(B(g))
z = z_rfb with probability beta
    z_goal otherwise
||z||_2 = sqrt(d)
```

`beta` is the RFB sampling probability. RFB will not be implemented as a
simple replacement of BFM0's random latent branch. The vMF draw,
Householder alignment, dynamics-conditioned forward value, and goal-latent
mixture form one contract. If `dim(h) != d`, integration stops until an
explicit learned projection is defined and validated; truncation or zero
padding is not allowed.

#### Controlled training comparison

All arms use the same data split, reset population, online transition budget,
expert ratio, evaluator, checkpoint schedule, and seeds:

| Arm | Method |
|---|---|
| A | current post-alignment FB-CPR-Aux baseline |
| B | baseline plus BFB dynamics context |
| C | baseline plus BFB context and RFB latent sampling |

**Caption.** Arm A isolates the effect of the existing Skate data and semantic
alignment. Arm B measures dynamics conditioning. Arm C measures the additional
effect of dynamics-centered latent exploration. Higher behavioral success and
lower held-out prediction error are preferred; an arm is not accepted from
training loss alone.

The first comparison uses only exact action contexts. Projected hip-pitch
contexts are then introduced as a separate ablation, so controller-range error
cannot be mistaken for a BFB/RFB effect. Phase and Continuous data are reported
separately before any pooled result.

Evaluation will include:

| Metric | Meaning |
|---|---|
| `L_ctx` | held-out normalized next-state prediction MSE; lower is better |
| mean survival | control steps before persistent fall; higher is better |
| fall rate | fraction of rollouts ending in persistent fall; lower is better |
| horizon completion | fraction reaching the fixed horizon; higher is better |
| root tilt p95 | 95th percentile torso inclination in degrees; lower is better |
| board retention | fraction of non-fall steps with valid robot-board relation; higher is better |
| phase-conditioned success | survival/completion grouped by push, transitions, and steer |
| latent coverage | distinct successful behaviors retained across evaluated latent targets |
| action saturation | fraction of physical normalized actions at range limits; lower is generally better |

**Caption.** Metrics are computed on identical held-out reset manifests and
reported by physics signature, motion phase, and seed. `p95` is a tail
statistic rather than a mean. Latent coverage counts only behaviorally
successful rollouts and must not be inferred from embedding distance alone.

### Problems and Solutions

| Planning-stage risk | Required control |
|---|---|
| BFB/RFB source is a JAX benchmark implementation, while Skate-BFM is a PyTorch continuous humanoid controller | Port equations and update boundaries, not framework-specific code; verify tensor-by-tensor parity on fixed batches. |
| Dynamics context may encode phase or pose instead of physics | split by physics signature, evaluate unseen signatures, and compare contexts at matched motion phase |
| Incomplete context at reset may bias evaluation | report warm-up separately and use one fixed context initialization rule in every arm |
| RFB may collapse exploration around an inaccurate `h` | monitor `L_ctx`, angular spread of sampled `z`, and successful latent coverage |
| Existing hip-pitch projection may dominate outcomes | run exact-context comparison first and projected-context ablation second |
| Added losses may destabilize pretrained BFM0 | require finite gradients, bounded parameter drift, frozen-policy checkpoints, and early stop on behavioral regression |

**Caption.** These are controls for a future experiment, not observed
BFB/RFB results. A fixed-batch numerical pass is necessary but insufficient;
the final decision is based on held-out closed-loop behavior.

### Verified and Unverified Conclusions

Verified:

- [x] The authoritative source defines BFB with dynamics context conditioning
  `F`, a shared unconditioned `B`, transition-prediction context learning, and
  stop-gradient context during FB learning.
- [x] The authoritative source defines RFB with vMF sampling, Householder
  alignment to the dynamics embedding, and mixing with a `B(goal)` latent.

Unverified:

- [ ] BFB has been integrated into Skate-BFM.
- [ ] RFB has been integrated into Skate-BFM.
- [ ] Context prediction identifies HUSKY physics rather than motion phase.
- [ ] BFB improves held-out dynamics adaptation over the post-alignment
  FB-CPR-Aux baseline.
- [ ] RFB improves Skate latent coverage without reducing stability.
- [ ] BFB/RFB training resolves any Experiment 3 semantics mismatch.
- [ ] BFB/RFB produces a stable and diverse Skate motion library.
