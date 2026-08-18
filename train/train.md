# Skate-BFM Training Experiments

This document contains exactly three project experiments. Every audit,
preflight, smoke run, and diagnostic belongs to one of these experiments; it
is not a separate experiment.

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
| Output | Hugging Face `Yak9Ce3teeh/skate-sim-dataset` |

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
- [Hugging Face raw/Phase/Continuous dataset](https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset)

### Problems and Solutions

| Problem | Method used |
|---|---|
| One episode contains push, transitions, and steer | Use official phase clock and contiguous phase-run segmentation. |
| Falls contaminate the preceding motion tail | Remove the confirmed fall and a 0.15 s pre-fall margin. |
| Short phase fragments cannot provide Seq8 next-state samples | Require at least 9 frames. |
| Phase-only records lose longer temporal transitions | Build Continuous 500-frame clips from the same raw data. |
| HUSKY has 23 joints but MotionLib expects 29 | Map shared joints by name and set six wrists to zero. |
| Processed data could lose source identity | Store source NPZ, frame range, command, physics seed, phase, board state, and action. |

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
| Base expert | 862 LAFAN motions |
| Skate expert | Phase: 6,038 motions; Continuous option: 890 clips |
| Completed formal run | Phase dataset |
| Online simulator | four independent `HuskyBfmOnlineEnv` instances |
| Control step | 0.02 s |
| Actor/model device | CUDA |
| Replay | CPU `DictBuffer`, capacity 100,000 |
| Episode horizon | 1,024 transitions |
| Stored/executed action | BFM29 / HUSKY23 |
| Expert batch | 64 Base Seq8 + 64 Skate Seq8 = 1,024 rows |
| Online reset | uniform motion, then uniform local frame |

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
| Phase boundaries can remove required previous-action context | Measure current/history/five-action representability; Phase strict coverage is 64.8366%, with `steer2push=12.1379%`. |

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
q_target_bfm[j] = q0_bfm[j] + 5 * s_bfm[j] * a_bfm[j]
```

Therefore the only exact per-joint physical-target bridge is:

```text
a_bfm_eq[j] =
    (q0_src[j] + s_src[j] * a_src[j] - q0_bfm[j])
    / (5 * s_bfm[j])

               = b[j] + k[j] * a_src[j]
```

The matching procedure was:

1. compare raw source action, raw source action divided by 5, and the affine
   physical-target inverse;
2. reconstruct the source target with `q0_src + s_src*a_src` and the BFM target
   with `q0_bfm + 5*s_bfm*a_bfm`;
3. classify each component as `EXACT` when `|a_bfm_eq| <= 1`, otherwise use an
   explicit diagnostic `PROJECTED = clip(a_bfm_eq,-1,1)`;
4. audit all 452,885 collected frames and every joint;
5. compare source and BFM target transitions in MuJoCo;
6. test whether a learned/piecewise hip-tail correction generalizes to held-out
   transitions;
7. compare the translated source action with the fresh Actor's first action at
   the same reset, without executing the translated action.

The source and current controllers share position-actuator semantics, PD gains,
damping, force limits, 50 Hz policy rate, and no delay/filter. They still
differ in target parameterization, clipping, MuJoCo integration
(`0.005*4` source versus `0.002*10` current), and solver settings. Thus an
exact target bridge does not imply identical low-level next-state dynamics.

Measured mapping results:

```text
raw source action range       = [-4.933043, 6.152792]
raw components outside [-1,1] = 32.1595%
affine BFM-equivalent range   = [-2.343024, 1.716985]
exact component coverage      = 99.600839%
exact full-frame coverage     = 92.1503%
```

Naive mappings were rejected:

| Mapping | Physical target RMSE |
|---|---:|
| raw source action copied as BFM action | 1.330264 rad |
| raw source action divided by 5 | 0.530560 rad |
| affine bridge with projection | 0.021775 rad |

The remaining full-frame failures are concentrated in hip pitch. The exact
bridge reconstructs representable targets with RMSE `1.97e-17 rad` and maximum
error `4.44e-16 rad`; projected hip-tail errors reach `1.490767 rad`. Fixed
center coverage would require left/right hip range multipliers of `1.510x` and
`1.860x` for 99.9% coverage, or `2.156x` and `2.343x` for all observed
targets. This is a control-parameterization mismatch, not a small numerical
noise issue.

The two dominant affine rows are:

```text
left hip pitch:  a_bfm_eq = +0.090089 + 0.493240 * a_src
right hip pitch: a_bfm_eq = -0.540537 + 0.493240 * a_src
```

Removing the default-position offset would recover only `3.76%` of left-hip
violations and `79.41%` of right-hip violations. The remaining tail therefore
cannot be solved by recentering alone; reachable range is also insufficient.

The temporal part was audited separately because
`action[t]` is the action already applied before `state[t]`, while the source
transition to `state[t+1]` is `action[t+1]`. Under the formal reset
distribution, current-action/history/five-action context coverage was:

| Dataset | Current valid | History valid | Strict five-action context |
|---|---:|---:|---:|
| Phase | 79.9611% | 66.6396% | 64.8366% |
| Continuous | 92.2596% | 87.1721% | 85.8434% |

Hip pitch explained `99.9612%` of Phase strict-context failures and `99.9810%`
of Continuous failures. In `steer2push`, current hip violation was `60.15%`
and strict five-context failure was `87.86%`; violation runs averaged 4.32
frames, p95 9, maximum 14. This shows temporal persistence rather than a
single corrupted frame.

A 4,096-transition MuJoCo refinement reduced normalized next-state RMSE from
`0.4689` to `0.4482`, but the held-out RMSE ratio was only `0.9610` and
left-only/right-only hip tails did not improve consistently. Learned,
piecewise, and dynamics-aware correction was therefore rejected. The retained
production rule is exact affine translation plus explicit `PROJECTED` fallback;
canonical source actions and formal online semantics remain unchanged.

At the final P0 matched reset, 421 contexts were fully exact and 91 contained a
projected component. The translated source action was diagnostic only:

| Context | Actor/expert cosine | Actor/expert L2 |
|---|---:|---:|
| Fully exact | 0.4351 | 1.0477 |
| Contains projected component | 0.1853 | 1.6058 |

This is why the source action cannot currently be treated as a universally
exact BFM expert action, even though most individual components are
representable.

#### Post-alignment frozen preflight

The final current-stack check used fresh official BFM0, 512 matched Phase
resets, exact source physics, seed 4728, and a 51-step horizon. It compared:

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

The reset population contained 421 fully exact and 91 projected source-action
contexts. Formal Actor/expert first-action cosine was `0.435` for exact
contexts and `0.185` for projected contexts. No board-relative reset variable
strongly predicted failure; the largest absolute survival Spearman
coefficient was `0.136`.

### Problems and Solutions

| Problem | Current treatment |
|---|---|
| Formal 100k training is numerically valid but behavior is worse | Freeze further long training and use matched frozen evaluation. |
| Physical reset did not reproduce source dynamics | Restore exact source physics and exact robot-board state. |
| Nonphysical wrist actions contaminated the BFM action interface | Apply one 29D active-subspace projection everywhere. |
| Expert reset and random latent represented unrelated motion | Add same-reset future-expert tracking z for expert rollout slots. |
| Source and BFM action coordinates differed | Use exact affine target translation plus explicit `PROJECTED` fallback. |
| Hip-pitch tails exceed BFM action range | Record representability and projection; do not silently enlarge range or adopt weak learned correction. |
| Observation scale and history semantics are still asymmetric | Keep as explicit unresolved items; do not claim successful retraining. |

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
- [x] Exact source-to-BFM action translation is valid for 99.600839% of
  components; the remaining projected tail is mainly hip pitch.
- [x] The current post-alignment structural preflight passes without changing
  model or normalizer hashes.

Unverified:

- [ ] The expert/online angular-velocity scale asymmetry has been resolved.
- [ ] Hip-pitch range and low-level controller semantics have been fully
  aligned.
- [ ] Aligned tracking z materially improves frozen official BFM0 behavior.
- [ ] A post-alignment short retraining improves survival or Skate behavior.
- [ ] The trained model is a stable and diverse Skate motion library.
- [ ] Held-out validation/test generalization.

Relevant implementation:

- [`scripts/train_skate_bfm.py`](scripts/train_skate_bfm.py)
- [`scripts/evaluator.py`](scripts/evaluator.py)
- [`../src/skate_bfm/integration/actions.py`](../src/skate_bfm/integration/actions.py)
- [`../src/skate_bfm/integration/online.py`](../src/skate_bfm/integration/online.py)
