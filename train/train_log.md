# Training Log

## 0. Workspace Initialization

- Date: 2026-08-03
- Completed: created the dedicated `train` branch for model training, added
  the local dataset directory, and reserved `model/motion_library/` for
  generated model outputs.

## 1. Single Rollout Processing

- Date: 2026-08-03
- Completed: added single-rollout command segmentation, failure/reset cleanup,
  source-format motion export, and synchronized video or MuJoCo pose replay.

## 2. Live HUSKY Phase Inspection

- Date: 2026-08-03
- Completed: verified the official fixed phase schedule, added live phase
  output, classified steering from the heading command, tracked skateboard
  heading changes, and refined fall detection.

## 3. Large-Scale Collection Preparation

- Date: 2026-08-04
- Completed: configured the balanced HUSKY command grid, multi-round parallel
  collection, resumable target-based replacement rollouts, and separate raw
  rollout and cleaned expert-motion duration reporting.

## 4. Engineering Progress Tracking

- Date: 2026-08-06
- Completed: added a shared project milestone graphic and synchronized
  progress sections for the `train` and `main` branch documentation.

## 5. Training Script Directory Organization

- Date: 2026-08-09
- Completed: moved the project-owned HUSKY rollout collection script and its
  configuration from `train/scripts/` to
  `train/scripts/data_collection/`. Collection plan and summary artifacts were
  placed there as well, and the obsolete `train/scripts/temp/` directory was
  removed. This was directory organization only; no data format, model
  behavior, training logic, or collection parameters were changed.

## 6. M1.1 Skate Expert Integration Test

- Date: 2026-08-09
- Collection script:
  `train/scripts/data_collection/rollout_split.py`
- Collection command:

  ```bash
  python train/scripts/data_collection/rollout_split.py \
    --live --record --headless \
    --robot-xml husky_sim/upstream/test_scene/mjlab_scene.xml \
    --policy husky_sim/upstream/ckpts/test.onnx \
    --device cpu \
    --round-id 901 --rollout-id 001 \
    --episode-id m1_1_rollout_001 \
    --output-dir /tmp/skate_bfm_m1_1.0L6F7i \
    --randomize-physics --physics-seed 20260809 \
    --initial-v 1.0 --initial-h 0.0 \
    --max-policy-frames 50 --status-interval 0.2 \
    --no-render-previews
  ```

- Environment: official HUSKY MuJoCo test scene and ONNX policy on CPU, with
  the existing official HUSKY startup/reset physics randomization.
- Raw rollout:
  `/tmp/skate_bfm_m1_1.0L6F7i/round_901/rollout_001/raw_rollout/m1_1_rollout_001.npz`;
  50 frames at 50 Hz, one second of collection, containing one valid `push`
  segment.
- Conversion script: `train/scripts/convert_husky_to_bfm.py`.
- Converted output:
  `/tmp/skate_bfm_m1_1.0L6F7i/round_901/rollout_001/bfm_motionlib/skate_expert.pkl`;
  one motion, 50 frames, 0.98 seconds of MotionLib duration.
- MotionLib loaded: yes. The official BFM-Zero `MotionLibRobot` loaded both the
  Skate motion and the original 862-motion LAFAN training library.
- Expert batch generated: yes. The official
  `load_expert_trajectories_from_motion_lib()` produced matching Skate and Base
  batches with `state` `[16, 64]`, `last_action` `[16, 29]`, and
  `privileged_state` `[16, 463]`, all finite `torch.float32`.
- Issues recorded without representation changes: HUSKY provides 23 actuated
  joints, so the six BFM wrist joints remain fixed at zero; `smpl_joints` is a
  zero placeholder; and this minimal integration sample covers only `push`,
  not all Skate motion classes.
- Training performed: no. No BFM-Zero model, loss, latent, or optimizer code
  was modified.

## 7. M1.2 Base + Skate Expert Sources

- Date: 2026-08-09
- Original behavior: official BFM-Zero `humanoidverse/train.py` creates one
  MotionLib-backed `expert_slicer`. Agent updates, expert rollout context, and
  prioritization all consume that Base/LAFAN buffer.
- Current sources:
  - The project-owned `train/scripts/train_skate_bfm.py` is based on the
    official BFM-Zero training entry point. The official source directory
    remains unchanged.
  - `train/scripts/isaac_env/humanoidverse/` is the vendored BFM-Zero
    Isaac/HumanoidVerse runtime, including agents, Hydra configuration,
    MotionLib, simulator backends, and robot assets. LAFAN data and checkpoints
    are excluded.
  - The training entry imports its environment config and expert loader from
    this vendored runtime and has no runtime dependency on
    `model/bfm-zero-source/`.
  - `expert_base` is created in the copied `Workspace.train_online()` through
    the original Base MotionLib and observation construction.
  - `expert_skate` is created there only when
    `TrainConfig.skate_expert_motion_file` is set. It uses a separate instance
    of the same official `MotionLibRobot` and the same official expert loader.
  - `expert_slicer` remains an object alias to `expert_base` so existing agent
    updates retain their original behavior. It is not a third expert source.
- Changed files:
  - `train/scripts/train_skate_bfm.py`
  - `train/scripts/isaac_env/`
  - `train/scripts/data_collection/rollout_split.py`
  - `pyproject.toml`
  - `README.md`
  - `train/README.md`
  - `src/skate_bfm/exp/h1_bfm_coverage/core.py`
  - `tests/test_h1_bfm_coverage.py`
  - `train/train_log.md`
- Sampling interface:

  ```python
  batch_base = replay_buffer["expert_base"].sample(batch_size, seq_length)
  batch_skate = replay_buffer["expert_skate"].sample(batch_size, seq_length)
  ```

  Source metadata is stored on each buffer as `source = "base"` or
  `source = "skate"` and is not included in observations or model inputs.
- Validation:
  - Base-only loading and three repeated samples: passed.
  - Skate-only loading and three repeated samples: passed.
  - Both sources initialized simultaneously without replacement: passed.
  - Base source names contain no `skate/` motions; Skate source names all use
    the `skate/` prefix and contain no `fall` motions.
  - Both batches contain `observation` and `next.observation`, each with
    `state` `[16, 64]`, `last_action` `[16, 29]`, and `privileged_state`
    `[16, 463]` as finite `torch.float32`.
- Source statistics:
  - Base: 862 motions, 258,600 source frames at 30 Hz, 8,591.303 seconds in
    official MotionLib, and 430,138 frames in the 50 Hz expert buffer.
  - Skate: one positive `push` motion, 50 frames at 50 Hz, 0.98 seconds, and no
    `fall` motion.
- Not decided in M1.2: Base/Skate training sampling ratio, model adaptation,
  and Base/Skate environment mixing.
- Training entry dependencies: added the official runtime requirements
  `exca`, `POT`, `mediapy`, and `wandb` to the existing `motionlib` optional
  dependency group and installed them in the `skatebfm` environment.
- Independent Isaac runtime: installed the locked Python 3.10 environment
  under `train/scripts/isaac_env/.venv` with Isaac Sim 4.5.0 and Isaac Lab
  2.0.2. The local binary environment is ignored by Git and can be rebuilt
  with `uv sync --locked`.
- Isaac environment validation: launched the vendored headless runtime,
  resolved the G1 XML/USD/mesh assets from
  `train/scripts/isaac_env/humanoidverse/data/robots`, loaded the project LAFAN
  library, created one 29-DoF HumanoidVerse environment, reset it, and
  completed one physics step. No retained test script was added.
- Host warning: Isaac Sim reported duplicate Vulkan ICD entries for the
  NVIDIA GPU. The tested headless environment still initialized and stepped
  successfully, but the host driver installation should be cleaned before
  long training runs.
- Training performed: no. Agent update methods, losses, latent sampling, and
  model architectures were not modified. Temporary validation code was not
  retained.

## 8. M1.3 Expert Sampling Integration

- Date: 2026-08-09
- Added `skate_expert_ratio`, defaulting to `0.5`, to control the proportion
  of complete expert sequences sampled from `expert_skate`; the remainder is
  sampled from `expert_base`.
- `expert_slicer` uses the project-owned Base/Skate sampler only when Skate
  expert data is enabled and the ratio is positive. With no Skate source or
  ratio `0.0`, it remains the original `expert_base` object.
- Ratio `0.0`: passed Base-only backward-compatibility validation.
- Ratio `0.5`: passed with 64 Base and 64 Skate sequences for a 1024-frame
  batch at sequence length 8.
- Ratio `1.0`: passed with 128 Skate sequences for the same batch shape.
- Sequence integrity and mixed batch shape/dtype checks passed. Every
  eight-frame sequence came from exactly one source.
- The strict-loaded frozen BFM-Zero checkpoint accepted the mixed batch
  through the unmodified `encode_expert()` path. Output was finite
  `[1024, 256]` with latent norm 16.
- Formal training performed: no. B, F, Actor, discriminator, critic,
  auxiliary critic, sampling mathematics after expert acquisition, and all
  losses remain unchanged. Temporary validation code was not retained.

## 9. M2.1 Skate Online Replay Integration

- Date: 2026-08-10
- HUSKY runtime: existing `husky_sim/src/skate_husky/HuskyLiteEnv` using the
  official generated MuJoCo scene. The runtime now exposes robot body kinematics
  for the existing BFM privileged observation construction and a public control
  mapping setter; it is still the existing simulator, not a replacement.
- Observation source: existing H1 BFM0 state/history semantics, extended to
  the official training contract:
  `state[64]`, `privileged_state[463]`, `last_action[29]`, and
  `history_actor[372]`. Board pose, velocity, and heading remain raw metadata
  only and are not appended to neural-network inputs.
- Action adapter: existing name-based `Bfm0ToHusky23`.
  Stored BFM action is `29D`; executed HUSKY action is the mapped `23D`
  action. Official HUSKY neutral controls and per-joint scales are shared by
  the integration and H1 runner.
- Latent: each bounded rollout step stores the actual frozen BFM0
  `model.sample_z()` result with `z_dim=256`; no zero or post-hoc latent is
  inserted.
- Replay: official `DictBuffer` only. Project-owned registration creates
  `replay_buffer["train_skate"]` and keeps
  `replay_buffer["train"]` as the same-object compatibility alias. No
  `train_base`, replay mixing, or custom replay buffer was added.
- Rollout validation: 64 frozen-checkpoint HUSKY steps passed. Replay sample
  validation passed for observation, action, z, next observation, termination,
  truncation, and finite values.
- `last_action` validation passed over all consecutive steps:
  `observation_t.last_action = a_(t-1)` and
  `next_observation_t.last_action = a_t * 5` under the official normalized
  action convention.
- Reset boundary validation passed. The bounded rollout marks its final step
  as `truncated`; attempting another step before `reset()` raises, so no
  pre-reset to post-reset transition can enter replay. The lightweight HUSKY
  runtime does not emit a native terminal signal in this path; physical
  termination remains a later runtime dependency.
- B/F forward preflight passed with the frozen official checkpoint and no
  gradients: `F` output `[2,16,256]`, `B` output `[16,256]`, and
  `F^T B` finite. No optimizer step or formal training was performed.
- Tracking compatibility: `expert_tracking` is explicitly Base-only while
  `expert_slicer` remains the Base/Skate training sampler. The official
  tracking-z helper uses `expert_tracking` when present.
- Known later dependencies: native HUSKY termination semantics for the full
  training runtime, auxiliary reward/Q_aux mapping, Base online replay
  retention, and any later BFB/RFB/MEBE work. These are outside M2.1.

## 10. M2.1b Formal HUSKY Online Training Path

- Date: 2026-08-10
- Added the explicit `online_env` configuration with `base` and `skate`
  choices. `base` preserves the original HumanoidVerse Isaac path; `skate`
  makes the formal `Workspace` construct and step `HuskyBfmOnlineEnv`.
- Skate mode is currently restricted to `collect_only=True`, one environment,
  `DictBuffer`, no prioritization, and no Isaac evaluation. This fail-closed
  boundary prevents optimizer updates before later training dependencies are
  resolved.
- Formal online environment: HUSKY MuJoCo through the existing
  `HuskyBfmOnlineEnv`, existing 29D-to-23D action adapter, and existing
  BFM-compatible observation adapter.
- Replay: `train_skate`; `train` is the same-object compatibility alias.
- Workspace dry run: passed with 64 real HUSKY transitions. Workspace sampled
  and checked the resulting replay fields, 29D stored action, 23D executed
  action, 256D rollout latent, and observation widths 64/463/29/372.
- Latent baseline: `model.sample_z()` with the original
  `update_z_every_step` period. This is a temporary single-HUSKY baseline, not
  a full reproduction of the original vectorized rollout-context logic.
- Reset boundary: the bounded final transition is truncated, and the existing
  HUSKY online wrapper requires reset before another step. No cross-reset
  transition is written.
- Optimizer step: no. Workspace recorded zero `agent.update()` calls, and
  Skate mode rejects `collect_only=False`.
- Base backward compatibility: passed by constructing a one-environment
  HumanoidVerse Isaac Workspace with the original 29D action and BFM
  observation contract. The Base training loop was not run.
- Known blockers before training: pretrained BFM0 initialization, Skate
  auxiliary reward/Q_aux definition, and native HUSKY physical termination.

## 11. M2.2a Pretrained BFM0 Initialization and Expert/Replay Merge

- Date: 2026-08-10
- Pretrained source:
  `model/bfm-zero-official/`, configured through
  `BFM0_PRETRAINED_CHECKPOINT`.
- Checkpoint format: model-only BFM0 directory containing `config.json`,
  `init_kwargs.json`, and `model.safetensors`; 537 tensors were inspected.
  The bundle does not contain an agent config, optimizer file, replay buffer,
  or training status.
- Strict compatibility: passed for the complete model config, observation
  space (`state[64]`, `privileged_state[463]`, `last_action[29]`,
  `history_actor[372]`), action dimension 29, all state keys, and every tensor
  shape. Missing model components: none.
- Loaded components: B, target B, F, target F, Actor, discriminator, critic,
  target critic, auxiliary critic, target auxiliary critic, observation
  normalizer, and auxiliary reward normalizer.
- Resume precedence:
  `work_dir/checkpoint/` Skate resume first, official BFM0 pretrained
  initialization second. A new Skate workdir fails closed if no pretrained
  checkpoint is configured; random initialization is not allowed.
- Checkpoint optimizer states: no. Pretrained initialization therefore keeps
  the fresh optimizers created from the current Skate config, but does not
  restore momentum or execute any optimizer step. The first adaptation
  optimizer-state policy remains to be defined.
- Expert loading is independent from the online environment. The online
  environment remains `HuskyBfmOnlineEnv`; one minimal HumanoidVerse context
  is constructed only to load Base and Skate MotionLib expert buffers and is
  never stepped for online replay.
- Skate Workspace buffers: `expert_base` yes, `expert_skate` yes when
  configured, `expert_slicer` yes, `expert_tracking` is the Base buffer,
  `train_skate` yes, and `train` is the same-object alias of `train_skate`.
- Mixed expert validation used ratio 0.5 and sequence length 8. A 1024-frame
  sample contained 64 complete Base sequences and 64 complete Skate
  sequences. Tracking sampling remained Base-only.
- Formal pretrained HUSKY dry run: passed for 64 steps using actions and
  latents from the strict-loaded official BFM0 model. The replay retained the
  29D stored action, 23D executed HUSKY action, and 256D latent contract.
- Combined preflight: expert encoding produced finite `[1024, 256]` latents;
  Skate replay forward outputs were finite with F `[2, 16, 256]`, B
  `[16, 256]`, and a compatible discriminator forward.
- Parameter fingerprint changed: no. Model buffer fingerprint changed: no.
  The collect-only model remained frozen in inference mode.
- `agent.update()` calls: 0. `backward()` calls: 0. Optimizer steps: 0.
  Formal training: no. No checkpoint or metrics were generated, so
  `train_res.md` remains unchanged.
- Known remaining blockers: Skate auxiliary reward/Q_aux, native HUSKY
  physical termination, and the first Skate FB adaptation protocol.
