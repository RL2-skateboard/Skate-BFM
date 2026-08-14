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

## 12. M2.2b-0 Fixed Evaluation Protocol

- Date: 2026-08-10
- Added `train/evaluation_protocol.json` as the fixed comparison contract for
  BFM0, Original FB + Skate, and future BFB/RFB/FB-MEBE ablations. It freezes
  rollout IDs, seeds, command metadata, 128-step horizon, seen/unseen dynamics
  seeds, context schema, physical projection, entropy bins, and Base tracking
  configuration.
- Added `train/scripts/evaluate_skate_bfm.py`. It creates a separate
  `eval_skate_transition` buffer and never writes to `train` or `train_skate`.
  Unseen dynamics and future context trajectories are explicitly
  evaluation-only.
- Fixed held-out set: four rollouts and 512 transitions. Two dynamics
  realizations are marked seen and two are held-out unseen. Every realization
  records the exact COM, friction, reset joint offsets, randomization seed,
  command metadata, and transition IDs.
- FB diagnostics use the current vendored BFM-Zero `update_fb()` equations
  under `torch.no_grad()`: `fb_loss`, diagonal/off-diagonal terms,
  orthonormality terms, B norm, and z norm. No loss or optimizer implementation
  was changed.
- Added the Skate-BFM representation diagnostic `F_i^T B_j`: diagonal mean,
  off-diagonal mean, margin, Top-1, Top-5, mean rank, and median rank. It is
  not described as an official BFB or RFB metric.
- Fixed pretrained baseline result: FB loss `1775099.875`, Top-1 `0.25`,
  Top-5 `0.6875`, mean rank `5.703125`, and median rank `3`.
- Physical rollout evaluation uses only real MuJoCo fields: bounded duration,
  root height/tilt and linear/angular velocity, board position/heading and
  linear/angular velocity. Native survival/fall, contact, slippage, force,
  command error, and aligned joint-reference error remain unavailable and are
  not synthesized.
- Current official HUSKY play randomization audit: robot torso COM, skateboard
  COM, robot/deck/foot/wheel friction, and reset joint position offsets.
  External push and observation corruption are disabled by the official play
  configuration. No new randomization was added.
- Physical behavior projection `husky_skate_phi_v1` uses six real fields:
  root linear velocity x/y, root angular velocity z, board linear velocity
  x/y, and board angular velocity z. Fixed 10-bin discretization produced
  entropy `4.6439129236` nats over 512 states; the fixed board x/y occupancy
  used 20 bins per axis.
- The evaluator was run twice with the same protocol. Metrics, resolved
  manifest, and coverage artifact were byte-identical. Metrics SHA256:
  `4372b6b4a48b6518330e65d36c3d5d545fa2d27b98591e0f126e8eac184f7fc5`.
- Base retention reuses `HumanoidVerseIsaacTrackingEvaluation` with
  `train/dataset/BFM-Zero/evaluation/lafan_29dof.pkl`, fixed seed 20260810,
  one episode per motion, and the existing official tracking metrics. This
  preflight identified and validated the entrypoint but did not run the full
  1024-env Base evaluation.
- Context hooks reserve state/action/next-state trajectories with short,
  medium, and long lengths 16/64/256. BFB `h`, RFB `kappa`, and MEBE
  density/beta fields are `unavailable` or `not_applicable`; no fake values
  are inserted.
- Parameter mutation: no. Model buffer mutation: no. `agent.update()` calls:
  0. `backward()` calls: 0. Optimizer steps: 0. Formal training: no.
  `train_res.md` remains unchanged.

### Method Source Grounding

- BFB/RFB reference: `skylooop/BeliefConditionedFB`, revision
  `30e7487ca033c3619ec744ed55f916ece005c425`; reviewed
  `agents/dynamics_fb.py`, `agents/dynamics_rfb.py`, and
  `utils/networks.py`.
- BFB source behavior: a dynamics trajectory is encoded by
  `dynamic_transformer`; the context is learned with next-state prediction,
  stop-gradient is applied during FB learning, dynamics context conditions F,
  and B remains shared with no dynamics input.
- RFB source behavior: vMF samples are drawn around the north pole, aligned to
  normalized dynamics context with a Householder reflection, projected to the
  latent sphere, and mixed with a B-goal latent by the source
  `sample_mixed_z()`.
- FB-MEBE reference: `MATH-286-Pro/FB-MEBE`, revision
  `344385dcbabd541240c27c3ee41fdc4de9c548ae`; reviewed
  `agent_meta/fb/agent.py`, `density_estimator/agent_normalizing_flow.py`, and
  `pretrain.py`.
- FB-MEBE source behavior: density is estimated on an achieved physical/goal
  state projection; complete achieved states are sampled from replay with
  weight proportional to `q(s)^(-beta)`, then mapped through `B(s_E)` to
  `z_E`. Training mixes this exploration latent with uniform latent, and
  online `refresh_z()` uses the same exploration mechanism with `p_reverse =
  0.8` after enough estimator data. No MEBE estimator or 0.8 branch was added
  to Skate-BFM.
- Q_aux mapping to FB-MEBE behavior regularization remains unresolved.

## 13. M2.2b-1 First B/F-only Skate Adaptation

- Date: 2026-08-10
- Added a fail-closed `skate_update_mode` with `none` and `fb_only`.
  `none` preserves the M2.2a collect-only behavior. `fb_only` is the only
  accepted Skate adaptation path and never calls the full
  `FBcprAuxAgent.update()`.
- Training used the official pretrained BFM0 initialization from
  `model/bfm-zero-official`. Each milestone was an independent run from that
  same initialization; no run resumed from another milestone.
- The training replay contained 1024 transitions from two `train_seen_*`
  HUSKY rollouts. It used Skate dynamics only, reused the two protocol-defined
  seen dynamics seeds, and contained zero unseen or evaluation transitions.
  `replay_buffer["train"]` remained the same object as
  `replay_buffer["train_skate"]`.
- The executed M2.2b-1 checkpoints had `skate_expert_motion_file=null`, so
  their actual expert sampler was Base-only. The configured
  `skate_expert_ratio=0.5` had no effect without an `expert_skate` buffer.
  This is a provenance correction; the checkpoints were not retrained in the
  M2.2b-2 audit. Expert latents still used `encode_expert()`, the vendored
  `sample_mixed_z()`, and `relabel_ratio=0.8` semantics. No RFB, BFB, MEBE,
  dynamics context, or new exploration branch was added.
- The adaptation path directly called the vendored `update_fb()`. Only
  `forward_optimizer` and `backward_optimizer` stepped. F, B, target F, and
  target B changed; Actor, discriminator, QD, Qaux, and their target modules
  remained unchanged. The observation normalizer changed according to the
  original update semantics, and the z-buffer changed after each update.
- Fixed evaluator version: `skate-bfm-fixed-eval-v1`, 512 held-out
  transitions, with evaluation and unseen dynamics isolated from training.
  Results:

  | Updates | FB loss | Top-1 | Top-5 | Mean rank | Median rank | Margin | Entropy (nats) |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 0 | 1692189.5000 | 0.2656 | 0.7344 | 5.2969 | 3 | 846.7319 | 4.703526 |
  | 1 | 934192.1250 | 0.3750 | 0.7031 | 5.4531 | 2 | 1274.1001 | 4.853298 |
  | 10 | 1134625.5000 | 0.3125 | 0.7500 | 4.7031 | 2 | 769.3938 | 4.115609 |
  | 100 | 1793956.3750 | 0.3281 | 0.6875 | 5.2813 | 2 | 520.5139 | 3.558375 |

- The fixed BFM0 evaluator was repeated and produced identical metrics. The
  `update_1`, `update_10`, and `update_100` checkpoints are stored locally
  under `results/m2.2b-1/` and are not committed to Git.
- Base retention was not assigned a score. The official entrypoint starts
  without IsaacLab on this server; a one-environment MuJoCo fallback was
  started but stopped because it is not the protocol's 1024-environment
  evaluation and would not provide a comparable Base retention result.
- Native HUSKY termination, Qaux, and command-aligned downstream evaluation
  remain unresolved. Formal full training, Actor updates, BFB, RFB, and
  FB-MEBE remain disabled.

## 14. M2.2b-2 Baseline Reproducibility and Evaluator Fidelity Audit

- Date: 2026-08-10
- Evaluator fidelity: `PASS`. The current evaluator equations for
  `target_F`, `target_B`, `target_M`, `F`, `B`, `M`, `diff`, `fb_offdiag`,
  `fb_diag`, orthonormality terms, discount, and inactive `q_loss` match the
  vendored `FBAgent.update_fb()` source. The suspected `fb_diag` discrepancy
  was not confirmed: upstream uses
  `-diagonal(diff).mean() * num_parallel`, and the evaluator uses the same
  equation. Training loss was not modified.
- Canonical evaluator version remains `skate-bfm-fixed-eval-v1`; protocol
  conditions, seeds, rollout IDs, horizon, dynamics split, projection,
  entropy bins, context lengths, and Base evaluator configuration were not
  changed.
- Current official checkpoint identity:
  `model.safetensors` SHA256
  `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
  `config.json` SHA256 is
  `52f94d2946ed8912fc12ac9c25b4bf0e68ccdc669a05ea104e8b6c178e91fb46`;
  `init_kwargs.json` SHA256 is
  `b8df2d6006fbeda9a0bb9a9eb3f21dcccadf165f2252fce108714f81655a0094`.
- Current canonical provenance was generated at git commit
  `68548c999138e20d92563378ac13eb4f9e9e09ce`. The protocol SHA256 is
  `ebc20a7c22849d7ce9e27ec627f226d30fcce6bdbd94d5903862e03719efc16a`;
  the evaluator source SHA256 is
  `ea65dd5447e660c2f5e0972feb49620cb76d9325947b5437114e25958135150a`;
  the training entry SHA256 is
  `8fb9f17ad225781218cc1da2316cd44954e2ac122cef6842b82959d538bc3349`.
  Runtime fingerprint is
  `db91a56663a0512dabc981a9d1398d8b69c0e3348f5a159ba3c94f3e78feb7ba`
  (Python 3.12.13, PyTorch 2.5.1, CUDA 12.4, MuJoCo 3.11.0,
  NVIDIA GeForce RTX 4060 Ti, torch device CUDA, TF32 enabled, deterministic
  algorithms disabled).
- Clean-process reproducibility: `PASS`. Two fresh Python runs produced
  identical behavior coverage, eval buffer, buffer config, manifest, and
  metrics artifacts. The fixed held-out set remained 512 transitions.
- Historical revision reproduction: `PARTIAL`. A temporary worktree at
  `d4dccc13f760ebdc068ecbd24e57cc6bb67ced0a` ran successfully using the
  current official checkpoint bytes and the same HUSKY scene. Its metrics and
  coverage matched the current v1 evaluator, but it did not reproduce the
  old logged values `1775099.875 / 0.25 / 0.6875 / 5.703125 /
  4.6439129236`.
- Historical checkpoint identity is `UNVERIFIABLE`: M2.2b-0 did not record a
  checkpoint hash. The earliest auditable drift level is therefore Level 1,
  checkpoint identity; later levels cannot prove the original run used the
  same model bytes. The old recorded baseline is superseded for canonical
  provenance comparison, not deleted.
- The provenance now records loaded parameter and buffer fingerprints,
  resolved agent configuration hash, HUSKY source/scene/randomization hashes,
  rollout initial observation/first z/first action fingerprints, complete
  eval-buffer fingerprint, transition-ID fingerprint, fixed diagnostic
  indices, and diagnostic-batch fingerprint.
- Behavior entropy changes across B/F-only checkpoints even though Actor
  parameters are unchanged. The first z and initial observation fingerprints
  are the same, while first-action fingerprints differ; adaptation reports
  show the shared observation normalizer changed. Therefore the result is:
  parameter-frozen Actor `!=` functionally-frozen policy. The normalizer was
  not frozen because that would change upstream update semantics.
- Corrected v1 canonical metrics for update `0/1/10/100` are unchanged from
  the measured M2.2b-1 values and are recorded in `train_res.md` with the
  complete provenance policy. No training was rerun.

## 15. M2.2b-3 Base + Skate Expert Mixture Boundary

- Date: 2026-08-10
- Completed the intended Base+Skate 50/50 expert-mixture experiment as
  Experiment 0B, while retaining the historical Base-only Experiment 0A as
  the control.
- Used the official BFM0 checkpoint for three independent milestones:
  `update_1`, `update_10`, and `update_100`.
- Added fail-closed configuration validation so a nonzero Skate expert ratio
  cannot silently run without a configured Skate MotionLib artifact.
- Loaded 64 complete Base sequences and 64 complete Skate sequences with
  sequence length 8. The Skate artifact is
  `train/dataset/skate-expert-pose/motion_library/skate_expert.pkl`,
  SHA256
  `660c18145a21457d3541b49ccc802ba3f99170804836cedaebb9d245b837fd86`.
- Used 1024 Skate replay transitions for every milestone. The fixed replay
  tensor, transition IDs, and sampled HUSKY dynamics have fingerprints
  recorded in `train_res.md`.
- Directly invoked the vendored `update_fb()` only. F/B and target F/B
  changed; Actor, discriminator, QD, Qaux, and all forbidden optimizer calls
  remained unchanged.
- Evaluated all treatment checkpoints with
  `skate-bfm-fixed-eval-v1` on 512 held-out transitions. No training or
  evaluation leakage was detected.
- Result: mixed at updates 1 and 10, favorable at update 100 on FB loss,
  margin, Top-5, and mean rank. This is a representation-boundary result,
  not a downstream skate-task success claim.
- Base retention remains `NOT RUN`; native termination, Qaux, and
  command-aligned downstream evaluation remain unresolved. Full FB-CPR-Aux
  training remains `NO`.

## 16. M2.3a-0 Target Bank and Command Alignment Audit

- Date: 2026-08-11
- Audited the canonical 50-frame raw HUSKY rollout behind
  `skate_expert.pkl`, including robot root state, board state, joints, action,
  phase, timestamps, and command fields.
- Confirmed the raw rollout has 50 frames at 50 Hz, 0.98 s duration, finite
  values, no fall/reset frames, and one continuous `push` phase. No physically
  distinguishable steer-left, steer-right, or transition segment was found.
- Confirmed command semantics from the HUSKY source: `command_v` is the
  forward linear-velocity command scalar and is multiplied by `2.0` in the
  test-scene ONNX observation; `command_h` is the relative heading/steering
  command in radians, positive for left and negative for right.
- Confirmed both commands are present in raw metadata and every raw frame:
  `command_v=1.0`, `command_h=0.0`.
- Built one auditable 8-frame target, `skate_target_00`, frames `24-31`
  (`0.48-0.62 s`). It was selected from disjoint 8-frame candidates by
  highest board forward velocity subject to low lateral velocity, low board
  yaw change, and no fall.
- Physical evidence for the target: mean board forward velocity
  `0.8116 m/s`, mean lateral velocity `-0.0023 m/s`, board yaw delta
  `+0.3580 deg`, and no fall/reset frames. The target is `aligned` with the
  recorded forward/zero-heading command.
- Computed target latent inference for official BFM0, Base-only update100,
  and Base+Skate update100. All latent norms are approximately `16.0`.
  Base-only vs Base+Skate cosine similarity is `0.999963`; latent similarity
  was recorded as a diagnostic and was not used to assign the physical label.
- No training, rollout, Actor execution, or optimizer step was performed.
- Output:
  `train/dataset/skate-expert-pose/target_bank/target_bank.json`
- Full field inventory, raw hashes, checkpoint hashes, latent fingerprints,
  candidate windows, selection rule, and command evidence are recorded in the
  target bank JSON.

## 17. M2.3b-0 Frozen Actor Target-Conditioned Rollout Preflight

- Date: 2026-08-11
- Status: `completed_evaluation_only`
- Target: `skate_target_00`, frames `24-31`, the aligned forward-push
  window with `command_v=1.0` and `command_h=0.0`.
- Checkpoints: official BFM0, Base-only `update_100`, and Base+Skate
  `update_100`.
- Latent conditions: four fixed random latents with seeds
  `2026081101..2026081104`, plus one runtime-recomputed target latent per
  checkpoint. Every latent remained fixed for its complete rollout.
- Protocol: `skate-bfm-fixed-eval-v1`, four dynamics conditions
  (`seen_001`, `seen_002`, `unseen_001`, `unseen_002`), 128 steps, and
  `control_dt=0.02 s`. The HUSKY reset remained canonical; the simulator was
  not teleported to expert frame 24, and no command was injected into the
  BFM Actor input.
- Scale: `3 checkpoints x 4 dynamics x (4 random + 1 target) = 60`
  inference-only rollouts.
- Expert observations for frames `24-31` were loaded once from the Skate
  MotionLib and reused as CPU tensors. Each checkpoint then applied its
  native normalizer and backward map to recompute its own target latent.
  Runtime fingerprints matched the audited target-bank fingerprints:
  official `097548...`, Base-only `8cd703...`, Base+Skate `fc203e...`.
- Reproducibility: `PASS`. Initial observation, root-state, and board-state
  fingerprints were byte-identical across random/target conditions for each
  dynamics rollout ID.
- Frozen-state audit: `PASS`. Full model parameters, model components
  (`F`, `B`, target maps, Actor, discriminator, QD, and Qaux), and buffers
  were unchanged before/after all 20 rollouts for each checkpoint. The
  evaluation made `0` optimizer steps, `0` backward calls, `0` `agent.update`
  calls, and `0` `update_fb` calls.
- Physical metrics were projected in the initial board-heading frame and are
  recorded in `results/m2.3b-0-target-conditioned/target_conditioned_metrics.json`.
  This result directory is local and ignored because it contains generated
  evaluation artifacts.
- Directional result: the target increased forward displacement and mean
  forward velocity over the matched random mean in all six
  checkpoint-by-split comparisons. Lateral drift and heading drift did not
  improve consistently, so the overall target-conditioned response is
  `mixed`, with a consistent forward-response advantage but no task-success
  claim.
- Base-only versus Base+Skate: the adapted checkpoints had similar target
  forward response, while neither treatment produced a consistent stability
  advantage over the other. The representation result therefore does not
  establish a downstream physical advantage for Base+Skate.
- No training, Actor update, replay update, exploration change, action-format
  change, or evaluation-protocol-v1 change was performed.

## 18. M2.4-0 Project Code Cleanup

- Date: 2026-08-11
- Status: `completed_behavior_preserving_cleanup`
- Renamed current target-bank entrypoint
  `train/scripts/audit_skate_target_bank.py` to
  `train/scripts/build_target_bank.py`.
- Renamed current target-conditioned entrypoint
  `train/scripts/evaluate_skate_target_conditioned.py` to
  `train/scripts/eval_target.py`.
- Kept `train/scripts/evaluate_skate_bfm.py` unchanged as the canonical
  representation evaluator filename because its path is part of historical
  provenance records.
- Centralized checkpoint resolution/loading, model/buffer/component hashing,
  MotionLib expert loading, and runtime target encoding in the project-owned
  training runtime. The target entrypoints no longer import one another.
- Removed the confirmed unused `collect_skate_online_replay()` wrapper.
  The formal `Workspace` collection path and `HuskyBfmOnlineEnv` semantics
  were unchanged.
- No vendored file under `train/scripts/isaac_env/humanoidverse/` changed.
- Numerical regression passed:
  - target bank JSON byte-identical; target `skate_target_00`, frames `24-31`,
    physical fields, and all three latent fingerprints identical;
  - one target-conditioned rollout had identical reset, first-action,
    transition, and physical-metric fingerprints;
  - canonical BFM evaluation had identical 512-transition metrics and
    behavior coverage.
- Training performed: no. Optimizer steps: `0`.

## 19. M2.4a Full Training Dependency Audit

- Date: 2026-08-11
- Status: `completed_read_only_audit`
- Scope: determine what prevents one legal native
  `FBcprAuxAgent.update()` call. No training or update method was executed.
- Source of truth:
  - `train/scripts/isaac_env/humanoidverse/agents/fb/agent.py`
  - `train/scripts/isaac_env/humanoidverse/agents/fb_cpr/agent.py`
  - `train/scripts/isaac_env/humanoidverse/agents/fb_cpr_aux/agent.py`
  - `train/scripts/train_skate_bfm.py`
  - `src/skate_bfm/integration/`
  - `husky_sim/src/skate_husky/lite_env.py`

### Native Full-Update Call Graph

```text
expert_slicer.sample + train.sample
  -> device transfer + discount = 0.98 * ~terminated
  -> observation-normalizer update
  -> eval-mode normalization of train and expert observations
  -> encode_expert(expert_next_obs)
  -> update_discriminator(expert_obs/z, train_obs/z)
  -> sample_mixed_z -> z-buffer add -> probabilistic relabel
  -> update_fb(obs, action, discount, next_obs, goal, z)
  -> update_critic(obs, action, discount, next_obs, z)
  -> weighted train_batch["aux_rewards"]
  -> auxiliary reward EMA normalizer
  -> update_aux_critic(...)
  -> update_actor(QD + Qaux + F-derived Q)
  -> soft-update target F, B, QD, and Qaux
```

M2.2 `fb_only` stepped only the F and B optimizers and soft-updated target
F/B. M2.3 was frozen inference with no training. Those experiments establish
interface and representation boundaries, not expected full-update training
performance.

### Resolved Runtime Configuration

| Field | Value |
| :--- | :--- |
| Agent | `FBcprAuxAgent` |
| Batch / sequence length | `1024 / 8` |
| Discount | `0.98` |
| F / B learning rate | `3e-4 / 1e-5` |
| Actor / QD / Qaux learning rate | `3e-4 / 3e-4 / 3e-4` |
| Discriminator learning rate | `1e-5` |
| F/B target tau | `0.01` |
| QD/Qaux target tau | `0.005` |
| `expert_asm_ratio` / `train_goal_ratio` | `0.6 / 0.2` |
| `relabel_ratio` | `0.8` |
| `q_loss_coef` | `0.0` |
| `reg_coeff` / `reg_coeff_aux` | `0.05 / 0.02` |
| Discriminator gradient penalty | `10.0` |
| QD / Qaux / Actor pessimism | `0.5 / 0.5 / 0.5` |

Configured auxiliary rewards and scales:

```text
penalty_torques:            0.0
penalty_action_rate:       -0.1
limits_dof_pos:           -10.0
limits_torque:              0.0
penalty_undesired_contact: -1.0
penalty_feet_ori:          -0.4
penalty_ankle_roll:        -4.0
penalty_slippage:          -2.0
```

Zero-scaled keys remain mandatory because upstream accesses and logs every
configured key before applying its scale.

### Runtime Structural Audit

- Formal replay: `replay_buffer["train"] is replay_buffer["train_skate"]`.
- Replay size and sample: `1024` transitions and a full `1024` batch.
- Observation shapes:
  `state [1024,64]`, `privileged_state [1024,463]`,
  `last_action [1024,29]`, `history_actor [1024,372]`.
- Action / latent: `[1024,29]` and `[1024,256]`, finite `float32`.
- Latent norm min/mean/max:
  `15.999999 / 16.000000 / 16.000000`.
- Termination sample: `0` terminated and `1` horizon-truncated transition.
- `aux_rewards`: absent; no keys were silently synthesized.
- Expert Base source: `862` LAFAN motions.
- Expert Skate source: `1` motion, `50` frames, `50 Hz`, forward push,
  SHA256 `660c18145a21457d3541b49ccc802ba3f99170804836cedaebb9d245b837fd86`.
- Full expert sample: `128` complete 8-frame sequences, split exactly
  `64 Base + 64 Skate`.
- Forward-only finite shapes:
  - expert and mixed z: `[1024,256]`;
  - discriminator logits/reward: `[1024,1]`;
  - F and target F: `[2,1024,256]`;
  - B and target B: `[1024,256]`;
  - FB target matrix: `[2,1024,1024]`;
  - QD, target QD, Qaux, target Qaux: `[2,1024,1]`;
  - Actor output: `[1024,29]`.
- All six optimizers exist with one parameter group and zero state entries;
  none was stepped.
- Target parameter lists exist and match source shapes:
  F `76/76`, B `6/6`, QD `76/76`, Qaux `76/76`.
- Observation normalizer is checkpoint-loaded and was held in eval mode.
  Full upstream training intentionally updates it before eval normalization.
- Auxiliary reward normalizer exists with scalar `mean`, `mean_square`, and
  `counter` buffers and expects `[1024,1]`. Its `EMA.forward()` always mutates
  those buffers, so the audit inspected state without calling it.
- Full-model parameter hash before/after:
  `6e4c5279dee203d5c971b09269294d50482d35555d0fbd6c8890efd593c524fe`.
- Full-model buffer hash before/after:
  `5bd0b8dea2f792a3401b4341c9d04dc4a09c89fccf66f8ffdb825a8e14dd5dc5`.

### Termination Audit

- Current `HuskyBfmOnlineEnv.step()` always writes `terminated=False`.
- The fixed horizon writes `truncated=True`; fall, invalid state, and board
  separation do not currently terminate the transition.
- The wrapper refuses another step after termination/truncation until reset,
  so reset-crossing transitions cannot enter replay.
- Official HumanoidVerse computes reset conditions in
  `LeggedRobotBase._check_termination()` and maps
  `reset & ~time_outs` to terminated while mapping timeouts to truncated in
  `gymnasium_wrapper.py`.
- The resolved BFM0 config disables contact, gravity, low-height, limit,
  motion-end, and motion-far termination switches, so official configured
  behavior is also predominantly timeout truncation.
- Status: `PARTIAL`. The discount tensor is structurally valid and reset
  leakage is prevented, but no Skate-specific failure semantics exist.

### Auxiliary Reward Audit

**`penalty_torques`**

- Upstream: `LeggedRobotBase._reward_penalty_torques()`.
- Meaning/raw variables: sum of squared applied joint torques; `self.torques`.
- Current HUSKY source: actuator force is internal to MuJoCo but is not
  exposed by `HuskyLiteEnv`.
- Current replay: no torque or reward field.
- Exact / approximate: `NO / NO`.
- Status: `BLOCKED`.

**`penalty_action_rate`**

- Upstream: `LeggedRobotBase._reward_penalty_action_rate()`.
- Meaning/raw variables: squared difference between previous and current
  29-DoF actions.
- Current HUSKY source: current stored BFM action and previous BFM action are
  available through the action/history contract.
- Current replay: source values exist, but the scalar reward is not emitted.
- Exact / approximate: `YES / YES` under the existing BFM action convention.
- Status: `PARTIAL`.

**`limits_dof_pos`**

- Upstream: `LeggedRobotBase._reward_limits_dof_pos()`.
- Meaning/raw variables: 29-DoF position violation against resolved soft
  limits and optional curriculum state.
- Current HUSKY source: 23 physical joint positions and MuJoCo limits exist;
  the BFM observation expands them to 29 dimensions.
- Current replay: expanded joint state exists, but exact official 29-DoF
  limits/curriculum realization and reward are absent.
- Exact / approximate: `NO / YES`.
- Status: `PARTIAL`.

**`limits_torque`**

- Upstream: `LeggedRobotBase._reward_limits_torque()`.
- Meaning/raw variables: torque excess over resolved soft torque limits.
- Current HUSKY source/replay: neither applied torque nor the official
  29-DoF soft-limit realization is exposed or stored.
- Exact / approximate: `NO / NO`.
- Status: `BLOCKED`.

**`penalty_undesired_contact`**

- Upstream:
  `LeggedRobotMotions._reward_penalty_undesired_contact()`.
- Meaning/raw variables: whether any configured penalized body has contact
  force magnitude above `1`.
- Current HUSKY source: MuJoCo contacts exist internally, but the penalized
  body mapping and contact forces are not exposed.
- Current replay: no contact data.
- Exact / approximate: `NO / YES` only through a non-equivalent pose/contact
  heuristic.
- Status: `BLOCKED`.

**`penalty_feet_ori`**

- Upstream: `LeggedRobotBase._reward_penalty_feet_ori()`.
- Meaning/raw variables: left/right foot tilt, gated by vertical contact force
  above `1`.
- Current HUSKY source: raw body orientation exists; exact contact gating is
  missing.
- Current replay: neither foot orientation nor contact force is stored.
- Exact / approximate: `NO / YES` without equivalent contact gating.
- Status: `BLOCKED`.

**`penalty_ankle_roll`**

- Upstream: `LeggedRobotMotions._reward_penalty_ankle_roll()`.
- Meaning/raw variables: squared left/right ankle-roll positions.
- Current HUSKY source: both physical ankle-roll joint positions exist and
  enter the BFM observation.
- Current replay: source state exists, but the scalar reward is not emitted.
- Exact / approximate: `YES / YES` under the current joint adapter.
- Status: `PARTIAL`.

**`penalty_slippage`**

- Upstream: `LeggedRobotBase._reward_penalty_slippage()`.
- Meaning/raw variables: foot velocity magnitude gated by foot contact force
  magnitude above `1`.
- Current HUSKY source: body velocity exists; exact contact-force gating is
  missing.
- Current replay: neither foot velocity nor contact force is stored.
- Exact / approximate: `NO / YES` using a non-equivalent kinematic contact
  heuristic.
- Status: `BLOCKED`.

### Full Training Dependency Matrix

| Dependency | Required by | Current source | Status | Hard blocker? | Next action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Base expert | expert sampling | LAFAN, 862 motions | READY | No | Retain |
| Skate expert | expert sampling | 1 x 50-frame push | READY | No | Expand later |
| `expert_slicer` | update start | 64/64 sequence sampler | READY | No | Retain |
| Train replay | all updates | 1024 formal transitions | PARTIAL | Yes | Add exact aux contract |
| Observation | all networks | 64/463/29/372 fields | READY | No | Retain |
| Action | F/QD/Qaux/Actor | stored 29D | READY | No | Retain |
| z | discriminator/F/B/critics | stored 256D | READY | No | Retain |
| terminated | discount | always false | PARTIAL | No | Audit semantics later |
| truncated | episode boundary | bounded horizon | READY | No | Retain |
| Observation normalizer | all networks | official checkpoint | READY | No | Retain |
| Discriminator | CPR reward | official checkpoint | READY | No | Retain |
| F | representation/Actor | official checkpoint | READY | No | Retain |
| B | encoding/representation | official checkpoint | READY | No | Retain |
| Main critic / QD | CPR/Actor | official checkpoint | READY | No | Retain |
| Auxiliary rewards | Qaux | absent | BLOCKED | Yes | M2.4b |
| Aux reward normalizer | Qaux | checkpointed scalar EMA | READY | No | Feed exact scalar |
| Qaux network | Qaux/Actor | official checkpoint | READY | No | Retain |
| Qaux data | Qaux/Actor | absent aux rewards | BLOCKED | Yes | M2.4b |
| Actor network | action output | official checkpoint, 29D | READY | No | Retain |
| Actor training interface | full Actor loss | depends on Qaux data | BLOCKED | Yes | Unblock Qaux data |
| Target F / B | FB targets | shape-matched | READY | No | Retain |
| Target QD / Qaux | critic targets | shape-matched | READY | No | Retain |

### Judgment

- Representation training ready: `YES`.
- Critic/discriminator interface ready: `YES`.
- Actor training interface ready: `NO`.
- Full `FBcprAuxAgent.update()` ready: `NO`.
- Earliest independent hard blocker: configured Skate auxiliary reward data
  is absent/incomplete.
- Performance limitation, not a technical blocker: the Skate expert set is
  one 50-frame forward-push motion. It is sufficient for technical training
  feasibility smoke and insufficient for final skill coverage.
- Next milestone: `M2.4b — Skate Auxiliary Reward Contract`.
- Parameter mutation: `NO`; model-buffer mutation: `NO`.
- Optimizer steps: `0`; backward calls: `0`; `agent.update`: `0`;
  `update_fb`: `0`; `update_actor`: `0`; training: `NO`.

### Code Change Summary

1. `train/scripts/audit_training.py`

   - Changed: added the retained read-only M2.4a audit entrypoint.
   - Why: reproduce resolved config, real replay/expert batches, network
     forward contracts, optimizer/target presence, and mutation checks.
   - Original logic: no project-owned full-training audit entrypoint.
   - New logic: collect-only construction plus `torch.no_grad()` inspection;
     all update methods remain uncalled.
   - Algorithm behavior changed: `NO`.
   - Affected module: training-readiness tooling only.

2. `README.md` and `train/README.md`

   - Changed: advanced project status to completed M2.4a, documented the hard
     blocker, and added the audit command.
   - Why: keep branch and project progress aligned with verified evidence.
   - Original logic: M2.4a was listed as next.
   - New logic: M2.4b auxiliary reward contract is next.
   - Algorithm behavior changed: `NO`.
   - Affected module: documentation only.

3. `docs/assets/project_progress.svg` and
   `docs/assets/development_substage.svg`

   - Changed: moved the visual current position to M2.4a and marked full
     training as blocked by auxiliary reward data.
   - Why: keep both progress figures synchronized with the audit.
   - Original logic: M2.4-0 current, M2.4a next.
   - New logic: M2.4a complete, M2.4b next.
   - Algorithm behavior changed: `NO`.
   - Affected module: documentation assets only.

4. `train/train_log.md` and `train/train_res.md`

   - Changed: retained the call graph, resolved config, contracts, reward
     audit, readiness matrix, validation, and final judgments.
   - Why: preserve the M2.4a evidence without retaining generated results.
   - Original logic: records ended at M2.4-0.
   - New logic: M2.4a is recorded as a non-training readiness audit.
   - Algorithm behavior changed: `NO`.
   - Affected module: training records only.

### Overall Code Change Summary

- Model architecture changed: `NO`.
- Loss changed: `NO`.
- Optimizer behavior changed: `NO`.
- Training loop changed: `NO`.
- Expert sampling changed: `NO`.
- Online latent sampling changed: `NO`.
- Exploration changed: `NO`.
- Replay semantics changed: `NO`.
- Observation format changed: `NO`.
- Action format changed: `NO`.
- Termination semantics changed: `NO`.
- Auxiliary reward semantics changed: `NO`.
- Evaluation protocol changed: `NO`.
- Training-readiness audit added: `YES`.
- Vendored BFM-Zero source modified: `NO`.
- Training performed: `NO`.

## 20. M2.4b-1 Phase-wise Expert Reward Audit

- Date: 2026-08-11
- Status: `completed_read_only_audit`
- Scope: phase-wise semantics of the eight currently configured BFM auxiliary
  rewards. No training, replay mutation, optimizer step, backward call, or
  agent update was performed.
- Retained diagnostic entrypoint: `train/scripts/audit_rewards.py`.
- Source of truth:
  - official reward definitions in vendored
    `legged_robot_base.py` and `legged_robot_motions.py`;
  - current `aux_rewards_scaling` in `train/scripts/train_skate_bfm.py`;
  - original raw rollout phase mapping, state, action, and board fields;
  - HUSKY's generated MuJoCo XML and recorded physics randomization.

### Expert Phase Structure

The current tracked MotionLib expert remains one 50-frame forward push.
It cannot answer phase-wise steering questions. This read-only diagnostic
therefore used two separate phase-rich, recorded HUSKY policy rollouts and
did not add them to MotionLib or the formal training replay.

| Raw rollout | Phase | Frame range | Frames | Duration (s) |
| :--- | :--- | :--- | ---: | ---: |
| `round998_rollout9981_h_pos020` | push | `[0,120)`, `[299,350)` | `171` | `3.42` |
| same | push2steer | `[120,149)` | `29` | `0.58` |
| same | steer_left | `[149,284)` | `135` | `2.70` |
| same | steer2push | `[284,299)` | `15` | `0.30` |
| `round998_rollout9982_h_neg020` | push | `[0,120)`, `[299,350)` | `171` | `3.42` |
| same | push2steer | `[120,149)` | `29` | `0.58` |
| same | steer_right | `[149,284)` | `135` | `2.70` |
| same | steer2push | `[284,299)` | `15` | `0.30` |

Aggregate requested groups: push `342` frames (`6.84 s`), push2steer `58`
(`1.16 s`), steer `270` (`5.40 s`, preserving left/right source labels), and
steer2push `30` (`0.60 s`). `steer_forward` and `fall` are
`PHASE NOT AVAILABLE`.

### Phase-Local Replay Fidelity

Each phase was independently replayed from its recorded start `qpos/qvel`;
recorded `action[t]` was applied to reconstruct `state[t+1]`. This avoids
contact-solver drift from one phase contaminating later phase evidence.

- `round998_rollout9981_h_pos020`: `PASS`; aggregate RMSE joint position
  `6.79e-6 rad`, root position `6.23e-6 m`, board position `5.56e-6 m`,
  board linear velocity `1.93e-5 m/s`.
- `round998_rollout9982_h_neg020`: `PASS`; aggregate RMSE joint position
  `6.04e-6 rad`, root position `4.13e-6 m`, board position `4.99e-6 m`,
  board linear velocity `1.64e-5 m/s`.
- Per-phase root/joint/board position, root/board orientation, and board
  velocity all passed. The detailed RMSE and max-error report is retained in
  ignored `results/m2.4b-1-reward-audit/summary.json`.

### Key Phase Statistics

Values are phase means. Full count, mean, standard deviation, p50, p90, max,
and nonzero fraction for every trace field are in the generated JSON.

| Phase | Action rate 29D | World slip | Board-relative slip | World feet ori | Surface feet ori | Ankle roll | Original weighted aux | Surface candidate aux |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| push | `30.070` | `1.028` | `0.020` | `0.035` | `0.040` | `0.011` | `-5.130` | `-3.206` |
| push2steer | `66.355` | `2.883` | `0.202` | `0.122` | `0.113` | `0.013` | `-12.501` | `-7.204` |
| steer | `1.496` | `3.126` | `0.044` | `0.297` | `0.294` | `0.024` | `-6.615` | `-0.450` |
| steer2push | `46.363` | `1.754` | `0.204` | `0.075` | `0.085` | `0.013` | `-8.228` | `-5.275` |

- 29D action rate uses the BFM action convention (`5x` normalized action);
  23D executed target-rate statistics are also retained. The six omitted
  BFM wrist dimensions contribute exactly `0`.
- Transition action-rate spikes are expected: 29D mean is `66.355` in
  push2steer and `46.363` in steer2push, versus `1.496` during steady steer.
- `limits_dof_pos` is zero in all transition/steer frames and only sparse in
  push (`1.46%` nonzero); `limits_torque` is sparse push-only. Both are
  23-to-29 mapped physical diagnostics.
- `penalty_torques` phase means are push `3768`, push2steer `5936`, steer
  `1423`, steer2push `6206`; it and `limits_torque` retain zero Qaux scale.
- No penalized pelvis/shoulder/hip contact was observed. Contact pairs show
  expected foot-ground and foot-board switching in push/transitions and
  two board-supported feet throughout steady left/right steer.

### Semantic Conclusions

- **PUSH:** world slip includes the support foot's legitimate skateboard
  transport. Surface-relative slip is much lower; ground push contact is
  separately represented.
- **PUSH2STEER:** action rate and board-relative slip have short transition
  spikes. These are acceptable transient costs, not evidence against the
  action-rate reward.
- **STEER:** world slip is continuously high (`3.126`) while board-relative
  slip is low (`0.044`). The original world-frame slippage term contributes
  about `-6.25` of the `-6.615` original weighted auxiliary objective.
  This is a systematic semantic conflict, not a transition spike.
- **STEER2PUSH:** action-rate and board-relative-slip increases are transition
  costs. The candidate remains negative mainly because switching is active;
  no new sustained conflict was identified.
- Feet orientation is sustained in steer, but world and contact-surface
  versions are nearly identical (`0.297` versus `0.294`). Current evidence
  does not support a frame-definition conflict.
- Ankle-roll penalty is roughly twice push during steer (`0.024` versus
  `0.011`), while board roll is small and not consistently correlated across
  left/right rollouts. It requires an ablation before retaining or removing.

### Reward Semantic Matrix

| Reward | Classification | Evidence |
| :--- | :--- | :--- |
| `penalty_torques` | `DIAGNOSTIC_ONLY` | reliable HUSKY torque query, zero current scale, 23-to-29 mapping |
| `penalty_action_rate` | `KEEP_WITH_MAPPING` | transition spikes; steady steer is low; wrist contribution is zero |
| `limits_dof_pos` | `KEEP_WITH_MAPPING` | sparse/near-zero mapped limit violation |
| `limits_torque` | `DIAGNOSTIC_ONLY` | mapped effort limits and zero current scale |
| `penalty_undesired_contact` | `KEEP` | no penalized non-foot contact in either rollout |
| `penalty_feet_ori` | `KEEP` | surface frame does not materially change steer penalty |
| `penalty_ankle_roll` | `ABLATION_REQUIRED` | sustained steering increase without consistent board-roll coupling |
| `penalty_slippage` | `REDEFINE` | world-frame term penalizes legitimate board transport |

### Outputs and Boundary

- Ignored output directory:
  `results/m2.4b-1-reward-audit/`.
- Trace: `expert_reward_trace.csv`, containing all required raw/original,
  surface-relative, and candidate columns; unavailable values would be
  `NaN`.
- Figures: `reward_traces.png`, `slippage_comparison.png`,
  `feet_orientation_comparison.png`, and `ankle_roll_board_roll.png`.
- Full raw-expert video: not available; no replacement video was generated.
- Formal replay modified: `NO`.
- Aux reward training semantics changed: `NO`.
- Full `FBcprAuxAgent.update()` ready: `NO`.
- Next milestone: `M2.4b-2 — Skate Aux Reward Contract`.

### Code Change Summary

1. `train/scripts/audit_rewards.py`

   - Changed: added a phase-local raw-rollout reward diagnostic.
   - Why: audit original auxiliary semantics before defining any replay
     reward contract.
   - Original logic: no phase-wise auxiliary reward audit.
   - New logic: validates phase-local MuJoCo replay fidelity, emits a
     frame-level trace, phase statistics, contact pairs, and four figures.
   - Algorithm behavior changed: `NO`.
   - Affected module: read-only training diagnostics only.

2. `README.md`, `train/README.md`, `train/train_log.md`, and
   `train/train_res.md`

   - Changed: recorded M2.4b-1 evidence, command, semantic matrix, and next
     milestone.
   - Why: synchronize project status with the reward audit.
   - Original logic: M2.4a dependency audit was current.
   - New logic: M2.4b-1 semantics audit completed; M2.4b-2 is next.
   - Algorithm behavior changed: `NO`.
   - Affected module: documentation only.

### Overall Code Change Summary

- Model architecture changed: `NO`.
- Loss changed: `NO`.
- Optimizer behavior changed: `NO`.
- Training loop changed: `NO`.
- Expert sampling changed: `NO`.
- Online latent sampling changed: `NO`.
- Exploration changed: `NO`.
- Replay semantics changed: `NO`.
- Observation format changed: `NO`.
- Action format changed: `NO`.
- Termination changed: `NO`.
- Aux reward training semantics changed: `NO`.
- Aux reward diagnostics added: `YES`.
- Evaluation protocol changed: `NO`.
- Training performed: `NO`.

## 21. M2.4b-2 Skate Auxiliary Reward Contract

- Date: 2026-08-11
- Status: `completed_collect_only_validation`
- Scope: add the eight upstream-required raw auxiliary penalties to HUSKY
  post-step transitions and formal Skate replay. No task reward, model update,
  optimizer step, backward pass, termination change, or reward-normalizer
  update was performed.
- Reward formula authority: vendored BFM-Zero.
- Physical position and torque constraint authority: the active HUSKY MuJoCo
  `MjModel`, queried at runtime.

### Runtime Physical Mapping

All 23 HUSKY robot actuators passed fail-closed validation:

- one named actuator maps to one named hinge joint;
- transmission type is `mjTRN_JOINT`;
- `gear=[1, 0, 0, 0, 0, 0]`;
- force limiting is enabled with finite symmetric force ranges;
- `qfrc_actuator[joint_dof]` is therefore the same physical generalized-torque
  quantity as the derived `forcerange * gear[0]` joint limit.

All 23 physical HUSKY position ranges match their name-mapped upstream G1
position ranges. The upstream torque YAML files are not physical-limit sources:
`g1_29dof_hard_waist` differs at both hip-pitch limits (`139` upstream versus
`88` HUSKY), while `g1_29dof` differs at both hip-roll limits (`88` upstream
versus `139` HUSKY). HUSKY runtime limits are authoritative for the physical
23D reward, so neither mismatch is used as a fallback or blocker.

### Implemented Contract

- `penalty_torques`: sum of squared actual 23D `qfrc_actuator` torques.
- `penalty_action_rate`: sum of squared current minus previous executed,
  clipped 23D normalized HUSKY actions; previous action resets to zeros.
- `limits_dof_pos`: 95% soft-range violation over the actual 23 HUSKY
  position ranges.
- `limits_torque`: 95% derived actual 23D joint-torque-limit violation.
- `penalty_undesired_contact`: binary force-thresholded pelvis/shoulder/hip
  contact with ground or skateboard. Foot-ground and foot-board contact are
  allowed.
- `penalty_feet_ori`: original, contact-gated world-horizontal foot-normal
  penalty.
- `penalty_ankle_roll`: original squared left plus right ankle-roll position.
- `penalty_slippage`: dominant per-foot contact's surface-relative tangential
  velocity. Ground is stationary; board velocity is evaluated at the actual
  contact point from the board surface body.

Raw penalty values are positive and unscaled in the environment and replay.
The upstream agent still owns the weighted sum. Existing scales remain:
action rate `-0.1`, feet orientation `-0.4`, ankle roll `-4.0`, DoF limit
`-10.0`, slippage `-2.0`, undesired contact `-1.0`, torque `0.0`, and torque
limit `0.0`.

### Transition and Replay Validation

- `HuskyLiteEnv.reset()` clears previous executed action and returns a
  zero-valued 8-key auxiliary dictionary.
- `HuskyLiteEnv.step()` applies the action, advances MuJoCo, computes the raw
  post-step penalties, then advances the previous-action state.
- `SkateOnlineTransition` stores the 8-key reward snapshot without changing
  observation, 29D replay action, 23D executed action, z, or truncation
  semantics.
- `as_buffer_data()` writes every key as a float `[1,1]` tensor.
- Formal collect-only replay: `1024` transitions; `train is train_skate`:
  `YES`; all eight sampled fields are finite `[1024,1]`.
- The complete distributions and weighted raw sanity statistic are in
  [`train_res.md`](train_res.md); the ignored machine-readable report is
  `results/m2.4b-2-reward-contract/training_readiness.json`.

### M2.4b-1 Regression

The two phase-rich HUSKY raw policy rollouts were replayed with phase-local
fidelity `PASS`. Production and audit calculations agreed over all 700 frames
for action rate, DoF limit, undesired contact, world-horizontal feet
orientation, ankle roll, and surface-relative slippage.

- Left steer: world slip `3.14840`; production surface-relative slip
  `0.04082`.
- Right steer: world slip `3.10389`; production surface-relative slip
  `0.04815`.

The historical raw policy archive contains unclipped controls, so the
state-by-state regression reproduces those original targets to recover its
post-step contact state while comparing action-rate with the new clipped 23D
production convention. This preserves the M2.4b-1 conclusion without changing
the formal Skate runtime action contract.

### Code Change Summary

1. `husky_sim/src/skate_husky/lite_env.py`

   - Changed: added fail-closed physical actuator mapping and post-step
     eight-key raw auxiliary reward calculation.
   - Why: formal Skate replay requires physical HUSKY reward data.
   - Original logic: environment exposed only observations and last action.
   - New logic: reset clears reward/action state; step records original BFM
     penalties with surface-relative slippage and HUSKY physical constraints.
   - Algorithm behavior changed: `YES`, replay now has auxiliary reward data.
   - Affected module: HUSKY MuJoCo runtime and training data collection.

2. `src/skate_bfm/integration/online.py`

   - Changed: added an auxiliary reward snapshot to Skate online transitions
     and buffer serialization.
   - Why: preserve post-step environment penalties with their transition.
   - Original logic: no `aux_rewards` replay field.
   - New logic: all eight keys are serialized as `[1,1]` tensors.
   - Algorithm behavior changed: `YES`, replay schema now contains rewards.
   - Affected module: HUSKY-to-BFM online replay boundary.

3. `train/scripts/audit_training.py` and `train/scripts/audit_rewards.py`

   - Changed: added read-only 1024-replay reward distributions, physical
     actuator reporting, and production-versus-phase-audit regression.
   - Why: validate the reward contract without training.
   - Original logic: reward fields were absent and phase audit was
     diagnostic-only.
   - New logic: collect-only replay and phase-rich states validate production
     reward calculations.
   - Algorithm behavior changed: `NO`.
   - Affected module: read-only validation only.

4. `tests/test_integration.py`, `README.md`, `train/README.md`,
   `train/train_log.md`, `train/train_res.md`, and progress SVGs

   - Changed: added contract regression tests and synchronized project status.
   - Why: retain reproducible evidence and the current readiness boundary.
   - Original logic: auxiliary reward data was blocked.
   - New logic: replay/Qaux/Actor interfaces are ready; termination is the
     remaining blocker.
   - Algorithm behavior changed: `NO` for documentation and tests.
   - Affected module: validation and documentation.

### Overall Code Change Summary

- Model architecture changed: `NO`.
- Loss changed: `NO`.
- Optimizer behavior changed: `NO`.
- Training loop changed: `NO`.
- Expert sampling changed: `NO`.
- Online latent sampling changed: `NO`.
- Exploration changed: `NO`.
- Replay semantics changed: `YES`, real `aux_rewards` are stored.
- Observation format changed: `NO`.
- Action format changed: `NO`.
- Termination semantics changed: `NO`.
- Aux reward training semantics changed: `YES`, physical Skate reward
  contract added with contact-surface-relative slippage.
- Aux reward scaling changed: `NO`.
- Evaluation protocol changed: `NO`.
- Training performed: `NO`.
- Next milestone: `M2.4c — Native Termination Contract`.

## M2.4c Native Fall Termination Contract

- Date: `2026-08-11`
- Moved the existing Skate expert-collection `LiveFallDetector` into the
  production HUSKY runtime so collection and online replay use one
  implementation.
- Online post-step semantics: a persistent severe tilt (`>70 deg`) or
  persistent low root height (`<0.45 m`) plus illegal contact writes
  `terminated=True`; the 0.2 s confirmation window is `10` control frames at
  50 Hz. The fixed collection horizon writes `truncated=True`.
- A confirmed fall takes precedence on a shared final step:
  `terminated=True`, `truncated=False`; a subsequent step requires reset.
- Temporary foot lift-off and board separation are diagnostics only and do not
  terminate an episode. Fall recovery is outside the current task.
- Completed a 1024-transition collect-only audit: 14 terminated, 1 truncated,
  and 1009 normal transitions, with no termination/truncation overlap.
- No model update, optimizer step, backward call, normalizer update, or
  parameter/buffer mutation occurred.
- Next milestone: `M2.4d — Native Full-Update Smoke`.

## M2.4d-1 Native Full-Update Smoke

- Date: `2026-08-11`
- Added fail-closed Skate `full` mode: it requires the official BFM0
  checkpoint, `collect_only=False`, exactly 1 update, exactly 1024 HUSKY
  transitions, and a 64 Base / 64 Skate expert sequence mixture.
- Ran exactly one vendored `FBcprAuxAgent.update()` with all six optimizers.
- Replay result: 14 terminal falls, 1 horizon truncation, and 1009 normal
  transitions; `train is train_skate` and no reset-crossing transition.
- All native metrics, updated online/target modules, optimizer state, and
  normalizer buffers were finite. The z-buffer changed from 0 to 1024 entries.
- The smoke work directory contains diagnostics only; no checkpoint was saved
  or retained as a training baseline.
- Next milestone: `M2.4d-2 — Short Multi-Update Stability Smoke`.

## M2.4d-2 Short Multi-Update Stability Smoke

- Date: `2026-08-12`
- Added the fail-closed `full` boundary for exactly 10 updates, while keeping
  1 and 10 as the only supported smoke counts.
- Collected one fixed 1024-transition HUSKY replay, then reused it for exactly
  10 native `FBcprAuxAgent.update()` calls. Expert sampling was repeated each
  update with 64 Base and 64 Skate complete sequences.
- Replay: 14 terminal, 1 truncated, 1009 normal; no reset-crossing
  transitions.
- All 10 metric dictionaries were finite. All six optimizer states reached
  Adam step 10; all online and target modules changed and remained finite.
- Observation and auxiliary-reward normalizers remained finite at every update.
  The z-buffer sizes were `1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192,
  8192, 8192` with capacity `8192`.
- No checkpoint or performance evaluation was produced.
- Next milestone: `M2.4d-3 — 100-Update Stability Smoke`.

## M2.4d-3 100-Update Stability Smoke

- Date: `2026-08-12`
- Extended fail-closed `full` smoke mode to the only allowed counts `1`, `10`,
  and `100`.
- Collected one fixed 1024-transition HUSKY replay and executed exactly 100
  native `FBcprAuxAgent.update()` calls with a fresh official BFM0 load.
- All 100 returned metric dictionaries, online/target module states, optimizer
  states, and normalizer states remained finite. No 100x scale warning was
  triggered for monitored losses or Q metrics.
- All six Adam optimizers reached step 100. The z-buffer saturated normally at
  its configured capacity 8192 and remained finite.
- The fixed-replay experiment is diagnostic only: no checkpoint, online
  collect-update alternation, or skating-performance evaluation was used.
- M2.4 training preparation: `COMPLETE`.
- Next milestone: `M2.5 — Original BFM-Zero Skate Baseline`, restarting from
  the official checkpoint.

## M2.5a Native Closed-Loop Baseline Bring-Up

- Date: `2026-08-12`
- Restarted from the strict official BFM0 checkpoint, SHA256
  `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- Added the minimal `full + adaptation_updates=0` route in the project-owned
  training entry. Existing `full + {1, 10, 100}` fixed-replay smokes remain
  unchanged.
- Executed 2,000 nominal-HUSKY online transitions with one growing Skate
  replay. The first 1,500 transitions used A0, then 50 native updates produced
  A1; transitions 1,501-2,000 were collected by A1 before the final 50 native
  updates produced A2.
- The 1,024-row warmup uses the official pretrained Actor's native stochastic
  action distribution and `model.sample_z()`. It does not use the original
  Base-training random-action seed phase. Episode reset also clears the
  rollout latent before a new one is sampled.
- No randomization, loss, optimizer, reward, termination, expert-mixture,
  observation, action, or replay-schema change was made. No checkpoint was
  saved. This is a closed-loop health check, not a skating-quality result.
- Next milestone: `M2.5b — Original BFM-Zero Skate Baseline Training`.

### Code Change Summary

1. `train/scripts/train_skate_bfm.py`

   - Changed: added the `full + adaptation_updates=0` closed-loop collection
     and two native-update blocks.
   - Why: prove that updated-Actor HUSKY data re-enters the replay before a
     subsequent untouched vendored native update.
   - Original logic: `full` supported only 1/10/100 fixed-replay diagnostics.
   - New logic: zero selects the fixed 2,000-transition M2.5a schedule;
     positive supported values preserve the existing smoke path.
   - Algorithm behavior changed: `YES`, only the project training-loop
     schedule and online action sampling for M2.5a.
   - Affected module: project-owned Skate training entry.

2. `tests/test_integration.py`

   - Changed: covered full-mode routing for 0, 1, 10, and 100 updates.
   - Why: prevent the M2.5a path from replacing M2.4 smoke behavior.
   - Original logic: configuration-only full-mode coverage.
   - New logic: zero dispatches to closed loop; positive supported values
     dispatch to the existing smoke method and retain `train is train_skate`.
   - Algorithm behavior changed: `NO`.
   - Affected module: regression validation.

### Overall Code Change Summary

- Model architecture changed: `NO`.
- Loss changed: `NO`.
- Optimizer configuration changed: `NO`.
- Native update algorithm changed: `NO`.
- Training loop changed: `YES`; zero updates selects online
  collect/update/collect/update instead of a fixed replay smoke.
- Expert sampling and Expert MotionLib changed: `NO`.
- Online latent sampling changed: `NO`; retained `sample_z()`.
- Online action sampling changed: `YES`; M2.5a uses the native stochastic
  Actor distribution while diagnostic smoke collection remains deterministic.
- Exploration algorithm changed: `NO`; no wrapper noise or random-action
  warmup was added.
- Replay distribution behavior changed: `YES`; A1 data enters the replay.
- Replay schema, observations, actions, rewards, reward scaling, termination,
  domain randomization, evaluation protocol, and vendored BFM-Zero changed:
  `NO`.
- Training performed: `YES`; 2,000 transitions and 100 native updates.
- Performance conclusion: `NO`.

## M2.5b Original BFM-Zero Skate Baseline Training

- Date: `2026-08-12`
- Ran the first real 20,000-transition Skate baseline from a fresh official
  BFM0 load. The source model SHA256 was
  `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- Reused the M2.5a closed-loop implementation with the only schedule extension:
  38 blocks from transition 1,500 through 20,000, every 500 transitions, at
  50 untouched native `FBcprAuxAgent.update()` calls per block.
- Kept pretrained stochastic Actor warmup for 1,024 rows; no random-action
  seed phase, added exploration, domain randomization, reward change, model
  change, optimizer change, expert change, or termination change.
- Reached 20,000 replay rows with 19,592 normal, 389 confirmed terminal-fall,
  and 19 horizon-truncated transitions. `train is train_skate` remained true;
  reset-crossing transitions remained zero.
- Saved and reloaded native checkpoints at 10k and 20k. Both reloads matched
  model/normalizer state and all six optimizer states. The saved model SHA256
  values are recorded in `train_res.md`.
- Ran the existing fixed target-conditioned evaluator after training only.
  Evaluation generated 60 local rollout records and made zero optimizer,
  backward, agent-update, or replay writes. Each checkpoint used its own
  observation normalizer and B map to re-encode the same target window.
- All 60 fixed evaluation episodes reached the confirmed-fall terminal state
  before 128 steps. The result is therefore recorded as `INCONCLUSIVE`; board
  displacement is not interpreted as skate-task success.
- Next milestone: `M2.5c — Baseline Extension / Domain-Randomization Decision`.

### Code Change Summary

1. `train/scripts/train_skate_bfm.py`

   - Changed: minimally parameterized the verified closed-loop route for 2k
     bring-up or 20k baseline budgets, with fixed update schedule derivation,
     10k/20k native checkpoint save/reload verification, and block/episode
     summaries.
   - Why: execute the first long closed-loop baseline without duplicating the
     M2.5a loop or vendored training implementation.
   - Original logic: only the fixed 2,000-transition M2.5a bring-up schedule.
   - New logic: zero-update `full` mode accepts only 2k or 20k; positive
     1/10/100 counts retain the fixed-replay smoke path.
   - Algorithm behavior changed: `YES`, training duration/checkpoint schedule
     only.
   - Affected module: project-owned Skate training entry.

2. `train/scripts/eval_target.py`

   - Changed: accepts official/10k/20k checkpoint paths, recomputes each
     checkpoint's target latent, and ends a fixed evaluation rollout after its
     terminal fall transition.
   - Why: native M2.4c termination means an evaluator must not step across a
     completed episode; long-baseline checkpoints require independent latent
     inference.
   - Original logic: used three historical checkpoint names and assumed every
     rollout survived the full 128 steps.
   - New logic: preserves the fixed protocol and inference-only checks while
     recording actual terminal episode duration.
   - Algorithm behavior changed: `NO`.
   - Affected module: read-only fixed evaluation.

3. `tests/test_integration.py`

   - Changed: verifies 20k schedule, 38 update blocks, 1900 native updates,
     checkpoint steps, 2k compatibility, and unsupported-budget failure.
   - Why: retain the M2.5a/M2.4 routes while preventing schedule drift.
   - Original logic: tested M2.5a configuration/routing only.
   - New logic: adds lightweight schedule contract coverage without training.
   - Algorithm behavior changed: `NO`.
   - Affected module: regression validation.

### Overall Code Change Summary

- Model architecture, loss, optimizer configuration, native update algorithm,
  expert sampling, Expert MotionLib, online latent sampling, exploration,
  replay schema, observation/action format, auxiliary reward semantics/scaling,
  termination, and vendored BFM-Zero: `NO` change.
- Training loop: `YES`; the verified M2.5a loop runs to 20k.
- Replay distribution: `YES`; updated-Actor data grows to 20k rows.
- Domain randomization: `NO`; nominal HUSKY only.
- Checkpoint behavior: `YES`; complete native checkpoints at 10k and 20k.
- Evaluation protocol: `NO`; existing fixed evaluator, now terminal-safe.
- Training performed: `YES`; 20,000 transitions and 1,900 native updates.
- Performance evaluated: `YES`; fixed offline evaluation only.

## M2.5c-P0 - Phase Collection Pipeline Finalization

- Date: `2026-08-13`
- Local HEAD before this cleanup: `179e778`.
- Scope: production data-collection and phase-dataset pipeline cleanup only.
  No BFM training, model, loss, optimizer, replay, observation, action, or
  evaluation logic was changed.
- Formal raw output root: `dataset/sim_collected/phase/raw/`.
- `rollout_split.py` is now a canonical raw collector. It retains parallel
  command-grid collection, resume/replacement, official per-rollout HUSKY
  physics randomization, fixed phase recording, shared fall detection,
  robot-board state capture, progress, collection plan, and collection
  summary.
- Removed from the collector: legacy offline rollout parsing, key-event and
  key-map segmentation, manual command parsing, synchronized-video
  segmentation, pose-only BFM preview generation, per-phase intermediate
  files, and the old single-rollout test CLI. Historical artifacts remain in
  git history and prior experiment records.
- `convert_husky_to_bfm.py` now reads complete raw rollouts directly, validates
  frame alignment and metadata, uses recorded `phase_id` for contiguous
  segmentation, applies fall/reset hard boundaries, enforces the seq8 minimum
  of `9` frames, maps HUSKY 23DoF to BFM 29DoF by joint name, preserves paired
  board/action/phase fields and provenance, aggregates one final pkl, validates
  the official MotionLib and expert loader, and generates post-hoc full-scene
  QC videos.
- Final artifact paths:
  `dataset/sim_collected/phase/motion_library/skate_expert_phase.pkl`,
  `dataset/sim_collected/phase/motion_library/manifest.json`, and
  `dataset/sim_collected/phase/qc/`.
- Bounded validation used only `/tmp`: two parallel 50-frame raw rollouts
  (`h=+0.2` and `h=-0.2`), raw-to-phase aggregation, full MotionLib loading,
  official expert-loader seq8 sampling, provenance checks, and six QC video
  outputs. Existing regression tests passed: `8 passed`; `ruff` and
  `py_compile` passed.
- The full formal collection was not run in this cleanup stage.
- Formal command for the next stage:
  `python train/scripts/data_collection/rollout_split.py --parallel-config`

## M2.5c-P - Formal Phase Dataset Collection and Build

- Date: `2026-08-13`.
- Formal HUSKY raw collection completed at
  `dataset/sim_collected/phase/raw/`.
- The collector completed all `150` baseline rollouts and `8` replacement
  rollouts with no worker-level failures. The final raw dataset contains
  `452,885` frames at `50 Hz`, totaling `150.962` minutes.
- Terminal reasons were `125` full-length rollouts and `33` confirmed falls.
  Confirmed falls are valid episode boundaries, not collection failures.
- Formal phase segmentation and HUSKY-to-BFM conversion produced `6,038`
  phase motions with `452,291` accepted expert frames, totaling `148.751`
  minutes.
- Accepted phase motion counts: `1,522` push, `1,516` push2steer, `685`
  steer_left, `109` steer_forward, `717` steer_right, and `1,489`
  steer2push.
- Raw rollout rejection count: `0`. Motion conversion rejection count: `0`.
- Official `MotionLibRobot` validation, official expert-loader Seq8 sampling,
  robot/board/action/phase/source-frame provenance, and post-hoc full-scene QC
  all passed.
- QC used seed `20260813` and rendered `10` uniformly sampled motions for each
  of the six phases, for `60` samples total.
- Final artifacts:
  `dataset/sim_collected/phase/motion_library/skate_expert_phase.pkl`,
  `dataset/sim_collected/phase/motion_library/manifest.json`,
  `dataset/sim_collected/phase/qc/qc_manifest.json`, and six phase QC videos.
- Published dataset:
  `https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/phase`.
- Training was not launched. Continuous-dataset collection was not launched.

## M2.5c-C0 - Continuous Dataset Pipeline Preparation

- Date: `2026-08-14`.
- Renamed the formal Phase converter from `convert_husky_to_bfm.py` to
  `convert_phase.py` without changing its content or behavior.
- Created `convert_continuous.py` from the Phase converter with a fixed
  `500`-frame, `10.0 s`, non-overlapping Continuous clip contract.
- Both converters use the same M2.5c-P canonical raw collection; no raw data is
  copied, regenerated, or modified.
- Continuous clips preserve frame-aligned phase annotations and may cross
  normal phase transitions, but never cross confirmed fall or reset
  boundaries.
- Temporary conversion, MotionLibRobot, Seq8, provenance, and full-scene QC
  smoke checks passed. All temporary artifacts were removed.
- The formal Continuous dataset build was not launched.

## M2.5c-C Dataset Publication

- Date: `2026-08-14`.
- Published the validated Continuous MotionLib, manifest, QC manifest, and
  stitched QC video to
  `https://huggingface.co/datasets/Yak9Ce3teeh/skate-sim-dataset/tree/main/continuous`.
- The GitHub README and `train/README.md` now document the Continuous download
  path. No raw data, model, or training logic was changed.

## M2.5c Dataset Layout Reorganization

- Date: `2026-08-14`.
- Reorganized the Hugging Face dataset so `phase/` and `continuous/` are
  peer-level dataset stages, with the shared `raw/` collection at the same
  level.
- Moved the existing Phase MotionLib and QC files under `phase/`, and kept the
  shared raw collection at root `raw/`, using Git/LFS pointer renames.
- Added a root dataset README describing both stages and updated all current
  raw, Phase, and Continuous restore commands.
- No raw bytes, model files, dataset contents, or training logic were changed.

## M2.6-0a - Formal Trainer Parameterization

- Date: `2026-08-14`.
- The active trainer is no longer locked to the historical 20k pilot. Its
  formal default is 100,000 transitions with 1,024 warmup transitions, the
  first update at 1,500, an update interval of 500, and 50 native updates per
  block.
- The default 100k schedule contains 198 update blocks and 9,900 native
  `agent.update()` calls. Default checkpoints are 20k, 50k, and 100k, and
  replay capacity scales with the transition budget without overwrite.
- Added the formal `phase` / `continuous` expert dataset selector and removed
  the old tiny Skate expert as the active default. The selected MotionLib,
  adjacent manifest, dataset statistics, and SHA256 provenance are recorded.
- Official BFM0 strict initialization, fresh optimizer state, native
  `agent.update()`, Base/Skate 50/50 sampling, batch size 1024, sequence length
  8, model/loss/optimizer settings, and one online HUSKY environment are
  unchanged.
- Real formal expert preflight passed for both datasets. Phase loaded 6,038
  motions and Continuous loaded 890 motions; both produced finite current and
  next expert observations and finite expert latent tensors of shape
  `[1024, 256]`.
- A temporary Phase smoke passed with 2,000 transitions, update steps at 1,500
  and 2,000, two native updates, dynamic replay validation, finite model and
  optimizer states, Actor changes, and final checkpoint reload. All temporary
  artifacts were deleted.
- Parallel online environments remain deliberately deferred to M2.6-0b.
  Formal Phase 100k and Continuous 100k training were not launched.
