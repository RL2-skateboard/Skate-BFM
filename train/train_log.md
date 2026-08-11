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
