# Training Results

## M2.5b Original BFM-Zero Skate Baseline

- Date: 2026-08-12
- Status: completed.
- Initialization: fresh official BFM0 checkpoint,
  SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- Online replay: 20,000 HUSKY transitions, 29D stored BFM action, 23D executed
  HUSKY action, and eight finite auxiliary reward fields.
- Expert source: Base LAFAN plus Skate MotionLib, sampled as 64 Base and 64
  Skate complete sequences of length 8 per native update.
- Update schedule: first update at 1,500 transitions, then every 500
  transitions, 50 native updates per block, 38 blocks, 1,900 updates total.
- Dynamics: nominal HUSKY training dynamics; no training domain randomization.
- Checkpoints: 10k and 20k checkpoints saved and reloaded successfully,
  including model, buffers, and optimizer state.
- Fixed evaluation: `INCONCLUSIVE`. All 60 frozen evaluation episodes
  terminated under the confirmed physical-fall contract before the 128-step
  horizon. No skating task-success claim is made.
