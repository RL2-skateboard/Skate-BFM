# Training Log

## M2.5b Final Baseline

- Date: 2026-08-12
- Completed: consolidated the training branch around the final original
  BFM-Zero Skate baseline.
- Retained: strict official BFM0 initialization, Base/Skate MotionLib loading,
  64/64 sequence sampling, HUSKY online replay, eight physical auxiliary
  rewards, confirmed-fall termination, native FB-CPR-Aux updates, 20k
  scheduling, checkpoint reload validation, and frozen fixed evaluation.
- Removed: superseded H1, smoke, collection, conversion, target-bank builder,
  audit, generic Isaac training, and B/F-only adaptation entrypoints.
- Behavior: no BFM-Zero algorithm, loss, optimizer, reward scaling,
  termination, expert ratio, or M2.5b schedule was intentionally changed.
