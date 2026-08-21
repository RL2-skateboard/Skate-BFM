# Formal Phase Evaluation

| Behavior | Cases | Full completion | Joint MAE | Root Ori | Board XY | Coupling XY | Feet on board |
|---|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.950 | 0.48521 | 18.921 | 0.28981 | 0.18632 | 0.660 |
| steer | 20 | 0.900 | 0.43053 | 23.794 | 0.7495 | 0.78305 | 0.362 |
| push2steer | 20 | 0.800 | 0.45572 | 41.348 | 1.353 | 0.75437 | 0.406 |
| steer2push | 20 | 1.000 | 0.45069 | 40.701 | 2.2076 | 1.6518 | 0.179 |

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 8 | 0.4179 | 0.875 |
| forward | 1 | 0.50643 | 1.000 |
| right | 11 | 0.43281 | 0.909 |

## Transition Sections

### push2steer

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.48625 |
| transition | 20 | 0.50327 |
| post | 19 | 0.42746 |

### steer2push

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.45653 |
| transition | 20 | 0.49582 |
| post | 20 | 0.44722 |

## Protocol

- Full test: `False`
- Tracking parity: `PASS`
- Training: `False`
