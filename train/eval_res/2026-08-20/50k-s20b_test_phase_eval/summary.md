# Formal Phase Evaluation

| Behavior | Cases | Full completion | Joint MAE | Root Ori | Board XY | Coupling XY | Feet on board |
|---|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.150 | 0.59934 | 29.672 | 0.14635 | 0.15866 | 0.673 |
| steer | 20 | 0.200 | 0.45611 | 22.757 | 0.24271 | 0.36034 | 0.694 |
| push2steer | 20 | 0.000 | 0.70746 | 32.715 | 0.043013 | 0.069234 | 0.395 |
| steer2push | 20 | 0.000 | 0.50316 | 25.445 | 0.11175 | 0.18384 | 0.728 |

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 8 | 0.45646 | 0.125 |
| forward | 1 | 0.48697 | 0.000 |
| right | 11 | 0.45306 | 0.273 |

## Transition Sections

### push2steer

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.70746 |
| transition | 0 | - |
| post | 0 | - |

### steer2push

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.44928 |
| transition | 14 | 0.88446 |
| post | 2 | 0.8154 |

## Protocol

- Full test: `False`
- Tracking parity: `PASS`
- Training: `False`
