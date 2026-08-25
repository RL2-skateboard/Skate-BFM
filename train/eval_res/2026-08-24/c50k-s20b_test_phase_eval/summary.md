# Formal Phase Evaluation

| Behavior | Cases | Full completion | Joint MAE | Root Ori | Board XY | Coupling XY | Feet on board |
|---|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.100 | 0.61241 | 57.072 | 0.12434 | 0.17144 | 0.718 |
| steer | 20 | 0.000 | 0.55944 | 63.405 | 0.072864 | 0.16156 | 0.468 |
| push2steer | 20 | 0.000 | 0.62468 | 53.038 | 0.048843 | 0.094883 | 0.536 |
| steer2push | 20 | 0.000 | 0.49224 | 47.279 | 0.054981 | 0.10125 | 0.631 |

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 8 | 0.58432 | 0.000 |
| forward | 1 | 0.65591 | 0.000 |
| right | 11 | 0.53257 | 0.000 |

## Transition Sections

### push2steer

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.62468 |
| transition | 0 | - |
| post | 0 | - |

### steer2push

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.48599 |
| transition | 7 | 0.59308 |
| post | 0 | - |

## Protocol

- Full test: `False`
- Tracking parity: `PASS`
- Training: `False`
