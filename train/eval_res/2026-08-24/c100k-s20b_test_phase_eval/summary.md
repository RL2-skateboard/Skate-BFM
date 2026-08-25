# Formal Phase Evaluation

| Behavior | Cases | Full completion | Joint MAE | Root Ori | Board XY | Coupling XY | Feet on board |
|---|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.250 | 0.52562 | 27.794 | 0.11258 | 0.15834 | 0.762 |
| steer | 20 | 0.700 | 0.40244 | 24.417 | 0.12494 | 0.10277 | 0.922 |
| push2steer | 20 | 0.000 | 0.53226 | 24.703 | 0.20081 | 0.21329 | 0.897 |
| steer2push | 20 | 0.000 | 0.46144 | 29.97 | 0.14977 | 0.17119 | 0.856 |

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 8 | 0.45534 | 0.500 |
| forward | 1 | 0.4701 | 0.000 |
| right | 11 | 0.35782 | 0.909 |

## Transition Sections

### push2steer

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.50193 |
| transition | 19 | 0.655 |
| post | 2 | 0.51853 |

### steer2push

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.40358 |
| transition | 19 | 0.48091 |
| post | 18 | 0.54723 |

## Protocol

- Full test: `False`
- Tracking parity: `PASS`
- Training: `False`
