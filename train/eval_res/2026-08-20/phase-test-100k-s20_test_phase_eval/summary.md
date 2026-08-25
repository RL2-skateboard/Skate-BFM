# Formal Phase Evaluation

| Behavior | Cases | Full completion | Joint MAE | Root Ori | Board XY | Coupling XY | Feet on board |
|---|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.000 | 0.55396 | 24.256 | 0.11768 | 0.17724 | 0.838 |
| steer | 20 | 0.800 | 0.39892 | 18.267 | 0.14182 | 0.096047 | 0.978 |
| push2steer | 20 | 0.000 | 0.53075 | 26.407 | 0.13882 | 0.17846 | 0.839 |
| steer2push | 20 | 0.000 | 0.45562 | 30.991 | 0.20742 | 0.2374 | 0.808 |

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 0 | - | - |
| forward | 20 | 0.39892 | 0.800 |
| right | 0 | - | - |

## Transition Sections

### push2steer

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.50546 |
| transition | 17 | 0.58865 |
| post | 4 | 0.63473 |

### steer2push

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.36118 |
| transition | 20 | 0.47644 |
| post | 20 | 0.57305 |

## Protocol

- Full test: `False`
- Tracking parity: `PASS`
- Training: `False`
