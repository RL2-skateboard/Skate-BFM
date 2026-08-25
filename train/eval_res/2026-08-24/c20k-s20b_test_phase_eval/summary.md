# Formal Phase Evaluation

| Behavior | Cases | Full completion | Joint MAE | Root Ori | Board XY | Coupling XY | Feet on board |
|---|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 1.000 | 0.47106 | 12.918 | 0.28064 | 0.18599 | 0.792 |
| steer | 20 | 1.000 | 0.43966 | 19.557 | 0.42288 | 0.43262 | 0.648 |
| push2steer | 20 | 1.000 | 0.43472 | 58.247 | 2.2672 | 0.97047 | 0.257 |
| steer2push | 20 | 1.000 | 0.45173 | 31.314 | 2.1515 | 1.7696 | 0.276 |

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 8 | 0.43591 | 1.000 |
| forward | 1 | 0.52398 | 1.000 |
| right | 11 | 0.43472 | 1.000 |

## Transition Sections

### push2steer

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.46439 |
| transition | 20 | 0.48474 |
| post | 20 | 0.41393 |

### steer2push

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.42883 |
| transition | 20 | 0.44179 |
| post | 20 | 0.46201 |

## Protocol

- Full test: `False`
- Tracking parity: `PASS`
- Training: `False`
