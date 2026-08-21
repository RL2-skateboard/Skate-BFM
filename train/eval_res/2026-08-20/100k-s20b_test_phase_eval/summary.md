# Formal Phase Evaluation

| Behavior | Cases | Full completion | Joint MAE | Root Ori | Board XY | Coupling XY | Feet on board |
|---|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.250 | 0.53627 | 21.691 | 0.12911 | 0.18092 | 0.848 |
| steer | 20 | 0.750 | 0.43073 | 28.889 | 0.16033 | 0.15139 | 0.922 |
| push2steer | 20 | 0.050 | 0.52623 | 24.729 | 0.26434 | 0.147 | 0.904 |
| steer2push | 20 | 0.000 | 0.46078 | 29.756 | 0.19463 | 0.19418 | 0.856 |

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 8 | 0.46437 | 0.625 |
| forward | 1 | 0.48853 | 0.000 |
| right | 11 | 0.40101 | 0.909 |

## Transition Sections

### push2steer

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.50102 |
| transition | 19 | 0.64401 |
| post | 4 | 0.55589 |

### steer2push

| Section | Cases | Joint MAE |
|---|---:|---:|
| pre | 20 | 0.38267 |
| transition | 20 | 0.49798 |
| post | 18 | 0.54646 |

## Protocol

- Full test: `False`
- Tracking parity: `PASS`
- Training: `False`
