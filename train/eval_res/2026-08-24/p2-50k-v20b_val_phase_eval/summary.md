# Formal Val Phase Evaluation

| Behavior | Cases | Full completion | Board planar velocity | Board speed | Board direction | Board heading | Final board XY | Feet on board | Coupling XY | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.250 | 0.44608 | 0.41785 | 11.76 | 2.6601 | 0.40484 | 0.697 | 0.17439 | 0.56941 | 
| steer | 20 | 0.250 | 0.60019 | 0.45403 | 17.743 | 9.9253 | 0.68331 | 0.638 | 0.34978 | 0.44493 | 
| push2steer | 20 | 0.000 | 0.21368 | 0.20164 | 2.2861 | 0.77953 | 0.094098 | 0.344 | 0.072606 | 0.72776 | 
| steer2push | 20 | 0.000 | 0.40032 | 0.33694 | 11.522 | 6.1068 | 0.35147 | 0.738 | 0.18923 | 0.4935 | 

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 10 | 0.45069 | 0.200 |
| forward | 2 | 0.48908 | 0.000 |
| right | 8 | 0.42669 | 0.375 |

## Transition Sections

### push2steer

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.21368 | 0.20164 | 2.2861 | 0.77953 | 0.072606 | 1 | 0.34437 | 0.72776 |
| transition | 0 | - | - | - | - | - | - | - | - |
| post | 0 | - | - | - | - | - | - | - | - |

### steer2push

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.33935 | 0.28856 | 7.1619 | 4.5814 | 0.12736 | 1 | 0.875 | 0.42012 |
| transition | 18 | 0.69648 | 0.56404 | 23.078 | 10.524 | 0.39701 | 1 | 0.12685 | 0.90436 |
| post | 2 | 0.58371 | 0.47546 | 64.781 | 19.45 | 0.57757 | 0.92105 | 0.26758 | 0.5452 |

## Protocol

- Split: `val`
- Full test: `False`
- Tracking parity: `PASS`
- Binary task success: `NOT_YET_CALIBRATED`
- Training: `False`
