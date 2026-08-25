# Formal Val Phase Evaluation

| Behavior | Cases | Full completion | Board planar velocity | Board speed | Board direction | Board heading | Final board XY | Feet on board | Coupling XY | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.000 | 0.62283 | 0.388 | 39.082 | 7.9096 | 0.28851 | 0.689 | 0.16117 | 0.61278 | 
| steer | 20 | 0.050 | 0.4741 | 0.37665 | 6.9841 | 5.6509 | 0.16416 | 0.614 | 0.14664 | 0.52398 | 
| push2steer | 20 | 0.000 | 0.28347 | 0.26603 | 4.6677 | 1.3368 | 0.11265 | 0.424 | 0.09148 | 0.61409 | 
| steer2push | 20 | 0.000 | 0.37065 | 0.32502 | 6.2483 | 4.2356 | 0.1257 | 0.691 | 0.116 | 0.49101 | 

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 10 | 0.52034 | 0.100 |
| forward | 2 | 0.47749 | 0.000 |
| right | 8 | 0.54014 | 0.000 |

## Transition Sections

### push2steer

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.28347 | 0.26603 | 4.6677 | 1.3368 | 0.09148 | 0.99756 | 0.42361 | 0.61409 |
| transition | 0 | - | - | - | - | - | - | - | - |
| post | 0 | - | - | - | - | - | - | - | - |

### steer2push

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.35236 | 0.31017 | 5.8729 | 3.9364 | 0.11155 | 1 | 0.7162 | 0.48241 |
| transition | 8 | 0.63548 | 0.53945 | 11.269 | 9.962 | 0.17779 | 1 | 0.1625 | 0.60837 |
| post | 1 | 1.8623 | 1.758 | 17.093 | 5.8001 | 0.30325 | 1 | 1 | 0.65562 |

## Protocol

- Split: `val`
- Full test: `False`
- Tracking parity: `PASS`
- Binary task success: `NOT_YET_CALIBRATED`
- Training: `False`
