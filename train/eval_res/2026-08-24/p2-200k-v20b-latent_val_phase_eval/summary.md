# Formal Val Phase Evaluation

| Behavior | Cases | Full completion | Board planar velocity | Board speed | Board direction | Board heading | Final board XY | Feet on board | Coupling XY | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.650 | 0.35239 | 0.22085 | 15.747 | 6.4933 | 0.33265 | 0.892 | 0.06342 | 0.5084 | 
| steer | 20 | 1.000 | 0.17561 | 0.058217 | 6.9769 | 6.1905 | 0.23864 | 0.991 | 0.037905 | 0.3608 | 
| push2steer | 20 | 0.500 | 0.70314 | 0.48558 | 22.38 | 24.996 | 2.3959 | 0.537 | 0.66402 | 0.4493 | 
| steer2push | 20 | 0.250 | 0.64152 | 0.41381 | 26.355 | 20.601 | 2.1168 | 0.808 | 0.29067 | 0.45064 | 

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 10 | 0.3652 | 1.000 |
| forward | 2 | 0.36867 | 1.000 |
| right | 8 | 0.35334 | 1.000 |

## Transition Sections

### push2steer

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.19147 | 0.17603 | 2.2008 | 2.152 | 0.04132 | 1 | 0.95408 | 0.47449 |
| transition | 20 | 0.67443 | 0.45817 | 14.927 | 15.165 | 0.11792 | 0.99833 | 0.87167 | 0.51065 |
| post | 20 | 1.0433 | 0.66104 | 40.983 | 43.919 | 0.81889 | 0.86616 | 0.26782 | 0.42973 |

### steer2push

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.11335 | 0.088521 | 2.1364 | 2.3911 | 0.032451 | 1 | 0.99898 | 0.36052 |
| transition | 20 | 0.38919 | 0.30571 | 10.806 | 9.5377 | 0.11794 | 0.99 | 0.97333 | 0.41368 |
| post | 20 | 0.73262 | 0.48538 | 35.519 | 23.992 | 0.32992 | 0.98375 | 0.77628 | 0.50652 |

## Protocol

- Split: `val`
- Full test: `False`
- Tracking parity: `PASS`
- Binary task success: `NOT_YET_CALIBRATED`
- Training: `False`
