# Formal Val Phase Evaluation

| Behavior | Cases | Full completion | Board planar velocity | Board speed | Board direction | Board heading | Final board XY | Feet on board | Coupling XY | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.450 | 0.33112 | 0.26633 | 12.504 | 5.7848 | 0.28813 | 0.655 | 0.17769 | 0.49832 | 
| steer | 20 | 0.550 | 0.3666 | 0.21306 | 8.2945 | 8.5456 | 0.32712 | 0.954 | 0.11657 | 0.42378 | 
| push2steer | 20 | 0.000 | 0.70217 | 0.59919 | 25.788 | 8.5029 | 1.1881 | 0.782 | 0.22035 | 0.51935 | 
| steer2push | 20 | 0.000 | 0.4491 | 0.29752 | 18.44 | 16.298 | 0.87541 | 0.824 | 0.27202 | 0.43507 | 

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 10 | 0.47692 | 0.400 |
| forward | 2 | 0.40109 | 0.500 |
| right | 8 | 0.36303 | 0.750 |

## Transition Sections

### push2steer

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.26026 | 0.23472 | 5.5387 | 4.6123 | 0.072477 | 0.98776 | 0.98571 | 0.46844 |
| transition | 20 | 1.1664 | 1.0186 | 38.41 | 12.667 | 0.29086 | 0.96917 | 0.62494 | 0.57119 |
| post | 14 | 1.456 | 1.1465 | 85.845 | 15.936 | 0.65258 | 0.92857 | 0.22133 | 0.62964 |

### steer2push

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.12201 | 0.069144 | 3.3093 | 3.5829 | 0.059084 | 1 | 0.99388 | 0.34626 |
| transition | 20 | 0.4141 | 0.27011 | 12.209 | 12.73 | 0.16503 | 1 | 0.93333 | 0.40855 |
| post | 20 | 0.65571 | 0.43882 | 29.141 | 25.487 | 0.4269 | 0.99833 | 0.62972 | 0.52359 |

## Protocol

- Split: `val`
- Full test: `False`
- Tracking parity: `PASS`
- Binary task success: `NOT_YET_CALIBRATED`
- Training: `False`
