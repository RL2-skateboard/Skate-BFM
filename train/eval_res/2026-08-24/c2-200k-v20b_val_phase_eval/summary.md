# Formal Val Phase Evaluation

| Behavior | Cases | Full completion | Board planar velocity | Board speed | Board direction | Board heading | Final board XY | Feet on board | Coupling XY | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.900 | 0.2592 | 0.21115 | 7.5508 | 5.1831 | 0.389 | 0.920 | 0.071193 | 0.50864 | 
| steer | 20 | 0.900 | 0.20645 | 0.071757 | 6.2851 | 6.5923 | 0.23932 | 0.979 | 0.056045 | 0.36979 | 
| push2steer | 20 | 0.050 | 0.68046 | 0.51259 | 20.525 | 13.58 | 1.8209 | 0.800 | 0.25043 | 0.49729 | 
| steer2push | 20 | 0.150 | 0.48577 | 0.39003 | 17.488 | 10.177 | 1.3527 | 0.858 | 0.29943 | 0.44762 | 

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 10 | 0.3648 | 1.000 |
| forward | 2 | 0.39597 | 0.500 |
| right | 8 | 0.36948 | 0.875 |

## Transition Sections

### push2steer

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.15676 | 0.13729 | 2.6456 | 2.2094 | 0.039007 | 0.99898 | 0.98878 | 0.50722 |
| transition | 20 | 0.95054 | 0.76334 | 18.213 | 12.837 | 0.13025 | 0.99167 | 0.82064 | 0.55183 |
| post | 19 | 1.0259 | 0.74308 | 33.662 | 24.527 | 0.42125 | 0.99676 | 0.58627 | 0.46942 |

### steer2push

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.10241 | 0.048304 | 3.0705 | 3.3447 | 0.032137 | 1 | 1 | 0.36131 |
| transition | 20 | 0.2614 | 0.13309 | 9.1915 | 9.3618 | 0.11289 | 1 | 1 | 0.39684 |
| post | 20 | 0.61631 | 0.51651 | 22.29 | 13.836 | 0.32481 | 0.96954 | 0.83435 | 0.50831 |

## Protocol

- Split: `val`
- Full test: `False`
- Tracking parity: `PASS`
- Binary task success: `NOT_YET_CALIBRATED`
- Training: `False`
