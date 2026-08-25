# Formal Val Phase Evaluation

| Behavior | Cases | Full completion | Board planar velocity | Board speed | Board direction | Board heading | Final board XY | Feet on board | Coupling XY | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 1.000 | 0.37165 | 0.32932 | 9.6726 | 5.0639 | 0.61783 | 0.793 | 0.16268 | 0.47211 | 
| steer | 20 | 1.000 | 0.70389 | 0.51172 | 26.848 | 19.615 | 1.2164 | 0.674 | 0.42322 | 0.43619 | 
| push2steer | 20 | 0.950 | 0.97207 | 0.87913 | 26.837 | 10.967 | 4.6351 | 0.283 | 0.92854 | 0.43084 | 
| steer2push | 20 | 1.000 | 1.0447 | 0.9454 | 39.513 | 19.034 | 4.7402 | 0.271 | 1.5083 | 0.45034 | 

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 10 | 0.43899 | 1.000 |
| forward | 2 | 0.43361 | 1.000 |
| right | 8 | 0.43334 | 1.000 |

## Transition Sections

### push2steer

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.3026 | 0.29639 | 2.1188 | 1.6415 | 0.13125 | 0.99898 | 0.86429 | 0.46115 |
| transition | 20 | 1.2246 | 1.1031 | 27.833 | 4.5285 | 0.45032 | 0.98 | 0.50667 | 0.47971 |
| post | 20 | 1.1732 | 1.0368 | 33.782 | 14.249 | 1.1031 | 0.91704 | 0.068163 | 0.40992 |

### steer2push

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.41081 | 0.36076 | 11.387 | 6.1217 | 0.14435 | 0.99286 | 0.9398 | 0.42181 |
| transition | 20 | 0.84862 | 0.62045 | 47.625 | 17.487 | 0.54077 | 0.94 | 0.60333 | 0.45582 |
| post | 20 | 0.98234 | 0.86852 | 65.074 | 21.606 | 1.4809 | 0.64292 | 0.097917 | 0.46244 |

## Protocol

- Split: `val`
- Full test: `False`
- Tracking parity: `PASS`
- Binary task success: `NOT_YET_CALIBRATED`
- Training: `False`
