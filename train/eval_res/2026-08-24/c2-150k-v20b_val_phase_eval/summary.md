# Formal Val Phase Evaluation

| Behavior | Cases | Full completion | Board planar velocity | Board speed | Board direction | Board heading | Final board XY | Feet on board | Coupling XY | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.700 | 0.27276 | 0.24563 | 6.249 | 2.9905 | 0.28008 | 0.882 | 0.088718 | 0.51978 | 
| steer | 20 | 0.850 | 0.2285 | 0.07761 | 7.4951 | 7.8549 | 0.23068 | 0.963 | 0.059912 | 0.37317 | 
| push2steer | 20 | 0.000 | 0.55371 | 0.4583 | 13.553 | 10.181 | 1.3476 | 0.825 | 0.17743 | 0.53245 | 
| steer2push | 20 | 0.250 | 0.62199 | 0.42428 | 23.574 | 16.221 | 2.0453 | 0.705 | 0.36748 | 0.46794 | 

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 10 | 0.37915 | 0.900 |
| forward | 2 | 0.40225 | 0.500 |
| right | 8 | 0.35841 | 0.875 |

## Transition Sections

### push2steer

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.18032 | 0.15853 | 2.3861 | 2.0253 | 0.044565 | 0.99184 | 0.99592 | 0.51358 |
| transition | 20 | 0.75165 | 0.64235 | 10.166 | 10.126 | 0.1443 | 1 | 0.87333 | 0.57237 |
| post | 19 | 0.90091 | 0.74623 | 26.73 | 16.604 | 0.26291 | 0.99382 | 0.55274 | 0.55213 |

### steer2push

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.14684 | 0.068657 | 4.6757 | 5.0141 | 0.039176 | 1 | 1 | 0.35745 |
| transition | 20 | 0.37789 | 0.23906 | 11.478 | 11.667 | 0.11057 | 1 | 0.97333 | 0.41568 |
| post | 20 | 0.694 | 0.45029 | 33.191 | 21.143 | 0.36245 | 0.97167 | 0.65162 | 0.52849 |

## Protocol

- Split: `val`
- Full test: `False`
- Tracking parity: `PASS`
- Binary task success: `NOT_YET_CALIBRATED`
- Training: `False`
