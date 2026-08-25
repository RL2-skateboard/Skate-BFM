# Formal Val Phase Evaluation

| Behavior | Cases | Full completion | Board planar velocity | Board speed | Board direction | Board heading | Final board XY | Feet on board | Coupling XY | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.300 | 0.34574 | 0.27062 | 13.827 | 5.9859 | 0.27764 | 0.773 | 0.1518 | 0.5353 | 
| steer | 20 | 0.500 | 0.29922 | 0.17468 | 8.5525 | 6.9057 | 0.28299 | 0.909 | 0.11618 | 0.42345 | 
| push2steer | 20 | 0.000 | 0.63595 | 0.45799 | 30.381 | 7.8109 | 0.73494 | 0.930 | 0.19427 | 0.52147 | 
| steer2push | 20 | 0.000 | 0.38836 | 0.23244 | 17.918 | 8.6605 | 0.54436 | 0.927 | 0.15274 | 0.43461 | 

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 10 | 0.46242 | 0.400 |
| forward | 2 | 0.39759 | 0.500 |
| right | 8 | 0.3812 | 0.625 |

## Transition Sections

### push2steer

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.38158 | 0.30266 | 14.648 | 6.4306 | 0.098327 | 0.99082 | 0.98465 | 0.49757 |
| transition | 17 | 1.9401 | 1.1089 | 111.08 | 12.369 | 0.61933 | 0.99303 | 0.54461 | 0.62938 |
| post | 2 | 1.2401 | 0.81138 | 62.3 | 16.682 | 1.0336 | 1 | 0.375 | 0.60011 |

### steer2push

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.1232 | 0.095148 | 2.4076 | 2.5778 | 0.03921 | 1 | 1 | 0.37952 |
| transition | 20 | 0.36387 | 0.27219 | 10.161 | 10.601 | 0.14757 | 1 | 0.97857 | 0.47375 |
| post | 18 | 0.94559 | 0.50072 | 61.789 | 18.464 | 0.37246 | 0.98893 | 0.74395 | 0.53166 |

## Protocol

- Split: `val`
- Full test: `False`
- Tracking parity: `PASS`
- Binary task success: `NOT_YET_CALIBRATED`
- Training: `False`
