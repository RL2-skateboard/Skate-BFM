# Formal Val Phase Evaluation

| Behavior | Cases | Full completion | Board planar velocity | Board speed | Board direction | Board heading | Final board XY | Feet on board | Coupling XY | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.700 | 0.36919 | 0.26351 | 15.087 | 4.6685 | 0.32956 | 0.761 | 0.11228 | 0.53061 | 
| steer | 20 | 0.850 | 0.27477 | 0.16041 | 6.2982 | 6.2751 | 0.2297 | 0.992 | 0.048117 | 0.41831 | 
| push2steer | 20 | 0.000 | 0.54939 | 0.44957 | 15.999 | 6.9667 | 1.0053 | 0.803 | 0.22326 | 0.52352 | 
| steer2push | 20 | 0.050 | 0.44903 | 0.31873 | 16.35 | 14.112 | 1.1279 | 0.742 | 0.24125 | 0.46598 | 

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 10 | 0.47484 | 0.700 |
| forward | 2 | 0.37821 | 1.000 |
| right | 8 | 0.35768 | 1.000 |

## Transition Sections

### push2steer

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.23211 | 0.21898 | 2.4821 | 1.8729 | 0.075621 | 0.99592 | 0.93776 | 0.49782 |
| transition | 20 | 0.8933 | 0.66777 | 29.448 | 9.0841 | 0.28188 | 0.99833 | 0.76306 | 0.58116 |
| post | 13 | 0.98848 | 0.78564 | 40.145 | 14.087 | 0.42745 | 0.98805 | 0.58663 | 0.57148 |

### steer2push

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.11803 | 0.071653 | 3.1388 | 3.4392 | 0.035363 | 1 | 1 | 0.36673 |
| transition | 20 | 0.36463 | 0.23944 | 10.466 | 10.523 | 0.097929 | 1 | 0.99667 | 0.42748 |
| post | 20 | 0.56474 | 0.39615 | 23.158 | 19.623 | 0.30702 | 0.99949 | 0.61734 | 0.51108 |

## Protocol

- Split: `val`
- Full test: `False`
- Tracking parity: `PASS`
- Binary task success: `NOT_YET_CALIBRATED`
- Training: `False`
