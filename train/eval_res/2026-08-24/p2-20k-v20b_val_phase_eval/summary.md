# Formal Val Phase Evaluation

| Behavior | Cases | Full completion | Board planar velocity | Board speed | Board direction | Board heading | Final board XY | Feet on board | Coupling XY | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 1.000 | 0.39404 | 0.3068 | 11.281 | 9.5766 | 0.55768 | 0.664 | 0.18577 | 0.47711 | 
| steer | 20 | 0.950 | 0.9946 | 0.81615 | 31.709 | 12.632 | 1.7762 | 0.434 | 0.82913 | 0.4318 | 
| push2steer | 20 | 0.800 | 0.7526 | 0.64331 | 20.568 | 26.81 | 3.1827 | 0.395 | 0.81857 | 0.46009 | 
| steer2push | 20 | 1.000 | 1.0871 | 0.9804 | 60.756 | 17.709 | 5.2902 | 0.151 | 1.76 | 0.45047 | 

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 10 | 0.43483 | 1.000 |
| forward | 2 | 0.41959 | 1.000 |
| right | 8 | 0.43107 | 0.875 |

## Transition Sections

### push2steer

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.18071 | 0.16999 | 2.1659 | 1.7356 | 0.0729 | 1 | 0.87551 | 0.48161 |
| transition | 20 | 0.79128 | 0.69301 | 13.37 | 10.084 | 0.2197 | 0.99833 | 0.56167 | 0.50259 |
| post | 20 | 1.023 | 0.83244 | 31.27 | 37.598 | 0.97178 | 0.92546 | 0.18067 | 0.4638 |

### steer2push

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.7558 | 0.66131 | 25.759 | 7.1179 | 0.36823 | 0.97857 | 0.61224 | 0.44834 |
| transition | 20 | 0.85853 | 0.6619 | 57.421 | 16.439 | 0.91923 | 0.95 | 0.21333 | 0.48178 |
| post | 20 | 0.9129 | 0.77264 | 79.066 | 20.302 | 1.7797 | 0.74458 | 0.037083 | 0.45163 |

## Protocol

- Split: `val`
- Full test: `False`
- Tracking parity: `PASS`
- Binary task success: `NOT_YET_CALIBRATED`
- Training: `False`
