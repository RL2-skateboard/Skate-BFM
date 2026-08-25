# Formal Val Phase Evaluation

| Behavior | Cases | Full completion | Board planar velocity | Board speed | Board direction | Board heading | Final board XY | Feet on board | Coupling XY | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.400 | 0.37129 | 0.31005 | 8.8945 | 6.198 | 0.27424 | 0.808 | 0.19641 | 0.52827 | 
| steer | 20 | 0.500 | 0.49887 | 0.30709 | 12.723 | 10.949 | 0.47968 | 0.864 | 0.16858 | 0.47238 | 
| push2steer | 20 | 0.000 | 0.67008 | 0.46258 | 25.362 | 3.1559 | 0.75795 | 0.854 | 0.2143 | 0.52912 | 
| steer2push | 20 | 0.000 | 0.4958 | 0.33639 | 21.711 | 15.1 | 0.79727 | 0.840 | 0.25552 | 0.46349 | 

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 10 | 0.54747 | 0.400 |
| forward | 2 | 0.4241 | 0.500 |
| right | 8 | 0.39058 | 0.625 |

## Transition Sections

### push2steer

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.36053 | 0.32715 | 7.5246 | 1.9205 | 0.09423 | 0.99388 | 0.96926 | 0.49916 |
| transition | 18 | 2.0044 | 0.9904 | 108.31 | 4.376 | 0.3172 | 0.9706 | 0.65838 | 0.66185 |
| post | 3 | 0.64837 | 0.51448 | 13.64 | 14.543 | 0.98176 | 0.99753 | 0.17676 | 0.52637 |

### steer2push

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.16581 | 0.10554 | 4.0007 | 4.3354 | 0.066974 | 1 | 1 | 0.38117 |
| transition | 20 | 0.50473 | 0.33709 | 14.572 | 14.841 | 0.16681 | 0.99667 | 0.93 | 0.49801 |
| post | 20 | 0.99493 | 0.67829 | 47.396 | 25.84 | 0.51192 | 0.98101 | 0.58918 | 0.56625 |

## Protocol

- Split: `val`
- Full test: `False`
- Tracking parity: `PASS`
- Binary task success: `NOT_YET_CALIBRATED`
- Training: `False`
