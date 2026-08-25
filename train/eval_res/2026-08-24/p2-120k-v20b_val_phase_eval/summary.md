# Formal Val Phase Evaluation

| Behavior | Cases | Full completion | Board planar velocity | Board speed | Board direction | Board heading | Final board XY | Feet on board | Coupling XY | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push | 20 | 0.450 | 0.45953 | 0.39511 | 16.299 | 5.5091 | 0.51101 | 0.722 | 0.21078 | 0.50872 | 
| steer | 20 | 0.600 | 0.45644 | 0.21853 | 12.681 | 12.573 | 0.55552 | 0.790 | 0.12345 | 0.45475 | 
| push2steer | 20 | 0.000 | 0.58598 | 0.49384 | 13.509 | 8.0837 | 1.0163 | 0.805 | 0.22385 | 0.50142 | 
| steer2push | 20 | 0.000 | 0.53898 | 0.31514 | 20.819 | 20.721 | 1.1773 | 0.754 | 0.24092 | 0.44076 | 

## Steer

| Direction | Cases | Joint MAE | Full completion |
|---|---:|---:|---:|
| left | 10 | 0.53361 | 0.500 |
| forward | 2 | 0.41195 | 0.500 |
| right | 8 | 0.36687 | 0.750 |

## Transition Sections

### push2steer

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.26815 | 0.2351 | 6.3304 | 2.974 | 0.101 | 0.99898 | 0.92736 | 0.48176 |
| transition | 19 | 1.0579 | 0.88542 | 25.511 | 8.231 | 0.34054 | 1 | 0.6193 | 0.51587 |
| post | 16 | 1.0234 | 0.76507 | 27.392 | 18.071 | 0.50514 | 0.99842 | 0.54647 | 0.57685 |

### steer2push

| Section | Cases | Board planar velocity | Board speed | Board direction | Board heading | Coupling XY | Direction valid fraction | Feet on board | Joint MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | 20 | 0.15762 | 0.11449 | 3.9451 | 3.6589 | 0.061163 | 0.99884 | 0.98721 | 0.34626 |
| transition | 19 | 0.49919 | 0.29679 | 16.988 | 17.151 | 0.10948 | 1 | 0.9614 | 0.43204 |
| post | 19 | 0.80069 | 0.43443 | 34.689 | 33.991 | 0.35506 | 0.98683 | 0.56363 | 0.51951 |

## Protocol

- Split: `val`
- Full test: `False`
- Tracking parity: `PASS`
- Binary task success: `NOT_YET_CALIBRATED`
- Training: `False`
