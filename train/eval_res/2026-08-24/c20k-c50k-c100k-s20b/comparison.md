# Continuous 20k / 50k / 100k Comparison

- 80 fixed Test cases per checkpoint; Phase case-spec exact replay.
- Derived artifact only: original training/evaluation files are unchanged.

## Latent Projection

- Shared PCA explained variance: 0.10276, 0.07576, 0.05206.
- `z_t` is checkpoint-specific tracking latent from the evaluator Raw-window bridge, retained only through executed `T_exec`.

## Paired Comparison

### Continuous Internal

| Pair | Behavior | Completion delta | Joint MAE delta | Board XY delta | Coupling delta |
|---|---|---:|---:|---:|---:|
| 20k_vs_50k | push | -0.6295 | 0.1414 | -0.1563 | -0.0146 |
| 20k_vs_50k | steer | -0.7230 | 0.1198 | -0.3500 | -0.2711 |
| 20k_vs_50k | push2steer | -0.8956 | 0.1900 | -2.2184 | -0.8756 |
| 20k_vs_50k | steer2push | -0.8428 | 0.0405 | -2.0965 | -1.6683 |
| 50k_vs_100k | push | 0.2835 | -0.0868 | -0.0118 | -0.0131 |
| 50k_vs_100k | steer | 0.6105 | -0.1570 | 0.0521 | -0.0588 |
| 50k_vs_100k | push2steer | 0.1534 | -0.0924 | 0.1520 | 0.1184 |
| 50k_vs_100k | steer2push | 0.2146 | -0.0308 | 0.0948 | 0.0699 |
| 20k_vs_100k | push | -0.3460 | 0.0546 | -0.1681 | -0.0276 |
| 20k_vs_100k | steer | -0.1125 | -0.0372 | -0.2979 | -0.3298 |
| 20k_vs_100k | push2steer | -0.7422 | 0.0975 | -2.0664 | -0.7572 |
| 20k_vs_100k | steer2push | -0.6282 | 0.0097 | -2.0017 | -1.5984 |

### Phase vs Continuous

| Pair | Behavior | Completion delta | Joint MAE delta | Board XY delta | Coupling delta |
|---|---|---:|---:|---:|---:|
| 20k | push | 0.0140 | -0.0142 | -0.0092 | -0.0003 |
| 20k | steer | 0.0345 | 0.0091 | -0.3266 | -0.3504 |
| 20k | push2steer | 0.1066 | -0.0210 | 0.9142 | 0.2161 |
| 20k | steer2push | 0.0000 | 0.0010 | -0.0561 | 0.1178 |
| 50k | push | -0.2090 | 0.0131 | -0.0220 | 0.0128 |
| 50k | steer | -0.4530 | 0.1033 | -0.1699 | -0.1988 |
| 50k | push2steer | -0.0168 | -0.0828 | 0.0058 | 0.0256 |
| 50k | steer2push | -0.0652 | -0.0109 | -0.0568 | -0.0826 |
| 100k | push | -0.1065 | -0.0106 | -0.0165 | -0.0226 |
| 100k | steer | -0.0065 | -0.0283 | -0.0354 | -0.0486 |
| 100k | push2steer | -0.0432 | 0.0060 | -0.0635 | 0.0663 |
| 100k | steer2push | -0.0618 | 0.0007 | -0.0449 | -0.0230 |
