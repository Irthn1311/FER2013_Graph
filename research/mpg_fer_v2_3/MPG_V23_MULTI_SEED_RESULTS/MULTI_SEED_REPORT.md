# MPG-FER v2.3 Six-Seed Replication Results

Canonical evaluation: FP32, autocast disabled, TF32 disabled, frozen EMA checkpoint.

## Headline

- Private Raw Accuracy: 0.688028 +/- 0.006906
- Private Raw Macro-F1: 0.674688 +/- 0.008207
- Private TTA Accuracy: 0.704607 +/- 0.002634
- Private TTA Macro-F1: 0.694537 +/- 0.005630

## Individual seeds

| Seed | Epoch | Private raw acc | Private raw macro-F1 | Private TTA acc | Private TTA macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 45 | 0.695458 | 0.686000 | 0.707161 | 0.700749 |
| 1 | 42 | 0.692672 | 0.680797 | 0.702981 | 0.692280 |
| 42 | 57 | 0.687657 | 0.673482 | 0.706604 | 0.698158 |
| 43 | 49 | 0.686264 | 0.674364 | 0.706046 | 0.698730 |
| 123 | 44 | 0.675676 | 0.662070 | 0.700195 | 0.691383 |
| 3047 | 42 | 0.690443 | 0.671417 | 0.704653 | 0.685924 |

All six registered seeds are retained. PrivateTest did not select checkpoints, and no run was repeated because of its metric value.
