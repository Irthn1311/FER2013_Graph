# Contextual Local Pixel Reinspection Diagnostic Summary

## Decision: **PIXEL_REINSPECTION_NO_GO** (CASE PIX-C (NO-GO. Late evidence-recovery family is exhausted))

### Validation Performance (val.csv, 3589 samples)
| Probe | Mechanism | Parameters | Val Acc. (%) | Val Macro-F1 (%) | Selected Epoch | Delta vs P0 (pp) |
|---|---|---:|---:|---:|---:|---:|
| P0 Baseline | Frozen h_m^(L) + Canonical Readout | 225,224 | 67.76 | 66.71 | 2 | 0.00 |
| P1 Reinjection | Prior-weighted pixel sum d_m | 281,289 | 67.48 | 66.05 | 6 | -0.28 |
| P2 Contextual | Query h_m^(L) + spatial prior bias | 308,937 | 67.48 | 66.16 | 2 | -0.28 |

### Pixel Reinspection Dynamics
- **P1 Mean Gate Activation:** `0.850` | Norm Ratio: `0.277`
- **P2 Mean Gate Activation:** `0.700` | Norm Ratio: `0.430`
- **P2 Attention Entropy:** `2.391` (Effective Pixels: `10.9` / 256)
- **Mean KL(a || p):** `2.8575` | Mean |a - p|: `0.0063`
- **Argmax Pixel Change Rate:** `95.15%`
