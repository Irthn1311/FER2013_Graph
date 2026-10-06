# Contextual Scale Recomposition (CSR) Diagnostic Summary

## Decision: **CSR_DIAGNOSTIC_NO_GO** (CASE CSR-C (NO-GO for scale-level recomposition; move below SMC to pixel reinspection))

### Validation Performance (val.csv, 3589 samples)
| Probe | Mechanism | Parameters | Val Acc. (%) | Val Macro-F1 (%) | Selected Epoch | Delta vs C0 (pp) |
|---|---|---:|---:|---:|---:|---:|
| C0 Baseline | Frozen h_m^(L) + Canonical Readout | 225,224 | 67.76 | 66.71 | 2 | 0.00 |
| C1 Early Skip | Gated skip from h_m^(0) | 262,857 | 67.51 | 66.16 | 11 | -0.25 |
| C2 CSR | Query h_m^(L) attending to z_(m,s) | 373,449 | 67.73 | 65.71 | 5 | -0.03 |

### Recomposition Dynamics (Alpha vs Beta)
- **Argmax Scale Change Rate:** `67.73%` of occurrences altered their primary scale choice
- **Mean KL(Beta || Alpha):** `0.4722`
- **Mean |Beta - Alpha|:** `0.2649`
- **Beta Entropy:** `0.648` (Effective scales: `1.91` / 3)
- **Mean CSR Gate Activation:** `0.728`
- **Mean C1 Gate Activation:** `0.908`
