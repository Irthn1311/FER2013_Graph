# MPG-FER Motif Readout Diagnostic Summary

## Decision: **READOUT_DIAGNOSTIC_NO_GO** (CASE C (STOP readout-expansion direction; gain depends on end-to-end co-adaptation))

### Validation Performance (val.csv, 3589 samples)
| Probe | Architecture | Parameters | Val Acc. (%) | Val Macro-F1 (%) | Selected Epoch | Delta vs R0 Refit (pp) |
|---|---|---:|---:|---:|---:|---:|
| R0 Canonical | Mean + Max + AttnPool -> Proj | 225,224 | 67.90 (TTA: 69.21) | 66.19 | 57 | — |
| R0 Refit | Mean + Max + AttnPool -> Proj | 225,224 | 67.76 | 66.71 | 2 | 0.00 |
| R1 MultiSlot | S=7 Class-Agnostic Slots | 204,807 | 67.34 | 66.11 | 11 | -0.42 |
| R2 ClassCond | C=7 Class-Conditioned Queries | 76,423 | 67.15 | 66.02 | 5 | -0.61 |

### Attention Collapse Diagnostics
- **R1 Multi-Slot:** Mean Entropy = `3.311` | Effective Nodes = `30.71` / 49 | Query Cosine Sim = `0.157`
- **R2 Class-Conditioned:** Mean Entropy = `3.604` | Effective Nodes = `36.99` / 49 | Query Cosine Sim = `0.048`
