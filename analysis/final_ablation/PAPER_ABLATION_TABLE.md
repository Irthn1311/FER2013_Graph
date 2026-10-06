# Paper-Ready Cumulative Ablation Table

### English (Table for Paper)

| Configuration | Acc. (%) | Gain |
|---|---:|---:|
| Pixel Graph baseline | 62.50 | — |
| + Spatial Motif Composer | 63.92 | +1.42 |
| + Geometry-aware Motif Graph | 65.31 | +1.39 |
| + Multi-scale Composition | 65.73 | +0.42 |
| + Learnable Motif Readout | 70.33 | +4.60 |
| Full Model | 70.66 | +0.33 |

### Vietnamese (Bảng Tiếng Việt)

| Cấu hình | Acc. (%) | Gain |
|---|---:|---:|
| Pixel Graph baseline | 62.50 | — |
| + Spatial Motif Composer | 63.92 | +1.42 |
| + Geometry-aware Motif Graph | 65.31 | +1.39 |
| + Multi-scale Composition | 65.73 | +0.42 |
| + Learnable Motif Readout | 70.33 | +4.60 |
| Full Model | 70.66 | +0.33 |

### Alternate Explicit Display Name (with Dynamic Top-K Named)

| Configuration | Acc. (%) | Gain |
|---|---:|---:|
| Pixel Graph baseline | 62.50 | — |
| + Spatial Motif Composer | 63.92 | +1.42 |
| + Geometry-aware Motif Graph | 65.31 | +1.39 |
| + Multi-scale Composition | 65.73 | +0.42 |
| + Learnable Motif Readout | 70.33 | +4.60 |
| + Dynamic Top-K Relations (Full Model) | 70.66 | +0.33 |

### Full Metric Decomposition (Raw & TTA)

| Stage | Configuration | Raw Acc. (%) | Raw F1 (%) | TTA Acc. (%) | TTA F1 (%) | Gain | Epoch | Checkpoint SHA256 (first 10) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| P0 | Pixel Graph baseline | 59.04 | 55.28 | 62.50 | 60.03 | — | 66 | `baf81845a1` |
| P1 | + Spatial Motif Composer | 61.30 | 58.92 | 63.92 | 61.41 | +1.42 | 34 | `103958895c` |
| P2 | + Geometry-aware Motif Graph | 61.66 | 59.06 | 65.31 | 63.53 | +1.39 | 75 | `83d1b2269e` |
| P3 | + Multi-scale Composition | 61.94 | 60.29 | 65.73 | 64.57 | +0.42 | 79 | `7515832701` |
| P4 | + Learnable Motif Readout | 68.38 | 67.30 | 70.33 | 69.05 | +4.60 | 59 | `74a4437148` |
| P5 | Full Model | 68.77 | 67.35 | 70.66 | 69.82 | +0.33 | 57 | `23dbe9b145` |
