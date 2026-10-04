# Cumulative Ablation Study on FER2013 (MPG-FER)

### English (Paper-Facing Table)

| Configuration | Acc. (%) | Gain |
|---|---:|---:|
| Pixel Graph baseline | 62.50 | — |
| + Spatial Motif Composer | 63.92 | +1.42 |
| + Motif Graph | 63.69 | -0.22 |
| + Geometry Bias | 65.31 | +1.62 |
| + Multi-scale Composition | 65.73 | +0.42 |
| + Dynamic Top-K Relations | 64.59 | -1.14 |
| Full Model | 70.66 | +6.07 |

### Vietnamese (Bảng Tiếng Việt)

| Cấu hình | Acc. (%) | Gain |
|---|---:|---:|
| Pixel Graph baseline | 62.50 | — |
| + Spatial Motif Composer | 63.92 | +1.42 |
| + Motif Graph | 63.69 | -0.22 |
| + Geometry Bias | 65.31 | +1.62 |
| + Multi-scale Composition | 65.73 | +0.42 |
| + Dynamic Top-K Relations | 64.59 | -1.14 |
| Full Model | 70.66 | +6.07 |

### Full Evaluation Metrics (Raw & TTA)

| Config | Display Name | Raw Acc. (%) | Raw F1 (%) | TTA Acc. (%) | TTA F1 (%) | Gain | Selected Epoch | Checkpoint SHA256 (first 10) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A0 | Pixel Graph baseline | 59.04 | 55.28 | 62.50 | 60.03 | — | 66 | `baf81845a1` |
| A1 | + Spatial Motif Composer | 61.30 | 58.92 | 63.92 | 61.41 | +1.42 | 34 | `103958895c` |
| A2 | + Motif Graph | 60.91 | 58.34 | 63.69 | 62.75 | -0.22 | 80 | `b44011f654` |
| A3 | + Geometry Bias | 61.66 | 59.06 | 65.31 | 63.53 | +1.62 | 75 | `83d1b2269e` |
| A4 | + Multi-scale Composition | 61.94 | 60.29 | 65.73 | 64.57 | +0.42 | 79 | `7515832701` |
| A5 | + Dynamic Top-K Relations | 61.88 | 58.57 | 64.59 | 61.82 | -1.14 | 48 | `270b14ec75` |
| A6 | Full Model | 68.77 | 67.35 | 70.66 | 69.82 | +6.07 | 57 | `23dbe9b145` |
