# Phase 0: Source Audit of Canonical FULL MPG-FER Motif Readout

## 1. Checkpoint & Provenance Verification
- **Checkpoint Path:** `D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\research\mpg_fer_v2_3\official_runs\segment_02\mpg_fer_v2_3_run\best_val_acc.pt`
- **Checkpoint SHA256:** `23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e` (Exact match to frozen canonical lock)
- **Selected Epoch:** 57
- **Weights Type:** EMA
- **Strict Load Status:** PASS (0 missing keys, 0 unexpected keys)

## 2. Motif Node Tensor Identification
- **Extraction Point:** Output of final Motif Transformer Block (Layer 5) immediately prior to motif readout pooling.
- **Tensor Name:** `h_motif`
- **Tensor Shape:** `[Batch_Size, 49, 192]` (49 occurrence nodes, 192 feature channels).

## 3. Canonical Motif Readout Architecture
The canonical motif readout consists of three parallel pooling branches concatenated into a projection:
1. **Mean Pooling Branch:** `m_mean = h_motif.mean(dim=1)` (`[B, 192]`)
2. **Max Pooling Branch:** `m_max = h_motif.max(dim=1).values` (`[B, 192]`)
3. **Attention Pooling Branch:** `m_attention = (softmax(motif_attn_pool(h_motif)) * h_motif).sum(dim=1)` (`[B, 192]`)
   - Query Parameter: `self.motif_attn_pool = nn.Linear(192, 1, bias=True)` (193 parameters)
4. **Concatenation:** `cat([m_mean, m_max, m_attention], dim=-1)` (`[B, 576]`)
5. **Readout Projection:** `nn.Sequential(Linear(576, 384), LayerNorm(384), GELU())` (222,720 parameters)
   - Resulting Motif Representation: `r_M` of dimension `384`.
6. **Motif Auxiliary Classifier Head:** `nn.Linear(384, 7)` (2,695 parameters)
- **Total Motif Readout Parameters:** `225,608`

## 4. Canonical Validation Parity (3,589 samples, val.csv)

| Head | Raw Acc. (%) | Raw Macro-F1 (%) | TTA Acc. (%) | TTA Macro-F1 (%) |
|---|---:|---:|---:|---:|
| **Motif Auxiliary Head (Motif Only)** | 67.90 | 66.19 | 69.21 | 67.42 |

### Key Observation:
- On Validation, the **Motif Auxiliary Head achieves 69.21% TTA Accuracy**, confirming canonical parity.
- Validation parity verified. Proceeding to probe diagnostics.
