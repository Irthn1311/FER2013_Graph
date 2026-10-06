# Upstream Representation & Optimization Audit Summary

## Verdict: **UPSTREAM_AUDIT_STOP_INCREMENTAL**
**Classification:** UPSTREAM_D: No clean layerwise degradation, no material gradient conflict, and no localized divergence

**Actionable Hypothesis:** None. Incremental architectural interventions are exhausted under the current design framework. Stop incremental ad-hoc modifications.

### 1. Layerwise Probe Accuracy Curve (Validation, %)
| Layer | FULL Val Acc. (%) | FULL Val F1 (%) | NPF Val Acc. (%) | NPF Val F1 (%) |
|---|---:|---:|---:|---:|
| **P** | 56.92 | 53.73 | 56.73 | 51.63 |
| **M0** | 62.36 | 59.30 | 63.25 | 59.96 |
| **M1** | 65.12 | 62.96 | 64.89 | 61.32 |
| **M2** | 66.26 | 63.66 | 66.23 | 63.42 |
| **M3** | 67.65 | 65.82 | 67.43 | 64.62 |
| **M4** | 67.60 | 65.76 | 67.21 | 64.66 |
| **M5** | 67.60 | 66.49 | 67.29 | 65.26 |

### 2. FULL vs NPF Representation Similarity (CKA & Cosine)
| Layer | Linear CKA | Centered Cosine |
|---|---:|---:|
| **P** | 0.9627 | 0.8167 |
| **M0** | 0.9477 | 0.7093 |
| **M1** | 0.7792 | 0.4910 |
| **M2** | 0.7181 | 0.4594 |
| **M3** | 0.8671 | 0.6445 |
| **M4** | 0.8762 | 0.6313 |
| **M5** | 0.8733 | 0.6051 |

### 3. Gradient Alignment across Parameter Groups (cos(G_cls, G_motif))
| Parameter Group | Mean Cosine | Median Cosine | Fraction Negative | Fraction < -0.1 | Norm Ratio (G_motif / G_cls) |
|---|---:|---:|---:|---:|---:|
| **1_pixel_backbone** | 0.9618 | 0.9786 | 0.00 | 0.00 | 0.9546 |
| **2_spatial_motif_composer** | 0.9686 | 0.9831 | 0.00 | 0.00 | 0.9592 |
| **3_motif_gnn** | 0.9651 | 0.9823 | 0.00 | 0.00 | 0.9674 |
| **4_motif_readout** | 0.7308 | 0.7796 | 0.00 | 0.00 | 1.1677 |
| **5_final_classifier** | 0.0000 | 0.0000 | 0.00 | 0.00 | 0.0000 |
