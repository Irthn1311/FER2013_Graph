# Forward-Graph Differential Test: Cumulative A5 vs Table VI FIXED_POOL

## Classification: `CASE_2_SAME_NAME_DIFFERENT_SEMANTICS`

### Verdict & Mathematical Justification
The model named 'FIXED_POOL' in Table VI and the configuration 'A5' in the cumulative ladder share the phrase 'fixed pooling' in their high-level descriptions but implement completely opposite architectural interventions. In Table VI, 'Fixed spatial pooling' referred to replacing the learned Spatial Motif Composer of pixels with fixed 12x12 grid pooling, while keeping learnable attention pooling in the motif readout. In Cumulative A5, the Spatial Motif Composer was retained in full multi-scale learned form, while the motif readout pooling was replaced with uniform mean pooling (duplicating m_mean). Hence, the two experiments test different hypotheses and naturally yield divergent metrics (70.02% vs 64.59%).

### Intermediate Tensor Shapes & Invariants
| Tensor Stage | Cumulative A5 Shape | Old FIXED_POOL Shape | Shape Match |
|---|---|---|:---:|
| `pixel_projected` | `[2, 2304, 96]` | `[2, 2304, 96]` | YES |
| `pixel_nodes` | `[2, 2304, 96]` | `[2, 2304, 96]` | YES |
| `pixel_readout` | `[2, 128]` | `[2, 128]` | YES |
| `motif_nodes_initial` | `[2, 49, 192]` | `[2, 49, 192]` | YES |
| `motif_nodes_final` | `[2, 49, 192]` | `[2, 49, 192]` | YES |
| `motif_readout` | `[2, 384]` | `[2, 384]` | YES |
| `fusion_representation` | `[2, 512]` | `[2, 512]` | YES |
| `logits` | `[2, 7]` | `[2, 7]` | YES |

### Architectural Mechanism Contrast

| Mechanism | Cumulative A5 (+ Dynamic Top-K) | Old FIXED_POOL ('Fixed spatial pooling') |
|---|---|---|
| **Spatial Composition** | self.motif_composer (SpatialMotifComposer with learned prototypes, scale_saliency, scale_gate, 8/12/16) | self.fixed_pool_composer (FixedSpatialPoolComposer with fixed 12x12 mean pooling, NO prototypes, NO gate) |
| **Motif Readout Pooling** | Fixed uniform mean pooling: m_pool = h_motif.mean(dim=1). self.motif_attn_pool is bypassed and unused. | Learnable attention pooling: m_attention = (softmax(self.motif_attn_pool(h_motif)) * h_motif).sum(dim=1). |
| **Motif Readout Concat** | cat(m_mean, m_max, m_mean) -> m_mean is duplicated twice in the 576D vector! | cat(m_mean, m_max, m_attention) -> distinct mean, max, and learned attention pooling components. |

### Numerical Forward Comparison
- **Pixel Readout Max Abs Diff:** `0.000000` (shared pixel path identical)
- **Initial Motif Nodes Max Abs Diff:** `2.9858` (diverges immediately due to different composition)
- **Final Motif Nodes Max Abs Diff:** `3.9773`
- **Motif Readout Max Abs Diff:** `2.5044`
- **Synthetic Input Logits Max Abs Diff:** `0.6865`
- **Real FER2013 Samples Logits Max Abs Diff:** `0.5748`
