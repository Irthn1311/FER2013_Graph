# MPG-FER Cumulative Ablation Matrix

Strict Nested Subset Ladder: **A0 ⊂ A1 ⊂ A2 ⊂ A3 ⊂ A4 ⊂ A5 ⊂ A6**

| Config | Display Name | Pixel GNN | Composer | Scales | Motif GNN | Geom Bias | Relation Mode | Top-K Schedule | Motif Pooling | Direct Pixel Fusion | Classifier |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A0 | Pixel Graph baseline | Yes | None | — | No | No | none | — | none | Yes | full_concat_512 |
| A1 | + Spatial Motif Composer | Yes | single_scale_12 | [12] | No | No | none | — | fixed | Yes | full_concat_512 |
| A2 | + Motif Graph | Yes | single_scale_12 | [12] | Yes | No | dense | [48, 48, 48, 48, 48] | fixed | Yes | full_concat_512 |
| A3 | + Geometry Bias | Yes | single_scale_12 | [12] | Yes | Yes | dense | [48, 48, 48, 48, 48] | fixed | Yes | full_concat_512 |
| A4 | + Multi-scale Composition | Yes | multi_scale_8_12_16 | [8, 12, 16] | Yes | Yes | dense | [48, 48, 48, 48, 48] | fixed | Yes | full_concat_512 |
| A5 | + Dynamic Top-K Relations | Yes | multi_scale_8_12_16 | [8, 12, 16] | Yes | Yes | dynamic_topk | [8, 16, 16, 16, 24] | fixed | Yes | full_concat_512 |
| A6 | Full Model | Yes | multi_scale_8_12_16 | [8, 12, 16] | Yes | Yes | dynamic_topk | [8, 16, 16, 16, 24] | learnable_attention | Yes | full_concat_512 |

## Pairwise Transition Validation

| Transition | From | To | Intended Addition | Verified Fields | Status |
|---|---|---|---|---|:---:|
| A0->A1 | Pixel Graph baseline | + Spatial Motif Composer | Add Spatial Motif Composer (12x12) + Fixed pooling + enable motif losses | `composer, composer_scales, motif_pooling, applicable_losses` | VALID |
| A1->A2 | + Spatial Motif Composer | + Motif Graph | Enable Motif GNN (dense relations) | `motif_gnn, relation_mode, motif_topk_schedule` | VALID |
| A2->A3 | + Motif Graph | + Geometry Bias | Enable Geometry Bias in Motif GNN | `geometry_bias` | VALID |
| A3->A4 | + Geometry Bias | + Multi-scale Composition | Enable Multi-scale Composition (8, 12, 16) | `composer, composer_scales` | VALID |
| A4->A5 | + Multi-scale Composition | + Dynamic Top-K Relations | Enable Dynamic Top-K Relations [8, 16, 16, 16, 24] | `relation_mode, motif_topk_schedule` | VALID |
| A5->A6 | + Dynamic Top-K Relations | Full Model | Replace Fixed Pooling with Learnable Attention Pooling (Full Model) | `motif_pooling` | VALID |

## Loss Term Preservation & Schedule

All configurations share the identical scientific recipe:
- Seed: 42
- Optimizer: AdamW (lr=3e-4, weight_decay=5e-4)
- Batch size: 16 (gradient_accumulation=2)
- Scheduler: Linear warmup 5 epochs, cosine decay through epoch 85
- Early stopping: monitoring begins at epoch 85, patience 15
- EMA: 0.999
- Loss coefficients: final_ce=1.0, pixel_aux=0.05, consistency=0.15, supcon=0.05 across all configurations.
- Motif auxiliary losses (motif_aux=0.2, lambda_mi=0.025, lambda_div=0.01) are disabled ONLY for A0 where no motif representation exists mathematically, and enabled for all subsequent configurations A1-A6.
