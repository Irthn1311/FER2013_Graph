# Phase 0: Source Audit for Contextual Local Pixel Reinspection

## 1. Checkpoint & Provenance Verification
- **Checkpoint Path:** `D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\research\mpg_fer_v2_3\official_runs\segment_02\mpg_fer_v2_3_run\best_val_acc.pt`
- **Checkpoint SHA256:** `23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e` (Exact match to frozen canonical lock)
- **Seed:** 42

## 2. Extraction Point Specifications

| Tensor | Mathematical Symbol | Shape | Extraction Source Location | Semantics |
|---|:---:|---|---|---|
| Pixel GNN Output | $H_P$ | `[B, 2304, 96]` | `model.pixel_gnn` loop output | 2304 pixel nodes after 4 edge-aware layers |
| SMC Spatial Weights | $w_{(m,s,i)}$ | `64, 144, 256` per scale | `weights[scale]` in `SpatialMotifComposer` | Spatial softmax weights per scale support |
| Pre-Graph Scale Weights | $\alpha_{(m,s)}$ | `[B, 49, 3]` | `alpha` in `SpatialMotifComposer` | Scale gate weights over {8, 12, 16} |
| Final Contextual Motif | $h_m^{(L)}$ | `[B, 49, 192]` | `model.motif_gnn` loop output | Occurrence state after 5 Motif GNN blocks |
| Canonical Pixel Prior | $p_{(m,i)}$ | `[B, 49, 256]` | Normalized $\sum_s \alpha_{(m,s)} w_{(m,s,i)}$ | Spatial evidence prior over 256 support pixels |

## 3. Pixel Prior Normalization Invariant
- For every occurrence $m \in \{0, \dots, 48\}$, the support pixels $S_m = \bigcup_s S_{(m,s)}$ are fully indexed within `supports[16][m]` (256 pixels).
- The prior is normalized such that $\sum_{i=1}^{256} p_{(m,i)} = 1.0$ bitwise.

## 4. Hard Data Rule Enforcement
- Extraction and evaluation strictly use **Train (28,709)** and **Validation (3,589)** splits.
- Absolutely no Private/Test data was opened or accessed.
