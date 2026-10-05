# Phase 0: Source Audit for Contextual Scale Recomposition (CSR)

## 1. Checkpoint & Provenance Verification
- **Checkpoint Path:** `D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\research\mpg_fer_v2_3\official_runs\segment_02\mpg_fer_v2_3_run\best_val_acc.pt`
- **Checkpoint SHA256:** `23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e` (Strict parity confirmed)
- **Seed:** 42

## 2. Extraction Point Specifications

| Tensor | Mathematical Symbol | Shape | Extraction Source Location | Semantics |
|---|:---:|---|---|---|
| Pixel GNN Output | $H_P$ | `[B, 2304, 96]` | `model.pixel_gnn` loop output | 2304 pixel nodes after 4 edge-aware layers |
| Scale Candidates | $z_{(m,s)}$ | `[B, 49, 3, 192]` | `candidate_stack` in `SpatialMotifComposer` | Scale-specific candidate occurrences at scales 8, 12, 16 |
| Pre-Graph Scale Weights | $\alpha_{(m,s)}$ | `[B, 49, 3]` | `alpha` in `SpatialMotifComposer` | Softmax scale gating over 3 scales per occurrence |
| Initial Fused Motif | $h_m^{(0)}$ | `[B, 49, 192]` | `(alpha * candidate_stack).sum(dim=2)` | Early occurrence state prior to Motif Graph |
| Final Contextual Motif | $h_m^{(L)}$ | `[B, 49, 192]` | `model.motif_gnn` loop output | Post-relational occurrence state after 5 Motif GNN blocks |

## 3. Core Hypothesis Mapping
- The initial fusion $h_m^{(0)} = \sum_s \alpha_{(m,s)} z_{(m,s)}$ compresses local scales before cross-occurrence relational context is available.
- Contextual Scale Recomposition (CSR) uses $h_m^{(L)}$ as queries to re-evaluate the original scale candidates $z_{(m,s)}$, testing whether relational context can recover discarded scale evidence.

## 4. Hard Data Rule Enforcement
- Extraction and evaluation will strictly use **Train (28,709)** and **Validation (3,589)** splits.
- Absolutely no Private/Test data will be accessed during this diagnostic.
