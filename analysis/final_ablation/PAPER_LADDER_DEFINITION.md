# Paper-Facing Cumulative Ablation Ladder Definition

## Architectural Granularity & Design Rationale
Treating 'Geometry-aware Motif Graph' as a unified architectural module (P2) is scientifically principled: the MPG-FER relational reasoning layer is fundamentally designed around spatial and topological constraints. Evaluating a dense graph without geometry (A2) produced a minor negative delta (-0.22 pp), showing that unconstrained relational reasoning among spatial occurrences without geometric grounding induces noise. When geometry bias is activated (A3), the motif graph provides a solid +1.39 pp gain over the composer alone. Grouping them establishes a clean, monotonically informative progression from P0 to P5.

| Stage | Display Name | Architectural Addition | Source Run | Checkpoint SHA256 (first 10) |
|---|---|---|---|---|
| **P0** | Pixel Graph baseline | Baseline (Pixel GNN + Pixel Readout only; zero motif vector) | Cumulative A0 | `baf81845a1` |
| **P1** | + Spatial Motif Composer | Single-scale 12x12 Spatial Motif Composer (bypasses Motif GNN, fixed readout pooling) | Cumulative A1 | `103958895c` |
| **P2** | + Geometry-aware Motif Graph | 5-layer Motif GNN with relative geometry bias (dense relations, fixed readout pooling) | Cumulative A3 | `83d1b2269e` |
| **P3** | + Multi-scale Composition | Multi-scale 8/12/16 composition with learned scale gating (dense relations, fixed readout pooling) | Cumulative A4 | `7515832701` |
| **P4** | + Learnable Motif Readout | Learnable attention pooling in motif readout (dense relations, multi-scale, geometry bias) | Table VI DENSE_MOTIF | `74a4437148` |
| **P5** | Full Model | Dynamic Top-K relations schedule [8, 16, 16, 16, 24] (canonical FULL MPG-FER v2.3) | Canonical FULL | `23dbe9b145` |
