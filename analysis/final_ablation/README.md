# Accepted final ablation

The accepted P0–P5 ladder is a paper-facing selection of frozen source-locked experiments, not six newly executed runs during consolidation.

| Stage | Added module | PrivateTest flip-TTA (%) | Source experiment |
|---|---|---:|---|
| P0 | Pixel Graph baseline | 62.4965 | Cumulative A0 |
| P1 | + Spatial Motif Composer | 63.9175 | Cumulative A1 |
| P2 | + Geometry-aware Motif Graph | 65.3107 | Cumulative A3 |
| P3 | + Multi-scale Composition | 65.7286 | Cumulative A4 |
| P4 | + Learnable Motif Attention Readout | 70.3260 | Table VI DENSE_MOTIF |
| P5 | FULL + Dynamic Top-K | 70.6604 | Canonical FULL seed 42 |

P4 uses learnable attention pooling with dense motif relations. P5 uses the canonical `[8,16,16,16,24]` Top-K schedule. These comparisons are recorded experimental evidence, not proof that the entire multi-component causal contribution is isolated by a correlation analysis.

The old `FIXED_POOL` name meant **fixed spatial composition with a learned motif readout** (70.02% TTA). Cumulative A5 meant **learned multi-scale composition with fixed uniform motif readout** (64.59% TTA, duplicated mean pooling). They implement different interventions and are not interchangeable controls. The accepted ladder uses P4 DENSE_MOTIF and P5 FULL; it does not reinterpret Cumulative A5 as old FIXED_POOL.

The original metrics, selected epochs, checkpoint hashes, ladder definitions and forward/config forensic summaries are retained in this directory. [IMPORT_MANIFEST.json](../IMPORT_MANIFEST.json) identifies their exact source commit/path/hashes. Full cumulative runtime/source-copy trees remain in the immutable research closure archive.
