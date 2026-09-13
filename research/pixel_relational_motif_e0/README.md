# Pixel Relational Motif — E0

Implementation of GitHub Issue #74. This package implements **E0 only** for the locked Pixel–Graph–Motif research direction:

- E0.1: existence/stability/non-degeneracy of recurring local pixel-relation motifs;
- E0.2: necessity of ordered local pixel relations via matched destroyed-relation controls;
- E0.3: value of consistent spatial motif composition via matched geometry-shuffle controls.

It is deliberately isolated from `research/pure_gnn_v31/`. It does not implement the downstream M0 GNN and it has no API that accepts or reads the FER2013 PrivateTest split.

The implementation uses valid 5x5 windows (44x44 centers), a deterministic 24D center-relative descriptor, frozen Train-fit PCA (24->11, no whitening), an explicit log-local-contrast coordinate, a custom diagonal GMM with variance floors, bootstrap stability with motif-specific support-matched nulls, and deterministic occurrence/geometry controls.
