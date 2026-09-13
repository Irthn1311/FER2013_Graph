# Pixel Relational Motif — E0

Implementation of GitHub Issue #74. This package implements **E0 only** for the locked Pixel–Graph–Motif research direction:

- E0.1: existence/stability/non-degeneracy of recurring local pixel-relation motifs;
- E0.2: necessity of ordered local pixel relations via matched destroyed-relation controls;
- E0.3: value of consistent spatial motif composition via matched geometry-shuffle controls.

It is deliberately isolated from `research/pure_gnn_v31/`. It does not implement the downstream M0 GNN and it has no API that accepts or reads the FER2013 PrivateTest split.

The implementation uses valid 5x5 windows (44x44 centers), a deterministic 24D center-relative descriptor, frozen Train-fit PCA (24->11, no whitening), an explicit log-local-contrast coordinate, a custom diagonal GMM with variance floors, bootstrap stability with motif-specific support-matched nulls, and deterministic occurrence/geometry controls.

## E0.1 official-Train runner

The real-data E0.1 runner is intentionally Train-only. It validates exactly 28,709 rows, validates labels only for file integrity, then discards them before the unsupervised image-level `Train_fit/Train_heldout/Train_anchor` workflow. PublicTest and PrivateTest are not read by E0.1.

Run from this package directory after installing dependencies:

```bash
python -m pixel_relational_motif_e0.e01_runner \
  --train-csv /path/to/train.csv \
  --output-dir /path/to/e01_artifacts
```

The runner writes `e01_summary.json` plus `e01_dictionary.npz`. The summary records source-data SHA256, partition sizes, exact log-sigma decile edges, fixed-pool counts, PCA provenance, per-K image-level heldout likelihood statistics, 1-SE candidates, BH-stability evidence, non-degeneracy evidence, and the final canonical stable-component mapping.

Occurrence calibration is a second Train-only phase using that frozen dictionary:

```bash
python -m pixel_relational_motif_e0.e01_occurrence_runner \
  --train-csv /path/to/train.csv \
  --dictionary-npz /path/to/e01_artifacts/e01_dictionary.npz \
  --output-dir /path/to/e01_artifacts
```

It uses the original canonical posterior probabilities without renormalizing over stable motifs, then 3x3 spatial local maxima, radius-2 NMS, the lowest `tau*` with median Train nodes/image <=64, cap 80, and top-2 fallback. It writes `e01_occurrence_summary.json` and compact count diagnostics. If the canonical dictionary contains no stable components, this phase records `NO_STABLE_COMPONENTS` rather than fabricating fallback motifs.

A successful implementation run is not itself evidence that motifs exist; the scientific decision is made only from the generated E0.1 artifacts.
