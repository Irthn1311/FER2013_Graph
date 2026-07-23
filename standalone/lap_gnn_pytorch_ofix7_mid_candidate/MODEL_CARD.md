# Model Card

## Status

Baseline candidate only. S1/O1 optimizer and scheduler experiments are outside
this package and have not been promoted.

## Model

OFIX7-mid LAP-GNN uses grayscale FER2013 pixels, precomputed MediaPipe-derived
priors, variable-size face/context pixel graphs, five semantic anchor nodes,
three gated edge-aware GNN layers, and micro-motif-support readout.

## Locked Five-Seed Result

- Accuracy: 64.6698 +/- 0.5338%
- Macro-F1: 62.9967 +/- 0.8314%
- Weighted-F1: 64.5723 +/- 0.4959%
- Replication status: `STRONG_REPLICATION`

These are historical locked artifacts, not results produced by package smoke
tests.

## Limitations

FER2013 is noisy and imbalanced. The model consumes landmark-derived priors and
is not a no-prior pixel-only system. Five seeds remain a small replication set.
