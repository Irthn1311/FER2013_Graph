# Pure-GNN v3.1: Raw Pixel Graph FER Research Line

## Overview

`pure_gnn_v31` is an isolated research line and falsification platform for graph-based Facial Expression Recognition (FER2013).
It operates strictly from raw 48x48 pixel intensities with NO Conv2D, NO CNN backbone, NO MediaPipe, NO facial landmarks, NO superpixels, and NO Transformer blocks.

## Scientific Conditions

- **G0**: Local-only pixel graph processing (no non-local communication at coarse stage).
- **G0.5**: Uniform receiver-relative non-local context on 6x6 coarse graph ($36 \times 35 = 1260$ directed non-self edges, $w_{ij} = 1/35$).
- **G1**: Learned pair-specific relational graph with zero-initialized normalized pair gates.
- **G2**: Geometry-only local control (masks local node content features entering local gate to zero).
- **G3**: Coarsening control (simple local mean + stride-2 downsampling instead of fixed anti-aliased coarsening).

The primary scientific comparison is **G1 versus G0.5**.

## Research Governance

- Off-limits: Official test set (`test.csv`) and official validation set (`val.csv`) during architecture design and preflight.
- Research splits derived strictly from the 28,709-row official `train.csv`.
- Scientific training is disabled by default in preflight orchestrator notebooks.
