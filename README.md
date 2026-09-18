# FER2013 Graph Research

Graph-based facial expression recognition experiments on **FER-2013**, with a current focus on pixel-level graph representations, motif discovery, class-aware graph retrieval, and graph matching.

> **Status:** active research repository. This README describes the current experimental direction; benchmark numbers are intentionally not claimed here unless they are backed by a reproducible run.

## Research focus

The current D5 direction represents each 48×48 facial image as a graph and learns class-level node/edge motif prototypes. For each expression class, the model retrieves a soft subgraph and produces class logits through graph matching.

```text
FER-2013 CSV
  -> full 48x48 pixel graph
  -> class-level node/edge motif prototypes
  -> class-conditioned soft subgraph retrieval
  -> graph matching
  -> 7-class expression prediction
```

This branch is intentionally graph-first: no CNN classifier branch is required by the D5 pipeline.

## Data contract

Typical batch tensors:

| Tensor | Shape |
| --- | --- |
| node features | `[B, 2304, 7]` |
| edge index | `[2, 17860]` |
| edge attributes | `[B, 17860, 5]` |
| node mask | `[B, 2304]` |
| labels | `[B]` |

Main model outputs include class logits, node attention, edge attention, class-level node gates, class-level edge gates, and diagnostics.

## Repository map

```text
configs/        experiment configuration
data/           dataset and graph-data utilities
models/         graph models
training/       training logic
evaluation/     metrics and diagnostics
visualization/  motif / attention visualization
scripts/        reproducible command-line entry points
notebooks/      notebook workflows
research/       research notes and experiments
tests/          tests and smoke checks
```

The repository also contains historical experiment notes and handoff documents from earlier research iterations. They are retained for traceability but are not part of the minimal D5 execution path.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the graph repository:

```bash
python scripts/build_graph_repo.py --config configs/d5a.yaml --environment local
```

Run a smoke test:

```bash
python scripts/run_experiment.py   --config configs/d5a.yaml   --environment local   --mode smoke   --max_train_batches 3   --max_val_batches 2   --max_test_batches 2   --batch_size 2
```

Train:

```bash
python scripts/train_d5a.py   --config configs/d5a.yaml   --environment local
```

Evaluate:

```bash
python scripts/evaluate_d5a.py   --config configs/d5a.yaml   --environment local   --checkpoint outputs_local/checkpoints/best.pth
```

Visualize learned motifs / attention:

```bash
python scripts/visualize_d5.py   --config configs/d5a.yaml   --environment local   --checkpoint outputs_local/checkpoints/best.pth   --max_samples 16
```

## Kaggle workflow

Use `notebooks/kaggle_d5_end_to_end.ipynb`.

Recommended workflow:

1. Add FER-2013 train/validation/test CSV files as a Kaggle input.
2. Clone or upload this repository.
3. Start with smoke mode.
4. Move to a full build-and-train run only after the data contract is verified.

## Expected outputs

A complete run can produce:

```text
outputs/
├── checkpoints/
├── evaluation/
│   ├── confusion_matrix.png
│   └── predictions.csv
└── figures/
    ├── d5a_class_gates/
    └── d5a_attention/
```

## Current scope

The D5 path does **not** depend on the older candidate motif bank, 41D descriptor pipeline, D3.1 candidate slots, D4A generic slot pooling classifier, CNN classifier branches, or greedy top-K selection.

## Notes

This repository is used for iterative research, so experimental branches may diverge from `main`. Reproducible results should always be reported together with the exact branch/commit, configuration, data split, and checkpoint-selection protocol.
