# MPG-FER v2.1

Implementation for [Issue #93](https://github.com/Irthn1311/FER2013_Graph/issues/93). This directory is independent of the frozen, successful `research/mpg_fer_v2` baseline and does not modify it.

MPG-FER v2.1 remains a pure Pixel → Motif → Graph model: 2,304 explicit pixel nodes, four 96D edge-aware Pixel-GNN blocks, 48 soft motif prototypes, aligned 8×8/12×12/16×16 supports fused into exactly 49 motif occurrence nodes, five 192D geometry-aware motif graph blocks, 128D pixel plus 384D motif readouts, and the unchanged 512→256→7 FER classifier. It contains no CNN/ResNet/ViT backbone, KMeans, hard motif IDs, Q_M21, or graph-size inflation.

## Registered v2.1 changes

- LR: epochs 1–5 linear warmup, epochs 6–85 cosine decay to `1e-6`, then a fixed floor through `max_epochs=120`.
- Early stopping: checkpoint selection starts at epoch 1, but the 15-epoch patience counter starts only at epoch 85.
- Motif temperature: non-trainable serialized buffer, cosine scheduled from 0.70 at epoch 1 to 0.30 at epoch 35, then held fixed. EMA copies this buffer exactly rather than averaging it.
- MI weight: `0.05 → 0.025`, with the v2 normalized MI formulation retained.
- Regularization: pixel/motif dropout `0.10`, classifier dropout `0.25`, Pixel DropPath maximum `0.03`, Motif DropPath maximum `0.05`.
- Flip consistency: deterministic probability `0.50`, JS weight `0.15`, with flipping before relational feature extraction.
- SupCon: a training-only `Linear(512,128) → LayerNorm → L2 normalize` head, temperature `0.10`, and loss weight `0.05`. FER inference logits remain on the unchanged classifier path.
- Resume schema 3 records the scheduled temperature and early-stop monitor phase in addition to the complete hardened v2 state.

Trainable parameter count is `2,304,528`: an increase of `65,919` (`2.9446%`) from v2's `2,238,609`. This is the SupCon projection/normalization addition minus the removed scalar `raw_tau` parameter.

## Local validation

From the repository root with the project PyTorch environment:

```powershell
$env:PYTHONPATH = (Resolve-Path 'research\mpg_fer_v2_1\src').Path
python research\mpg_fer_v2_1\tools\sync_notebook.py
python -m pytest -q research\mpg_fer_v2_1\tests
python research\mpg_fer_v2_1\tools\bounded_gpu_audit.py --batch-size 16
```

The GPU command is one bounded synthetic AMP step with worst-case consistency forced on and SupCon active. It does not access FER2013 or start full training.

## Kaggle fresh/resume candidate

`notebooks/MPG_FER_v2_1_Kaggle_T4.ipynb` is generated from the package sources; do not hand-copy source into it. Its top configuration cell supports `RESUME_MODE="fresh"`, `"auto"`, or `"required"`.

Auto-resume only discovers explicitly named `mpg-fer-v2-1-resume*` inputs and restores the complete run state after SHA, schema, run-ID, source-hash, and config-hash validation. It deliberately cannot discover v1 or v2 resume artifacts. The 10.5-hour segment limit and 15-minute safety margin are unchanged.

Official Kaggle training is not authorized in Issue #93. See `IMPLEMENTATION_REPORT.md` for implementation evidence, the bounded local audit, estimates, and unresolved scientific risks.
