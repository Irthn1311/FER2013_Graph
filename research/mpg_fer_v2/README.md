# MPG-FER v2

Implementation for [Issue #92](https://github.com/Irthn1311/FER2013_Graph/issues/92). This directory is independent of the frozen `research/mpg_fer_v1` empirical baseline.

MPG-FER v2 remains a pure Pixel → Motif → Graph model: 2,304 explicit pixel nodes, four 96D edge-aware pixel GNN blocks, differentiable assignment to 48 motif prototypes, 49 aligned multiscale motif occurrence nodes, five 192D geometry-aware motif graph blocks, and a seven-class classifier. It contains no CNN backbone, hard motif IDs, TopK node removal, or connected-component extraction.

## Issue #92 changes

- bounded learnable assignment temperature (`0.15 <= tau <= 1.50`, initialized at `0.70`);
- normalized mutual-information-style motif loss `H_local - beta*H_global`;
- aligned reflection-mapped 8x8, 12x12, and 16x16 supports fused into exactly 49 nodes;
- 0.15 pixel/motif dropout, 0.30 classifier dropout, and per-residual DropPath;
- EMA validation and inference weights with decay 0.999;
- deterministic 20% flip-consistency scheduling with symmetric JS divergence;
- K/V projection before neighbor gather;
- 120-epoch warmup/cosine horizon and 50/20 minimum-epoch/patience contract;
- atomic full-state epoch-boundary resume bundles and proactive Kaggle segmentation.

## Local validation

From the repository root with the project PyTorch environment:

```powershell
$env:PYTHONPATH = (Resolve-Path 'research\mpg_fer_v2\src').Path
python research\mpg_fer_v2\tools\sync_notebook.py
python -m pytest -q research\mpg_fer_v2\tests
python research\mpg_fer_v2\tools\bounded_gpu_audit.py --batch-size 16
```

The GPU audit is bounded synthetic execution only. It does not open FER2013 or start the 120-epoch experiment.

## Kaggle fresh/resume workflow

Use `notebooks/MPG_FER_v2_Kaggle_T4.ipynb`. Only its top configuration cell is intended for runtime edits:

```python
RESUME_MODE = "auto"
RESUME_PATH = None
SEGMENT_NUMBER = 1
OUTPUT_DIR = "/kaggle/working/mpg_fer_v2_run"
```

Auto mode starts fresh unless exactly one explicitly named `mpg-fer-v2-resume*` input dataset contains a hash-valid `resume_latest.pt` and `resume_latest.json`. A continuation restores model, EMA, AdamW, scheduler, GradScaler, comparator, patience, history, all RNG states, DataLoader generator state, and optimizer-step count before `next_epoch`. It never treats `best_val_acc.pt` as a training resume bundle and cannot auto-discover v1 artifacts.

Concrete Kaggle continuation lifecycle:

1. Segment N finishes its current epoch, writes and fsyncs `resume_latest.pt`, `resume_latest.json`, history, and `segment_manifest.json`, then returns normally with `NEEDS_RESUME` before the 10.5-hour limit. The notebook creates `/kaggle/working/mpg_fer_v2_artifacts.zip` and completes successfully.
2. Download/persist the segment output. Create a **private Kaggle dataset** named with prefix `mpg-fer-v2-resume-`, for example `mpg-fer-v2-resume-<run-id>-seg01`. Upload the unzipped `resume_latest.pt` and `resume_latest.json`; include `resume_epoch_*.pt/.json` fallbacks when present.
3. Attach that private dataset to the next kernel version. Kaggle mounts it below `/kaggle/input` (a nested owner/dataset path is supported). `/kaggle/working` from the previous session is never assumed to survive.
4. Keep the reviewed notebook/source unchanged, set `RESUME_MODE="auto"`, increment `SEGMENT_NUMBER`, and run. Auto mode only scans paths having an `mpg-fer-v2-resume*` directory component, requires exactly one `resume_latest.pt`, validates its sidecar SHA-256, then the loader validates schema, run ID, source hash, and scientific config hash.
5. If latest SHA validation fails, execution stops and reports the newest independently hash-valid immutable snapshot. It never silently falls back; attach or specify a chosen verified fallback explicitly after review.

Sample order is derived from `(base_seed, epoch)`, while flip/translation/intensity augmentation is derived from `(base_seed, epoch, sample_index)`. This makes continuation independent of worker scheduling and worker-process RNG history. The segment guard reserves a default 15-minute safety margin for serialization and kernel completion.

Full training is intentionally not authorized in this implementation round. See `ROUND1_V2_IMPLEMENTATION_REPORT.md` for validation evidence and remaining risks.
