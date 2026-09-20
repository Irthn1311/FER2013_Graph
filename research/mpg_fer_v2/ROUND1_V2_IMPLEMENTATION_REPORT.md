# MPG-FER v2 — Round 1 Implementation Report

> Historical Round 1 snapshot. Superseded for execution readiness by `ROUND2_V2_FINAL_REVIEW.md`; its source hash and test count intentionally describe the pre-audit state.

Issue: [#92](https://github.com/Irthn1311/FER2013_Graph/issues/92)  
Base commit: `3c58b44f1ee4860bdd0a340d35b191b2da4db3c4`  
Reviewed source-tree SHA-256: `b762a2ad81979bae6d48b0af005e181be9e65e930fd1e54e40feab351234b2a0`  
Frozen empirical baseline: `research/mpg_fer_v1` (not modified)  
Scope boundary: implementation, deterministic tests, notebook generation, resume simulation, and bounded local GPU audit only. No FER2013 full training or Kaggle v2 run was launched.

## 1. Files created

Implementation package:

- `src/mpg_fer_v2/config.py`
- `src/mpg_fer_v2/data.py`
- `src/mpg_fer_v2/features.py`
- `src/mpg_fer_v2/graph.py`
- `src/mpg_fer_v2/losses.py`
- `src/mpg_fer_v2/motif.py`
- `src/mpg_fer_v2/model.py`
- `src/mpg_fer_v2/ema.py`
- `src/mpg_fer_v2/checkpoint.py`
- `src/mpg_fer_v2/train.py`
- `src/mpg_fer_v2/evaluate.py`
- `src/mpg_fer_v2/kaggle.py`
- `src/mpg_fer_v2/utils.py`
- `src/mpg_fer_v2/__init__.py`

Validation and delivery:

- `tests/test_motif_v2.py`
- `tests/test_model_v2.py`
- `tests/test_ema_resume.py`
- `tests/test_protocol_notebook.py`
- `tools/sync_notebook.py`
- `tools/bounded_gpu_audit.py`
- `notebooks/MPG_FER_v2_Kaggle_T4.ipynb`
- `README.md`
- this report

## 2. Architecture changes from v1

The 48x48 → 2,304 pixel-node → four 96D Pixel-GNN → 49 motif-node → five 192D Motif-Transformer → 512D joint-readout identity is preserved. Core widths, depths, head counts, motif count, occurrence count, and classifier dimensions are unchanged.

V2 changes only the registered targets: Q/K cosine soft assignment, bounded temperature, MI motif objective, aligned multiscale occurrence composition, learned three-scale fusion, stronger dropout/DropPath, EMA, limited flip consistency, and linear-before-gather K/V projection. No convolution, hard quantization, node removal, or extra motif graph was introduced.

## 3. Parameter count

Measured locally from the instantiated model:

```text
MPG-FER v1: 2,219,788
MPG-FER v2: 2,238,609
Increase:      18,821 (0.848%)
```

The increase comes from assignment Q/K projections, three small scale-local saliency heads, and the scale gate. Core graph capacity is unchanged.

## 4. Motif MI formulation

For assignments `A = p(m|h)`:

```text
H_local_raw        = mean_i[-sum_m A_i,m log A_i,m]
H_global_raw       = -sum_m A_bar_m log A_bar_m
H_local_normalized = H_local_raw / log(48)
H_global_normalized= H_global_raw / log(48)
L_MI               = H_local_normalized - beta * H_global_normalized
```

Defaults are `beta=1.0` and `lambda_mi=0.05`. The v1 utilization KL is absent, avoiding duplicate strong pressure toward uniform use. Prototype diversity regularization remains at `lambda_div=0.01`.

## 5. Temperature implementation

`raw_tau` is optimized, while the used temperature is:

```text
tau = 0.15 + (1.50 - 0.15) * sigmoid(raw_tau)
```

It initializes to approximately 0.70 and cannot approach zero. `tau` is included in every epoch's diagnostics and normal model/optimizer/resume state.

## 6. Multiscale implementation

The 7x7 anchor grid retains the v1 12x12/stride-6 conceptual centers `(5.5, 11.5, ..., 41.5)` on each axis. Precomputed reflection-mapped supports have exact shapes:

```text
8x8:  [49,  64]
12x12:[49, 144]
16x16:[49, 256]
```

Each scale independently pools WHAT, TYPE, and WHERE through scale-local learned occurrence weights and produces a 192D candidate. A learned softmax gate produces three weights per sample and anchor. Candidate representations and learned centers are fused with the same weights, so geometry remains in autograd. The downstream graph still has exactly 49 nodes.

## 7. Regularization changes

Defaults are pixel dropout 0.15, motif dropout 0.15, and classifier dropout 0.30. Per-sample residual DropPath increases linearly from 0.00 to 0.05 across four pixel blocks and 0.00 to 0.10 across five motif blocks. It is disabled by `eval()` and never drops graph nodes.

## 8. EMA implementation

`ModelEMA(decay=0.999)` deep-copies the complete model state. Floating tensors are exponentially averaged; non-floating buffers are copied exactly. EMA updates occur only after a successful optimizer step, not on accumulation microsteps. Public raw/TTA validation, best-checkpoint selection, and final inference use EMA weights. `best_val_acc.pt` is explicitly marked `EMA_INFERENCE_ONLY`.

## 9. Flip consistency implementation

The normal-label CE remains on the unflipped image. A stateless `(seed, epoch, accumulation_group)` decision selects approximately 20% of optimizer groups for an extra horizontal-flip forward. Symmetric Jensen-Shannon divergence between final classifier distributions is weighted by `lambda_consistency=0.05`. The decision is deterministic under resume and no intermediate node coordinates are constrained.

## 10. Training horizon and scheduler

Defaults are `max_epochs=120`, `min_epochs=50`, and `early_stop_patience=20`. LR warms linearly to `3e-4` over epochs 1–5 and cosine-decays to `1e-6` over epochs 6–120. Scheduler state is serialized and compatibility-checked; a resume continues at `next_epoch` with restored AdamW LR/moments and GradScaler state.

## 11. Exact resume state schema

`resume_latest.pt` schema version 2 contains:

```text
run_id, source_hash, config_hash, scientific_config
completed_epoch, next_epoch, global_optimizer_step
model_state_dict, ema_state_dict
optimizer_state_dict, scheduler_state_dict, grad_scaler_state_dict
best_comparator_state, best_epoch, best_metrics, early_stop_counter
history
Python, NumPy, torch CPU, torch CUDA RNG states
training DataLoader generator state
consistency scheduling state
```

The file is written to `resume_latest.tmp`, flushed/fsynced, and atomically replaced. A small `resume_latest.json` records the checkpoint SHA-256 and identity metadata. Immutable `resume_epoch_NNN.pt` snapshots are made every 10 epochs, retaining at least the latest two.

Config compatibility is fail-closed. Only `num_workers`, output path, resume input path, and segment number are ignored by the scientific config hash. Architecture, objective, schedule, seed, and device must match. Source hash and optional expected run ID must also match.

## 12. Kaggle segmentation design

At each epoch boundary the full resume bundle is durable before another epoch is considered. The 10.5-hour guard uses recent worst epoch duration; if another epoch risks crossing the limit it writes status `NEEDS_RESUME` and exits cleanly. Segment manifests support `NEEDS_RESUME`, `TRAINING_COMPLETED`, and `FAILED`.

Notebook auto-discovery accepts only a unique `resume_latest.pt` beneath an explicitly named `mpg-fer-v2-resume*` attached input and requires matching `resume_latest.json` SHA-256. It cannot discover v1 paths or use `best_val_acc.pt` for continuation. On resume it prints run ID, completed/next epoch, restored LR, optimizer step, best epoch, and patience.

## 13. Tests run

Environment:

```text
Python environment: C:\Users\ADMIN\anaconda3\envs\fer-graph
PyTorch: 2.11.0+cu126
CUDA build: 12.6
Command: python -m pytest -q research\mpg_fer_v2\tests
Result: 20 passed (final verification)
```

Covered requirements include bounded temperature; exact MI entropy values; MI gradients to prototypes and assignment projections; 8/12/16 support shapes and common centers; 49-node output; scale simplex; center gradient flow; DropPath train/eval behavior; EMA update/serialization; differentiable finite JS consistency; linear-before-gather equivalence; full resume round-trip; optimizer, scheduler/LR, GradScaler, RNG, comparator, patience, and history restore; and notebook source synchronization/resume routing.

## 14. Resume-invariance result

PASS. A deterministic tiny training trajectory run continuously for four epochs was compared with two epochs → atomic save → new process state/model objects → full restore → two epochs. Final online model tensors, EMA tensors, full AdamW state including step counters and moments, LR/scheduler state, and history were exactly equal. Independent tests also restored Python, NumPy, torch CPU, and DataLoader-generator RNG streams exactly.

## 15. Bounded GPU result

Measured local worst-case batch-16 AMP step, including the additional flip-consistency forward and backward:

```text
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GiB)
physical batch: 16
AMP: enabled
loss: 2.4585328102 (finite)
peak allocated: 2,812.6519 MiB
peak reserved:  3,000.0000 MiB
result: PASS
```

This is real local GPU evidence but not a T4 measurement. No FER2013 data was opened by this synthetic bounded audit.

## 16. Estimated epoch runtime

The measured v1 Kaggle T4 epoch time was approximately 317.5–318.0 seconds. V2 adds multiscale pooling and a second forward on about 20% of accumulation groups, while linear-before-gather reduces repeated K/V projection work. A conservative unmeasured estimate is **330–390 seconds per T4 epoch**. This is an estimate, not runtime evidence; a reviewed Kaggle bounded audit is required before scheduling a full run.

## 17. Unresolved risks

- No Kaggle T4 v2 execution has occurred; T4 peak VRAM and epoch runtime remain unmeasured.
- The 120-epoch scientific outcome, motif specialization, generalization gap, and accuracy are UNKNOWN until the reviewed run.
- Multi-segment continuation is covered by deterministic local simulation and notebook static synchronization, but has not yet crossed a real Kaggle session boundary.
- Worst-case local batch-16 memory fits a 4 GiB RTX 3050 Ti, but Kaggle library/kernel allocation differences still require a T4 gate.
- Lower motif entropy is diagnostic only and is not treated as scientific success without Public classification improvement.

## 18. Status

`READY_FOR_V2_REVIEW`
