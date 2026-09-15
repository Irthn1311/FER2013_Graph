# Pixel-GNN Only — FER2013 / TensorFlow / Kaggle

Architectural ablation specified in the user's chat, implemented separately from
the frozen OFIX7-mid reference at base commit
`62db5a620b61e129fa1bcabf1bd25418a4578b7e`.
No existing full-model source, configuration, runner, checkpoint or artifact was edited.

## Architecture

```text
48×48 grayscale
  → 2304 pixel nodes
  → [intensity, x_norm, y_norm, gx, gy]
  → shared Linear(5,96) + GELU + dropout(0.2)
  → 3 reference GatedEdgeLayer blocks (96 hidden, 32 edge hidden, dropout 0.25)
  → global mean over all 2304 node embeddings
  → Linear(96,7)
```

There is no CNN, convolution layer, MediaPipe invocation, landmark channel,
anatomical mask, anchor node, part context, attention or motif readout.
The original `GatedEdgeLayer` class is imported unchanged from the frozen
TensorFlow package. Its six-dimensional edge input is the architectural change
needed to remove the two landmark-dependent edge channels.
The reference package's Python model imports expose full-model class definitions;
the ablation never constructs those models/readouts or calls a prior loader.

The graph uses exactly the reference directed 8-neighborhood edge builder:
17,860 directed edges per image, no self loops, no KNN edges or anchors.
Coordinates use the reference normalization to `[-1,1]`; gradients use the
reference NumPy finite differences. Intensity is raw FER intensity divided by 255.

Retained edge channels, in their reference order:
`dx`, `dy`, `spatial_dist`, `abs_intensity_diff`, `abs_grad_mag_diff`,
`abs_laplacian_diff`. Gradient magnitude and absolute Laplacian are deterministic
pixel descriptors used only for edges; they are not additional node channels
or trainable convolution layers. `part_similarity` and `same_dominant_part` are absent.

Expected tensor shapes for batch size `B`:

| Tensor | Shape |
| --- | --- |
| Node features | `[B×2304, 5]` |
| Edge index | `[2, B×17860]` |
| Edge features | `[B×17860, 6]` |
| GNN node embeddings | `[B×2304, 96]` |
| Global pooled embedding | `[B, 96]` |
| Logits | `[B, 7]` |

Trainable parameters calculated from layer dimensions: **189,319**.
Kaggle reports the actual count and fails if it differs.

## Frozen training protocol

The ablation YAML extends `fer2013_ofix7_mid_tensorflow_seed42.yaml`.
The original batch16 configuration requires the resolved `training` and `loss`
mappings and the seed to equal that baseline. The separate fast configuration
allows the user's explicitly requested batch-size change only. All other
training settings remain equal; learning rate is not scaled automatically.

- Official train/validation/test counts: 28,709 / 3,589 / 3,589; no resplitting.
- Seed 42; same deterministic shuffle (`seed + epoch×1,000,003`).
- Training batch 16; evaluation batch 32, matching the baseline CLI.
- Maximum 90 epochs, same validation-loss early stopping (minimum 30, patience 15).
- Same CE-only objective, zero label smoothing.
- Same `TorchCompatibleAdamW`, LR `0.0003`, weight decay `0.001`, betas
  `0.9/0.999`, epsilon `1e-8`, and global gradient clipping at 5.
- Same validation-loss plateau scheduler, factor 0.5, patience 5, threshold
  `0.0001`, minimum LR `0.00003`.
- Same compiled gradient/update execution, G1-A optimizer settings and mixed precision.
- Same clean train evaluation every epoch and validation metrics.
- Same strict validation-accuracy improvement / earliest tie checkpoint selection.
- Test is loaded only after training, evaluated once from `best_val_accuracy.keras`.

The frozen TensorFlow loader applies no image augmentation. Its corruption
schedule modifies landmark priors only. Pixel images are unchanged here;
prior corruption is disabled because priors do not exist in this architecture.
No replacement image augmentation is introduced.

Use exactly the same FER CSV source/row order used to create the full baseline's
inputs. The loader checks official split sizes; without reading baseline priors
it cannot independently certify the provenance of your Kaggle dataset.

Accepted data layouts:

1. A directory containing `train.csv`, exactly one of `val.csv` / `validation.csv`,
   and `test.csv`, with `emotion,pixels` columns. Pass its `train.csv`.
   These files may retain a `Usage` column. Named split files take precedence;
   their rows are not filtered again by `Usage`. The loader logs the selected
   file and parsing progress. A single CSV with `Usage` and no sibling split
   files still uses the official Usage groups.
2. Original `fer2013.csv` with `emotion,pixels,Usage`, using `Training`,
   `PublicTest`, `PrivateTest` as train, validation, test respectively.

No MediaPipe dataset or full-model graph cache needs to be attached.

## Kaggle fast configuration — two GPUs / batch32

The runner now defaults to `fer2013_pixel_gnn_only_kaggle_fast_seed42.yaml`:

- Global training batch **32**, split as **16 complete graphs per GPU** on two T4s.
- Evaluation batch remains 32. Train/validation/test row order is preserved.
- Forward/backward execute in separate GPU:0/GPU:1 branches of a compiled TF graph.
- Gradients use the actual global example count, including a short final batch.
- Gradients are summed on GPU:0, then the original AdamW performs global
  clipping and one update. GPU:1 weights synchronize before the step returns.
- Both GPUs also run validation, clean train evaluation and final-test inference.
  The original evaluation loss aggregation and checkpoint policy are retained.
- CPU workers and TensorFlow threads are sized from CPU affinity/cgroup quota;
  prefetch is 4, with a bounded graph cache of 256 entries.
- Train evaluation reuses already loaded train images instead of reparsing CSV.
- Input tensors are collated on CPU and sliced by whole graph before GPU transfer.
- The configured FER CSV is used directly; there is no recursive input-tree scan.

This is explicit data parallelism with two model copies and one authoritative
optimizer; it does not wrap the frozen custom AdamW in MirroredStrategy.
Saved checkpoints contain the authoritative Pixel-GNN model, not an extra
scientific head or an ensemble. Parameters per model remain 189,319.

Batch32 changes the number of optimizer steps per epoch (898 rather than 1,795).
Consequently this throughput run is **not an architecture-only comparison**
against the batch16 full baseline. Resolved config/provenance records original
and effective batch sizes and `authorized_training_changes: [batch_size]`.
Seed remains 42; parallel RNG/reduction order is not guaranteed to reproduce
single-GPU floating-point results bit for bit. Runtime correctness and speed
must be verified on Kaggle; no two-GPU runtime was executed locally.

## Run on Kaggle

First upload/copy this edited repository to Kaggle, including both standalone
directories, `train.py` and `run_pixel_gnn.py`.
Enable a GPU accelerator. Run commands from the repository root.
Use `--gpus 2` to require both GPUs, or `--gpus 1` for the single-device path.
Automatic mode uses up to two visible GPUs. CPU/GPU utilization depends on
input preparation, kernel scheduling and gradient transfers; 100% is not promised.

Supported environment: Python 3.10–3.12, TensorFlow 2.13–2.18.
If needed, install the supplied runtime pins in a notebook cell, then restart
the session before importing TensorFlow:

```python
!pip install -r standalone/pixel_gnn_only/requirements-kaggle.txt
```

Optional one-batch smoke on Kaggle (train data only):

```python
!python -u run_pixel_gnn.py --gpus 2 --smoke --fer-csv /kaggle/input/YOUR_DATASET/fer13-split/train.csv
```

Or check architecture/gradients explicitly using synthetic data:

```python
!python -u run_pixel_gnn.py --gpus 2 --smoke --synthetic
```

Run using the YAML configuration (batch32):

```python
!python -u train.py --config standalone/pixel_gnn_only/configs/fer2013_pixel_gnn_only_kaggle_fast_seed42.yaml
```

`train.py` delegates to the same Pixel-GNN runner; it does not start the full
LAP-GNN model. No training logic is duplicated. Edit the YAML's `paths.fer_csv`
to match your attached Kaggle dataset and `paths.output_root` for outputs.
`runtime.gpus: auto` uses up to two visible GPUs; set it to `2` to require both.
To change batch size entirely in YAML, set `data.batch_size`,
`training.batch_size` and `resources.batch_size` to the same number.
The supplied fast YAML has all three set to 32.

For a larger throughput run use `--batch-size 64` (32 graphs/GPU). Its T4
memory requirements have not been measured; batch32 is the supplied default.
For the original strict batch16 protocol explicitly select the original config:

```python
!python -u run_pixel_gnn.py --gpus 2 --config standalone/pixel_gnn_only/configs/fer2013_pixel_gnn_only_seed42.yaml --fer-csv /kaggle/input/YOUR_DATASET/fer13-split/train.csv
```

Optional synthetic regression checks on Kaggle:

```python
!python standalone/pixel_gnn_only/tests/test_dataset_splits.py
!python standalone/pixel_gnn_only/tests/test_parallel_runtime.py
```

The latter checks graph rebasing, batch override guards, odd/single-graph
evaluation and the custom optimizer's global clipping/update against a
single-device update on a small synthetic model. They are not FER accuracy tests.

Both training and smoke print the model summary, layer inventory, absence of
Conv/CNN layers, shapes and actual parameter count. Smoke additionally verifies
finite gradients for all 58 trainable variables and a compiled optimizer update.
Smoke does not evaluate validation/test and cannot establish accuracy.
Training does not automatically run an extra smoke update before the first epoch.

Fast outputs are separate, under `/kaggle/working/outputs/pixel_gnn_only_fast_bs32_seed42`
and `/kaggle/working/outputs/pixel_gnn_only_smoke`. Existing nonempty output
directories passed explicitly are rejected. For the default output path the
runner appends a timestamp if needed, preserving all prior artifacts.

Artifacts include `model_summary.txt`, `architecture_report.json`, resolved
configs, environment/source/input provenance, history, training curves,
validation-selected Keras model/weights/metadata, final test metrics,
prediction probabilities, per-class metrics and confusion matrix.
`smoke_report.json` is produced only by `--smoke`.

## Files created

```text
train.py
run_pixel_gnn.py
standalone/pixel_gnn_only/README.md
standalone/pixel_gnn_only/requirements-kaggle.txt
standalone/pixel_gnn_only/configs/fer2013_pixel_gnn_only_seed42.yaml
standalone/pixel_gnn_only/configs/fer2013_pixel_gnn_only_kaggle_fast_seed42.yaml
standalone/pixel_gnn_only/pixel_gnn_only/__init__.py
standalone/pixel_gnn_only/pixel_gnn_only/artifacts.py
standalone/pixel_gnn_only/pixel_gnn_only/batching.py
standalone/pixel_gnn_only/pixel_gnn_only/dataset.py
standalone/pixel_gnn_only/pixel_gnn_only/evaluator.py
standalone/pixel_gnn_only/pixel_gnn_only/execution.py
standalone/pixel_gnn_only/pixel_gnn_only/graph.py
standalone/pixel_gnn_only/pixel_gnn_only/model.py
standalone/pixel_gnn_only/pixel_gnn_only/parallel.py
standalone/pixel_gnn_only/pixel_gnn_only/protocol.py
standalone/pixel_gnn_only/pixel_gnn_only/runtime.py
standalone/pixel_gnn_only/pixel_gnn_only/smoke.py
standalone/pixel_gnn_only/pixel_gnn_only/trainer.py
standalone/pixel_gnn_only/tests/test_dataset_splits.py
standalone/pixel_gnn_only/tests/test_parallel_runtime.py
```

The initial ablation left full-model files unchanged. The subsequent split-loader
fix changes only this ablation's dataset loader, this README and adds synthetic
CSV regression tests. Per the user's instruction, no local
tests, smoke, installation or training were run for this implementation.
Kaggle runtime, checkpoint roundtrip and accuracy remain unverified until run.
This implementation is not a parity claim for the changed architecture.
