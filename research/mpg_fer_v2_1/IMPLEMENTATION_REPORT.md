# MPG-FER v2.1 Implementation Report

## 1. Scope and provenance

- GitHub Issue: [#93](https://github.com/Irthn1311/FER2013_Graph/issues/93)
- Branch: `research/mpg-fer-v2-1-issue93`
- Base commit: `9a8628b4c7f1efa82317fe5094022e721de53ffd`
- Frozen baseline: `research/mpg_fer_v2`
- New implementation: `research/mpg_fer_v2_1`
- V2.1 source SHA-256: `d86d93655c83810d36c89a632baee0745f1de0f0e701b762b719d44445a6e679`
- Generated notebook SHA-256: `5efe63d34ea36c9dd493841d4a6c64edbf30a362beea9a4ddd4edefce4c9182a`

The frozen v2 tree has no tracked diff. No official Kaggle kernel or full FER2013 training was run in this implementation phase.

## 2. Exact differences from v2

The Pixel–Motif–Graph architecture is preserved: 2,304 pixel nodes, 32D relational input, directed 8-neighbor pixel graph, four D=96 Pixel-GNN blocks, 48 soft prototypes, exactly three 8×8/12×12/16×16 motif scales fused into 49 occurrences, five D=192 motif graph blocks, 128D pixel and 384D motif readouts, 512D fusion, and the 512→256→7 classifier. No CNN, ResNet, ViT, hard motif IDs, KMeans, Q_M21, new motif scales, graph-node increase, or hidden-width/depth increase was introduced.

Registered changes:

| Item | Frozen v2 | V2.1 |
|---|---:|---:|
| LR decay horizon | `max_epochs=120` | independent `lr_decay_end_epoch=85` |
| Early-stop monitor start | epoch 1 subject to `min_epochs=50` | epoch 85 |
| Early-stop patience | 20 | 15 |
| Motif temperature | trainable bounded scalar | deterministic serialized schedule |
| `lambda_mi` | 0.05 | 0.025 |
| Pixel dropout | 0.15 | 0.10 |
| Motif dropout | 0.15 | 0.10 |
| Classifier dropout | 0.30 | 0.25 |
| Pixel DropPath maximum | 0.05 | 0.03 |
| Motif DropPath maximum | 0.10 | 0.05 |
| Consistency probability | 0.20 | 0.50 |
| Consistency weight | 0.05 | 0.15 |
| SupCon | absent | 512→128 projection, weight 0.05 |
| Resume schema | 2 | 3 |

## 3. Corrected LR mathematics

For epochs 1–5:

```text
lr(epoch) = 3e-4 * epoch / 5
```

For later epochs:

```text
t = clamp((epoch - 5) / (85 - 5), 0, 1)
lr = 1e-6 + 0.5 * (3e-4 - 1e-6) * (1 + cos(pi*t))
```

Thus epoch 85 reaches the floor and epochs 86–120 stay at `1e-6`. `max_epochs` only bounds valid calls; it does not stretch the cosine. A unit test compares schedulers with `max_epochs=120` and `max_epochs=200` and confirms an identical epoch 1–120 trajectory.

| Epoch | Tested LR |
|---:|---:|
| 1 | `6.0000000000e-5` |
| 5 | `3.0000000000e-4` |
| 20 | `2.7480470704e-4` |
| 40 | `1.7966600314e-4` |
| 50 | `1.2133399686e-4` |
| 62 | `5.7945454578e-5` |
| 70 | `2.6195292961e-5` |
| 80 | `3.8726005797e-6` |
| 85 | `1.0000000000e-6` |
| 86 | `1.0000000000e-6` |
| 120 | `1.0000000000e-6` |

## 4. Early-stop state machine

Checkpoint comparison and saving remain active from epoch 1. The patience transition is:

```text
if improved:
    patience = 0
elif epoch < 85:
    patience = 0
else:
    patience += 1
```

Stopping requires `epoch >= 85` and `patience >= 15`. With no improvement from epoch 85 onward, the earliest stop is epoch 99. Tests verify no patience consumption through epoch 84, increments at epochs 85–99, stopping at 15, and reset on improvement.

## 5. Deterministic motif-temperature schedule

The trainable `raw_tau` parameter was removed. `current_tau` is a registered model buffer. For epoch `e`:

```text
progress = clamp((e - 1) / (35 - 1), 0, 1)
tau = 0.30 + 0.5 * (0.70 - 0.30) * (1 + cos(pi*progress))
```

Tested values are epoch 1 = 0.70, epoch 35 = 0.30, and epochs 36/120 = 0.30. Online and EMA models are set to the deterministic value before training and validation. EMA averages trainable parameters but copies buffers exactly, so it cannot numerically average scheduled tau. Resume bundles explicitly record tau and validate it against `model_state_dict`. Best checkpoints store `scheduled_tau`; a roundtrip test proves a best checkpoint at epoch 10 retains epoch-10 tau after the live model is advanced to the final floor.

## 6. MI and regularization

The normalized v2 objective remains:

```text
L_MI = H_local_normalized - 1.0 * H_global_normalized
```

Only its weight changes from 0.05 to 0.025. No additional entropy-minimization loss or hard assignment was added. Dropout and DropPath values were reduced exactly as registered in Issue #93 and are asserted by configuration/model tests.

## 7. Flip consistency

The existing stateless `(seed, epoch, optimizer-group)` decision remains resume-safe. Probability is 0.50 and the JS weight is 0.15. Flipping is still applied to the image before feature extraction and only to selected accumulation groups. History now includes raw consistency loss, weighted consistency loss, selected group IDs, and selected-group fraction.

## 8. Supervised contrastive formulation

The unchanged 512D fused representation feeds both independent paths:

```text
FER:    fusion(512) -> classifier(512->256->7)
SupCon: fusion(512) -> Linear(512,128) -> LayerNorm -> L2 normalize
```

For normalized embeddings `z`, `s_ij = z_i dot z_j / 0.10`. For each valid anchor, positives are other physical-batch samples with the same label; self is excluded and all other samples participate in the denominator. Anchors without positives are omitted. If no anchor is valid, the implementation returns `embeddings.sum() * 0`, preserving a safe differentiable zero.

The total registered loss is:

```text
CE_final
+ 0.20 * CE_motif
+ 0.05 * CE_pixel
+ 0.01 * L_div
+ 0.025 * L_MI
+ 0.15 * L_consistency  (selected groups only)
+ 0.05 * L_supcon
```

Tests cover finite positives/negatives, self exclusion, skipped invalid anchors, all-unique differentiable zero, L2 normalization, and gradient flow into the fusion tensor, projection head, pixel readout, and motif readout. Changing SupCon-head parameters leaves FER inference logits bit-identical.

## 9. Parameter count

| Model | Trainable parameters |
|---|---:|
| Frozen v2 | 2,238,609 |
| V2.1 | 2,304,528 |
| Absolute delta | +65,919 |
| Percentage delta | +2.944641% |

The SupCon `Linear(512,128)` plus LayerNorm adds 65,920 parameters; removing v2's scalar `raw_tau` gives the net +65,919 delta. Graph widths, depths, nodes, and inference classifier size are unchanged.

## 10. Resume compatibility and invariance

V2.1 preserves atomic saving, SHA sidecars, immutable snapshots, model/EMA/AdamW/scheduler/GradScaler restoration, global step, comparator, best epoch/metrics, patience, history, Python/NumPy/Torch RNG, DataLoader generator, deterministic sampler, deterministic augmentation, and consistency decisions.

Schema 3 additionally records:

- scheduled tau and the epoch it represents;
- early-stop monitor start/active state and patience;
- SupCon projection parameters through model/EMA/optimizer state;
- scheduler decay-end horizon and current phase through scheduler state.

The actual-component trajectory test uses motif composition, scheduled tau, MI, DropPath, EMA, deterministic 50% consistency, SupCon, stateless random augmentation, `EpochRandomSampler`, and `num_workers=2`. Continuous execution and save/terminate/recreate/load/resume match across epochs 1/5/6, 34/35/36, and 84/85/86 for model, EMA, optimizer, scheduler, sample order, consistency decisions, metrics/history, best state, patience, and optimizer steps.

V2 resume artifacts are intentionally incompatible with v2.1 because the scientific config, model state, and schema differ. Auto-discovery only accepts paths beginning `mpg-fer-v2-1-resume`.

Measured local CPU serialization benchmark:

| Item | Measurement |
|---|---:|
| Resume checkpoint | 36.6609 MiB |
| Atomic save + SHA-256 | 0.2218 s |
| Load + SHA-256 | 0.1973 s |
| Schema | 3 |

## 11. Tests

Command:

```powershell
$env:PYTHONPATH = 'D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\research\mpg_fer_v2_1\src'
C:\Users\ADMIN\anaconda3\envs\fer-graph\python.exe -m pytest -q research\mpg_fer_v2_1\tests
```

Result:

```text
43 passed in 200.40s (0:03:20)
```

The generated notebook contains six code cells; all compile, and the notebook/source synchronization test passes. The test suite does not establish FER accuracy or scientific improvement.

## 12. Bounded GPU audit

Command:

```powershell
python research\mpg_fer_v2_1\tools\bounded_gpu_audit.py --batch-size 16
```

Measured environment and result:

| Field | Measurement |
|---|---:|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| PyTorch / CUDA | 2.11.0+cu126 / 12.6 |
| Physical batch | 16 |
| AMP | enabled |
| Three motif scales | enabled |
| Consistency | forced on |
| SupCon | enabled; valid anchor fraction 1.0 |
| Optimizer step | succeeded |
| EMA updates | 1 |
| Peak allocated | 2,817.413 MiB |
| Peak reserved | 3,004.000 MiB |
| Warm-process step time | 2.0648 s |
| Initial cold-process observation | 8.0228 s |

All registered gradient audits were finite and non-zero: pixel projection/readout, assignment query, prototypes, scale gate, scale-8 saliency, motif readout, SupCon projection, and classifier. The raw/weighted consistency and SupCon losses were finite. This is a local bounded audit, not a T4 measurement and not FER training.

## 13. Notebook and Kaggle protections

`notebooks/MPG_FER_v2_1_Kaggle_T4.ipynb` is generated from all package source files. It supports fresh and exact-resume modes, checks source SHA and parameter count, preserves split guards, performs the real-16 gate only for a fresh run, verifies resume identity/state, carries the global best checkpoint between segments, and retains the 10.5-hour limit plus 15-minute margin.

The notebook was generated and statically validated only. It was not submitted to Kaggle in Issue #93.

## 14. Estimated Kaggle runtime

This is an estimate, not a measured v2.1 T4 runtime. Frozen v2 averaged about 324 seconds per epoch with 20% consistency. Raising the selected consistency fraction from 0.20 to 0.50 increases the dominant forward-pass load from roughly 1.2 to 1.5 views per batch; SupCon adds smaller projection/loss overhead. A first-order estimate is approximately 6.5–7.5 minutes per epoch on T4:

- earliest no-improvement stop at epoch 99: approximately 10.7–12.4 hours;
- full 120 epochs: approximately 13–15 hours;
- likely Kaggle segments: two, subject to actual T4 timing and the existing safe segment guard.

Only a reviewed future execution can replace these estimates with measurements.

## 15. Unresolved risks

1. No FER2013 training was run, so better accuracy, F1, calibration, motif quality, or low-LR refinement remains UNKNOWN.
2. SupCon is physical-batch-only. Class imbalance can reduce valid-anchor coverage, especially for rare classes; the logged valid fraction and mean positive count must be inspected in a real run.
3. The fixed tau schedule avoids lower-bound saturation but whether 0.30 is scientifically preferable to v2's learned ~0.15 is unmeasured.
4. Stronger consistency may improve invariance or over-regularize; it also materially increases runtime.
5. The bounded GPU measurement is on an RTX 3050 Ti Laptop GPU, not Kaggle T4. It establishes implementation feasibility and finite gradients, not final T4 memory/time.
6. The resume invariance test uses a compact actual-component harness rather than full 2.3M-parameter FER epochs; full-model state layout, checkpoint roundtrip, notebook synchronization, and bounded full-model backward are covered separately.
7. Starting patience at epoch 85 intentionally guarantees low-LR exposure but may train longer after the validation optimum; this is the registered experiment, not a proven improvement.

## 16. Final status

`READY_FOR_V2_1_REVIEW`
