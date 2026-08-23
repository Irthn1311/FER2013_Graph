# TensorFlow Step 4 seed42 validation-only baseline

## Evidence disposition

**CENSORED BUT DECISION-SUFFICIENT FOR THE PRIMARY DIAGNOSTIC.**

The registered Kaggle T4 run was censored by the Kaggle session time limit at
approximately 42,886 seconds (about 11 h 54 m). It is not a normally completed
baseline and did not reach the validation-only completion-marker boundary.
Thirty-one epochs are fully persisted. Epoch 32 completed training and
validation, but the timeout occurred during its clean full-train evaluation.

The run must not be resumed or rerun for evidence recovery. The persisted
primary diagnostic is final under the frozen policy: epoch 26 remains the best
validation-macro-F1 epoch, and the console-only epoch-32 validation macro-F1 is
lower. Independently, the persisted epoch-31 early-stop wait is 14/15 and the
epoch-32 validation loss is worse than the best validation loss, so the frozen
early-stopping update would deterministically request stop at epoch 32. This
stopping consequence is an inference, not a persisted epoch row.

## Evidence sources and integrity

- Preregistration: GitHub Issue `#7`, including
  [Protocol Amendment A](https://github.com/Irthn1311/FER2013_Graph/issues/7#issuecomment-5379703960).
- Runtime disposition:
  [Issue #7 runtime-forensic comment](https://github.com/Irthn1311/FER2013_Graph/issues/7#issuecomment-5387049394)
  and [PR #8 runtime-evidence comment](https://github.com/Irthn1311/FER2013_Graph/pull/8#issuecomment-5387050517)
  dated 2026-08-23.
- Downloaded archive: `outputs/new_fix/kaggle-issue7-validation-only.zip`.
- Archive size: `44,856,891` bytes.
- Archive SHA-256:
  `231d65bb6a670e49f1a589cdf0657d181315d42e1fc4955c1debdc15ed441c92`.
- Persisted history SHA-256:
  `4e45a57a0a68a256d1a112dc00a724b67828c4b344a32bb49b7dd334025d95f3`.
- Persisted resolved-config JSON SHA-256:
  `3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32`.
- Persisted latest-epoch summary SHA-256:
  `e9ea14ef9a4aa6ab4f72a20361c8fe3ba9f50fe831ff60a7926c9bc53deaf213`.
- Persisted runtime-progress SHA-256:
  `763f2b8fee93b8269a6f77142e58db64d82a55c47f81b09ee76f70a046d1cf1b`.
- Validation-only completion-marker SHA-256: `UNKNOWN` — marker missing due
  timeout before the wrapper boundary.
- Final telemetry SHA-256: `UNKNOWN` — final telemetry missing due timeout.
- Raw Kaggle console log: not present in the downloaded ZIP or local evidence
  folder. Epoch-32 console values below are transcribed from the authoritative
  runtime comments, which state that they were verified against the downloaded
  console log; they are not attributed to `history.json`.

The read-only audit consumed only extracted copies of `history.json` and
`resolved_config.json`. It did not load weights, run training or inference, or
read a test artifact. Audit version: `1.0.0`; `test_artifacts_read: false`.

## Preregistration provenance

- Base commit: `4e3a80525a33679fd9ea8e19a85807d19736c981`.
- Config:
  `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/configs/fer2013_ofix7_mid_tensorflow_seed42.yaml`.
- Input config SHA-256:
  `aa3bf2d3932bbad6c5f8cdcc347f4a9866e2c027d6135a60b5002a8f6a3b6908`.
- Resolved canonical config hash:
  `a4038682bf09c03786e86119001cf5f81ac5fec25d09062e6a0866484c32a3cf`.
- Validation-only wrapper version: `1.0.0`.
- Wrapper SHA-256:
  `c94c122066fdd19210c8ba64a2a61567b249fad4f69c69cb4236b68cce6ff7b4`.
- Frozen trainer SHA-256:
  `4c3cb1aa311578038ff656cb7d119103ae5a651135f8ee1c76e37c2c04c1fc75`.
- Frozen scientific payload SHA-256:
  `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- Execution-contract SHA-256:
  `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`.
- Seed: `42`.
- Bounded Kaggle smoke used: `false`.
- Resume: `false`; the censored run was not resumed.

The notebook constructed the registered argv below. Its first element was the
Kaggle runtime's `sys.executable`, which was not persisted as a literal command
string:

```text
<sys.executable> -B /kaggle/working/FER2013_Graph/standalone/lap_gnn_tensorflow_ofix7_mid_candidate/tools/train_validation_only.py
  --config /kaggle/working/FER2013_Graph/standalone/lap_gnn_tensorflow_ofix7_mid_candidate/configs/fer2013_ofix7_mid_tensorflow_seed42.yaml
  --fer-csv /kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/train.csv
  --prior-root /kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue
  --output-root /kaggle/working/outputs/tf_root_cause/step4_seed42_validation_only/issue7_seed42_kaggle_t4
  --device gpu --graph-workers 2 --tf-data-prefetch 2
  --batch-size 16 --eval-batch-size 32 --mixed-precision
  --no-xla --memory-growth --no-resume
  --clean-graph-cache-dir /kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records
```

No `--limit-*` argument was used.

## Assets and test-isolation boundary

- FER input: registered Kaggle split dataset; only `train.csv` and `val.csv`
  sample files were inspected.
- Prior input: registered MediaPipe prior dataset; only train/validation records
  and the shared prior schema were inspected.
- Clean-cache input: only train/validation indexes and their referenced shards
  were used for sample records.
- The shared cache-root `CACHE_COMPLETE.json` was read as non-sample aggregate
  metadata required by the frozen loader and may summarize the test split.
- Test CSV accessed: `false`.
- Test prior records accessed: `false`.
- Test cache index accessed: `false`.
- Test cache shards accessed: `false`.
- Test labels/metrics/predictions accessed: `false`.
- Test inference run: `false`.
- The archive contains no test metric, prediction, confusion-matrix, normal
  `TRAINING_COMPLETE.json`, or validation-only completion-marker artifact.

## Runtime integrity

- Registered environment: Kaggle Linux, Python `3.12.12`, TensorFlow `2.18.1`,
  Keras `3.15.0`, CUDA `12.5.1`, cuDNN `9`.
- GPUs exposed: two `Tesla T4` devices, driver `580.159.04`, 15,360 MiB each.
- Persisted epochs: `31` (`history.json` and `latest_epoch_summary.json` agree).
- Epoch-32 training progress: batch `1795/1795` persisted in
  `runtime_progress.json`.
- Actual termination: Kaggle session timeout during epoch-32 clean full-train
  evaluation.
- Normal max-epoch completion: `false`.
- Persisted early-stopping completion: `false`.
- Deterministic inferred stopping consequence: epoch 32 would advance the
  early-stop wait from 14/15 to 15/15 and request stop.
- Validation-only completion marker: `MISSING`.
- Final telemetry: `MISSING`.
- Epoch-32 clean-train metrics and train-validation gap: `UNKNOWN`.
- Epoch-32 checkpoint: `MISSING`; the persisted best-validation-accuracy
  checkpoint is epoch 31.
- Test phase reached: `false`.

## Read-only audit of 31 persisted epochs

`tools/audit_learning_history.py` was run without modification against the
persisted `history.json` and `resolved_config.json` extracted from the archive.

| Measurement | Persisted audit value |
|---|---:|
| Best validation loss | epoch 17, `1.0625020856350924` |
| Best validation accuracy | epoch 31, `0.6319308999721371` |
| Best validation macro-F1 | epoch 26, `0.601166548701511` |
| Best-epoch spread | `14` epochs |
| Clean-train macro-F1 at epoch 26 | `0.7562805286580438` |
| Primary train-validation macro-F1 gap | `15.511397995653287` pp |
| Final persisted clean-train macro-F1 | epoch 31, `0.8207655611897143` |
| Final persisted validation macro-F1 | epoch 31, `0.5938407974340496` |
| Final persisted train-validation gap | epoch 31, `22.69247637556647` pp |

The audit output is `GENERALIZATION_GAP_SIGNAL` for learning behavior and
`MATERIAL_POLICY_DRIFT` for monitor policy. The configured checkpoint and final
model-selection metric are `val_accuracy`; scheduler and early stopping monitor
`val_loss`; the configured final-test checkpoint field is
`best_val_accuracy`, but no final checkpoint was loaded for test evaluation.

Persisted learning-rate reductions:

- Epoch 23: `0.0003000000142492354` to `0.0001500000071246177`.
- Epoch 29: `0.0001500000071246177` to `0.00007500000356230885`.

Frozen prior-corruption schedule:

- Epochs 1–10: probability `0.1`.
- Epochs 11–30: probability `0.2`.
- Epoch 31 onward: probability `0.3`.

## Console-only epoch 32 and stopping inference

The following values are not in `history.json`, are not a saved checkpoint, and
must not be merged into the persisted audit table:

| Console-only measurement | Value |
|---|---:|
| Epoch-32 validation loss | `1.1512` |
| Epoch-32 validation accuracy | `63.58%` |
| Epoch-32 validation macro-F1 | `59.97%` |
| Epoch-32 clean-train metrics | `UNKNOWN` |

Persisted epoch 31 has `early_stopping_wait=14/15`. The frozen implementation
increments the wait when validation loss does not strictly improve and requests
stop when epoch is at least 30 and wait is at least 15. Since epoch-32
`val_loss=1.1512` is greater than the best persisted `1.0625020856350924` at
epoch 17, it would increment the wait to 15 and request stop. This is a
deterministic policy inference only; no epoch-32 row or completion marker was
written.

Epoch-32 validation macro-F1 (`59.97%`) is below the persisted epoch-26 maximum
(`60.1166548701511%`). Therefore the preregistered primary best-macro epoch and
its train-validation gap cannot change at the inferred terminal epoch.

## Diagnostic outcome

- Primary label: `GENERALIZATION_GAP_SIGNAL`.
- Preregistered hypothesis mapping: `H-GEN`.
- Independent policy label: `MATERIAL_POLICY_DRIFT` (`H-POLICY`).
- Evidence decision: retain the run as **censored but decision-sufficient for
  the primary diagnostic**; do not call it a completed baseline and do not rerun
  solely to recover non-primary artifacts.

The result supports treating train-validation separation as an important
immediate suspect and records that the configured monitors select materially
different epochs. It does not identify a causal mechanism, prove overfitting,
validate a model change, or authorize changes to architecture, graph, priors,
features, loss, optimizer, scheduler, early stopping, checkpoint policy, seed,
or data split. Any next intervention requires a separately preregistered
research-lead decision.
