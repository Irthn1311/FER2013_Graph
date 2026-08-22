# LAP-GNN TensorFlow OFIX7-mid Candidate

This package is a self-contained TensorFlow/Keras port of the locked PyTorch
OFIX7-mid standalone candidate. It consumes the same FER2013 CSV and verified
`d16_mediapipe_pixel_priors_v1` files. It never regenerates priors and normal
runtime imports neither PyTorch nor parent D16-D19 code.

The active model is fixed at 37 node channels, 8 edge channels, hidden size 96,
three mean-aggregation edge-context layers, five semantic message-passing
anchors, the 20-token micro-motif readout, and 7 classes. Trainable parameters:
1,061,192.

```bash
python -m lap_gnn_tf.cli.inspect_environment
python -m lap_gnn_tf.cli.compare_golden --package-root .
python -m lap_gnn_tf.cli.validate --config configs/fer2013_ofix7_mid_tensorflow_seed42.yaml --golden
```

Full training is intentionally not run as part of the port. The Kaggle notebook
contains an explicit `RUN_FULL_TRAINING` gate.

## Read-only learning-history audit

Issue #1's diagnostic reads only `history.json` and `resolved_config.json` from
an existing TensorFlow run. It does not import TensorFlow, load model weights,
run inference/training, or read test artifacts. The output directory is
required, must be empty or absent, and must be outside the source run.

```bash
python tools/audit_learning_history.py \
  --run-dir /path/to/existing/tensorflow-run \
  --output-dir /path/to/new/audit-output
```

The command writes deterministic `learning_diagnosis.json`,
`learning_diagnosis.md`, and `epoch_metrics_validation_only.csv` files. The
JSON separates provenance, raw measurements, configured monitor policy,
heuristic thresholds, and bounded interpretation. Missing train macro-F1 at
the best validation macro-F1 epoch yields
`UNKNOWN_TRAIN_EVAL_INCOMPLETE`; it is never approximated. Gap thresholds are
diagnostic conveniences only and are not evidence of overfitting, a basis for
changing training, or a final-model selection rule.

## Validation-only training execution

`tools/train_validation_only.py` calls the frozen trainer unchanged, but pins
the reviewed trainer SHA-256 and intercepts its first post-training boundary.
It stops after history and telemetry are written and before final-test
checkpoint loading, test-data construction, or inference. Normal
`lap-gnn-tf-train` behavior is unchanged.

```bash
python tools/train_validation_only.py \
  --config configs/fer2013_ofix7_mid_tensorflow_optimized_seed42.yaml \
  --fer-csv /path/to/fer2013.csv \
  --prior-root /path/to/d16_mediapipe_pixel_priors_v1 \
  --output-root /path/to/fresh/validation-only-output \
  --device gpu \
  --no-resume
```

On success the output contains `VALIDATION_ONLY_COMPLETE.json` with trainer,
config, history, and scientific-payload provenance plus explicit
`test_accessed: false`, `test_data_constructed: false`, and
`test_checkpoint_loaded: false` fields. The wrapper refuses trainer or payload
drift and never falls back to normal final-test behavior.

The optional `--limit-epochs`, `--limit-train-batches`,
`--limit-val-batches`, and `--limit-train-eval-batches` arguments are for
bounded smoke verification only. Do not use them for the later registered
baseline. This Issue adds and tests the path only; it does not launch that
baseline.
