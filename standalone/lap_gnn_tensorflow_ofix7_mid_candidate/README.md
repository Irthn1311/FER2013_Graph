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
