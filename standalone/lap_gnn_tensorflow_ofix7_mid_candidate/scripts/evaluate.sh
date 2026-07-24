#!/usr/bin/env bash
set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR}"
: "${FER_CSV:?Set FER_CSV}"
: "${PRIOR_ROOT:?Set PRIOR_ROOT}"
python -B -m lap_gnn_tf.cli.evaluate \
  --run-dir "$RUN_DIR" \
  --fer-csv "$FER_CSV" \
  --prior-root "$PRIOR_ROOT" \
  --checkpoint best_val_macro_f1 \
  --split test
