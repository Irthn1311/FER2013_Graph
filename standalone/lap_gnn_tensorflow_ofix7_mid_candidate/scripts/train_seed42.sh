#!/usr/bin/env bash
set -euo pipefail
: "${FER_CSV:?Set FER_CSV}"
: "${PRIOR_ROOT:?Set PRIOR_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
python -B -m lap_gnn_tf.cli.train \
  --config configs/fer2013_ofix7_mid_tensorflow_seed42.yaml \
  --fer-csv "$FER_CSV" \
  --prior-root "$PRIOR_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --device gpu \
  --graph-workers 2 \
  --batch-size 16 \
  --mixed-precision \
  --no-xla \
  --no-resume
