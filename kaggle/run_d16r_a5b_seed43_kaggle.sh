#!/usr/bin/env bash
set -e

echo "[A5b-seed43] Python/PyTorch/CUDA"
python - <<'PY'
import sys
import torch
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY

CONFIG="configs/d16/main_branch/d16r_a5b_heavy_opt_a4_ce_seed43_accmon_150.yaml"
PRIOR_DIR="outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
OUTPUT_DIR="outputs/d16_runs/main_branch/d16r_a5b_heavy_opt_a4_ce_seed43_accmon_150"
CHECK_DIR="outputs/d16_analysis/main_branch/d16r_a5b_heavy_opt_a4_ce_seed43_accmon_150_edge_context_gnn_check"
LAST_CKPT="${OUTPUT_DIR}/checkpoints/last.pt"

test -f "$CONFIG"
test -d "$PRIOR_DIR"
grep -q -- "--resume_from" d16/training/train_d16.py
grep -q -- "--resume_strict" d16/training/train_d16.py
mkdir -p "$OUTPUT_DIR"

if [[ -f "d16/scripts/check_d16_edge_context_gnn.py" ]]; then
  python d16/scripts/check_d16_edge_context_gnn.py \
    --config "$CONFIG" \
    --prior_dir "$PRIOR_DIR" \
    --output_dir "$CHECK_DIR"
fi

RESUME_ARGS=()
if [[ -f "$LAST_CKPT" ]]; then
  echo "[A5b-seed43] Auto-resuming from ${LAST_CKPT}"
  RESUME_ARGS=(--resume_from "$LAST_CKPT" --resume_strict true)
else
  echo "[A5b-seed43] No last.pt found; starting from scratch"
fi

python d16/training/train_d16.py \
  --config "$CONFIG" \
  --prior_dir "$PRIOR_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "${RESUME_ARGS[@]}"

for artifact in \
  checkpoints/best.pt \
  checkpoints/last.pt \
  test_metrics.csv \
  last_test_metrics.csv \
  per_class_metrics.csv \
  detected_vs_fallback_metrics.csv \
  detected_fallback_per_class_metrics.csv \
  confusion_matrix.csv \
  predictions.csv \
  d16_train_summary.json; do
  test -f "${OUTPUT_DIR}/${artifact}"
done

echo "[A5b-seed43] Done: ${OUTPUT_DIR}"
