#!/usr/bin/env bash
set -e

echo "[A5b-heavy-opt-150] Python/PyTorch/CUDA"
python - <<'PY'
import sys
import torch
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY

CONFIG="configs/d16/main_branch/d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150.yaml"
PRIOR_DIR="outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
OUTPUT_DIR="outputs/d16_runs/main_branch/d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150"
LAST_CKPT="${OUTPUT_DIR}/checkpoints/last.pt"

test -f "$CONFIG"
test -d "$PRIOR_DIR"
mkdir -p "$OUTPUT_DIR"

RESUME_ARGS=()
if [[ -f "$LAST_CKPT" ]]; then
  echo "[A5b-heavy-opt-150] Auto-resuming from ${LAST_CKPT}"
  RESUME_ARGS=(--resume_from "$LAST_CKPT" --resume_strict true)
else
  echo "[A5b-heavy-opt-150] No last.pt found; starting from scratch"
fi

python d16/training/train_d16.py \
  --config "$CONFIG" \
  --prior_dir "$PRIOR_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "${RESUME_ARGS[@]}"
