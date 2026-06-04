#!/usr/bin/env bash
set -e

echo "[A6-2b] Python/PyTorch/CUDA info"
python - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda_device", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_info_error", repr(exc))
PY

PRIOR_DIR="outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
CONFIG="configs/d16/main_branch/d16r_a6_2b_pairwise_relation_a5b_ce_seed42_accmon_150.yaml"
OUTPUT_DIR="outputs/d16_runs/main_branch/d16r_a6_2b_pairwise_relation_a5b_ce_seed42_accmon_150"
CHECK_DIR="outputs/d16_analysis/main_branch/a6_2b_pairwise_relation_check"
SMOKE_DIR="outputs/d16_smoke/main_branch/d16r_a6_2b_pairwise_relation_a5b_ce_seed42_accmon_150"

test -d "$PRIOR_DIR"
test -f "$CONFIG"

python d16/scripts/check_d16_a6_2b_pairwise_relation.py \
  --config "$CONFIG" \
  --prior_dir "$PRIOR_DIR" \
  --output_dir "$CHECK_DIR"

if [ "${RUN_SMOKE:-0}" = "1" ]; then
  python d16/training/train_d16.py \
    --config "$CONFIG" \
    --prior_dir "$PRIOR_DIR" \
    --output_dir "$SMOKE_DIR" \
    --max_epochs 1 \
    --limit_train_batches 2 \
    --limit_val_batches 2
fi

RESUME_ARGS=""
if [ -f "$OUTPUT_DIR/checkpoints/last.pt" ]; then
  RESUME_ARGS="--resume_auto true --resume_strict true"
fi

python d16/training/train_d16.py \
  --config "$CONFIG" \
  --prior_dir "$PRIOR_DIR" \
  --output_dir "$OUTPUT_DIR" \
  $RESUME_ARGS

python d16/scripts/check_d16_main_branch_run.py \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR"

python d16/scripts/collect_d16_a6_2b_results.py \
  --run_dir "$OUTPUT_DIR" \
  --output_dir outputs/d16_analysis/main_branch
