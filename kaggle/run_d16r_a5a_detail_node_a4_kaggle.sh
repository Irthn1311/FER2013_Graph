#!/usr/bin/env bash
set -e

CONFIG_PATH="configs/d16/main_branch/d16r_a5a_detail_node_a4_ce_seed42.yaml"
PRIOR_DIR="outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
OUTPUT_DIR="outputs/d16_runs/main_branch/d16r_a5a_detail_node_a4_ce_seed42"
CHECK_DIR="outputs/d16_analysis/main_branch/d16r_a5a_detail_node_feature_check"
RUN_CHECK_DIR="outputs/d16_analysis/main_branch/d16r_a5a_detail_node_a4_ce_seed42_check"
ANALYSIS_DIR="outputs/d16_analysis/main_branch"

echo "[D16R-A5a] Python/PyTorch/CUDA environment"
python - <<'PY'
import json
import platform
import sys

try:
    import torch
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    info = {"python": sys.version, "platform": platform.platform(), "torch_error": str(exc)}
print(json.dumps(info, indent=2))
PY

if [ ! -d "${PRIOR_DIR}" ]; then
  echo "[D16R-A5a] Missing prior dir: ${PRIOR_DIR}" >&2
  exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "[D16R-A5a] Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

python d16/scripts/check_d16_detail_node_features.py \
  --config "${CONFIG_PATH}" \
  --prior_dir "${PRIOR_DIR}" \
  --output_dir "${CHECK_DIR}"

if [ "${RUN_SMOKE:-0}" = "1" ]; then
  SMOKE_DIR="outputs/d16_smoke/main_branch/d16r_a5a_detail_node_a4_ce_seed42"
  echo "[D16R-A5a] RUN_SMOKE=1, running capped smoke first"
  python d16/training/train_d16.py \
    --config "${CONFIG_PATH}" \
    --prior_dir "${PRIOR_DIR}" \
    --output_dir "${SMOKE_DIR}" \
    --max_epochs 1 \
    --limit_train_batches 2 \
    --limit_val_batches 2
fi

python d16/training/train_d16.py \
  --config "${CONFIG_PATH}" \
  --prior_dir "${PRIOR_DIR}" \
  --output_dir "${OUTPUT_DIR}"

for required in \
  checkpoints/best.pt \
  checkpoints/last.pt \
  test_metrics.csv \
  last_test_metrics.csv \
  per_class_metrics.csv \
  detected_vs_fallback_metrics.csv \
  detected_fallback_per_class_metrics.csv \
  pred_count.csv \
  confusion_matrix.csv \
  predictions.csv \
  d16_train_summary.json
do
  if [ ! -e "${OUTPUT_DIR}/${required}" ]; then
    echo "[D16R-A5a] Missing required artifact: ${OUTPUT_DIR}/${required}" >&2
    exit 1
  fi
done

python d16/scripts/check_d16_main_branch_run.py \
  --run_dir "${OUTPUT_DIR}" \
  --output_dir "${RUN_CHECK_DIR}"

python d16/scripts/collect_d16_main_branch_results.py \
  --run_dirs "${OUTPUT_DIR}" \
  --output_dir "${ANALYSIS_DIR}"
