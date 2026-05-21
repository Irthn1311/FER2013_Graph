#!/usr/bin/env bash
set -euo pipefail

RUN_KEY="${1:-}"

case "${RUN_KEY}" in
  m8_extend100)
    CONFIG="configs/d14/d14_m8_l002_extend100.yaml"
    RUN_NAME="d14_m8_l002_extend100"
    MODE="train"
    ;;
  m16_extend100)
    CONFIG="configs/d14/d14_m16_l005_extend100.yaml"
    RUN_NAME="d14_m16_l005_extend100"
    MODE="train"
    ;;
  m8_aug_strong)
    CONFIG="configs/d14/d14_m8_l002_aug_strong.yaml"
    RUN_NAME="d14_m8_l002_aug_strong"
    MODE="train"
    ;;
  m16_aug_strong)
    CONFIG="configs/d14/d14_m16_l005_aug_strong.yaml"
    RUN_NAME="d14_m16_l005_aug_strong"
    MODE="train"
    ;;
  m8_deeper_readout)
    CONFIG="configs/d14/d14_m8_deeper_readout.yaml"
    RUN_NAME="d14_m8_deeper_readout"
    MODE="train"
    ;;
  m16_deeper_region_readout)
    CONFIG="configs/d14/d14_m16_deeper_region_readout.yaml"
    RUN_NAME="d14_m16_deeper_region_readout"
    MODE="train"
    ;;
  m16_focal_class_weight)
    CONFIG="configs/d14/d14_m16_l005_focal_or_class_weight.yaml"
    RUN_NAME="d14_m16_l005_focal_or_class_weight"
    MODE="train"
    ;;
  ensemble_eval)
    CONFIG="configs/d14/d14_ensemble_eval_m8_m16_k256.yaml"
    RUN_NAME="d14_ensemble_eval_m8_m16_k256"
    MODE="ensemble"
    ;;
  *)
    echo "Unknown D14 run_key: ${RUN_KEY}" >&2
    echo "Supported: m8_extend100 m16_extend100 m8_aug_strong m16_aug_strong m8_deeper_readout m16_deeper_region_readout m16_focal_class_weight ensemble_eval" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="/kaggle/working/outputs/d14_performance/${RUN_NAME}"

echo "[D14] performance-first track"
echo "[D14] no prototype, no motif-level SupCon, no motif/evidence/full-interpretability claim"
echo "[D14] run_key=${RUN_KEY}"
echo "[D14] config=${CONFIG}"
echo "[D14] output_dir=${OUTPUT_DIR}"

if [[ "${MODE}" == "ensemble" ]]; then
  python scripts/evaluate_d14_ensemble.py \
    --configs \
      configs/d13c/d13c_m8_supcon_l002_control.yaml \
      configs/d13c/d13c_m16_supcon_l005.yaml \
      configs/d13c/d13c_m16_supcon_l002_proj128.yaml \
      configs/d13c/d13c_m16_supcon_l010.yaml \
    --checkpoints \
      outputs/d13c_diagnostic/d13c_m8_supcon_l002_control/checkpoints/best.pt \
      outputs/d13c_diagnostic/d13c_m16_supcon_l005/checkpoints/best.pt \
      outputs/d13c_diagnostic/d13c_m16_supcon_l002_proj128/checkpoints/best.pt \
      outputs/d13c_diagnostic/d13c_m16_supcon_l010/checkpoints/best.pt \
    --names \
      d13c_m8_supcon_l002_control \
      d13c_m16_supcon_l005 \
      d13c_m16_supcon_l002_proj128 \
      d13c_m16_supcon_l010 \
    --environment kaggle \
    --output_dir "${OUTPUT_DIR}"
else
  python training/train_d14.py \
    --config "${CONFIG}" \
    --environment kaggle \
    --output_dir "${OUTPUT_DIR}"

  python scripts/check_d14_performance_run.py --output_dir "${OUTPUT_DIR}"
fi

ZIP_PATH="/kaggle/working/${RUN_NAME}_outputs.zip"
rm -f "${ZIP_PATH}"
cd /kaggle/working
zip -r "${ZIP_PATH}" "outputs/d14_performance/${RUN_NAME}"
echo "[D14] wrote ${ZIP_PATH}"

