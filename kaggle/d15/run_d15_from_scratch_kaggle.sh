#!/usr/bin/env bash
set -euo pipefail

RUN_KEY="${1:-}"

case "${RUN_KEY}" in
  m8_basic)
    CONFIG="configs/d15/d15_m8_scratch_basic.yaml"
    RUN_NAME="d15_m8_scratch_basic"
    ;;
  m16_basic)
    CONFIG="configs/d15/d15_m16_scratch_basic.yaml"
    RUN_NAME="d15_m16_scratch_basic"
    ;;
  m8_curriculum)
    CONFIG="configs/d15/d15_m8_scratch_curriculum.yaml"
    RUN_NAME="d15_m8_scratch_curriculum"
    ;;
  m16_curriculum)
    CONFIG="configs/d15/d15_m16_scratch_curriculum.yaml"
    RUN_NAME="d15_m16_scratch_curriculum"
    ;;
  m8_aug_strong)
    CONFIG="configs/d15/d15_m8_scratch_aug_strong.yaml"
    RUN_NAME="d15_m8_scratch_aug_strong"
    ;;
  m16_aug_strong)
    CONFIG="configs/d15/d15_m16_scratch_aug_strong.yaml"
    RUN_NAME="d15_m16_scratch_aug_strong"
    ;;
  m8_focal_class_weight)
    CONFIG="configs/d15/d15_m8_scratch_focal_class_weight.yaml"
    RUN_NAME="d15_m8_scratch_focal_class_weight"
    ;;
  m8_deeper_readout)
    CONFIG="configs/d15/d15_m8_scratch_deeper_readout.yaml"
    RUN_NAME="d15_m8_scratch_deeper_readout"
    ;;
  *)
    echo "Unknown D15 run_key: ${RUN_KEY}" >&2
    echo "Supported: m8_basic m16_basic m8_curriculum m16_curriculum m8_aug_strong m16_aug_strong m8_focal_class_weight m8_deeper_readout" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="/kaggle/working/outputs/d15_from_scratch/${RUN_NAME}"

echo "[D15] from-scratch performance-first main track"
echo "[D15] no D13/D14 checkpoint, no warm-start, no prototype, no motif/evidence/full-interpretability claim"
echo "[D15] run_key=${RUN_KEY}"
echo "[D15] config=${CONFIG}"
echo "[D15] output_dir=${OUTPUT_DIR}"

python training/train_d15.py \
  --config "${CONFIG}" \
  --environment kaggle \
  --output_dir "${OUTPUT_DIR}"

python scripts/check_d15_from_scratch_run.py --output_dir "${OUTPUT_DIR}"

ZIP_PATH="/kaggle/working/${RUN_NAME}_outputs.zip"
rm -f "${ZIP_PATH}"
cd /kaggle/working
zip -r "${ZIP_PATH}" "outputs/d15_from_scratch/${RUN_NAME}"
echo "[D15] wrote ${ZIP_PATH}"

