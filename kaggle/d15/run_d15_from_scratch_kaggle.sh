#!/usr/bin/env bash
set -euo pipefail

RUN_KEY="${1:-}"
RESUME_FROM="${RESUME_FROM:-}"

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
if [[ -n "${RESUME_FROM}" ]]; then
  echo "[D15] resume_from=${RESUME_FROM}"
fi

TRAIN_ARGS=(
  training/train_d15.py
  --config "${CONFIG}" \
  --environment kaggle \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 32 \
  --num_workers 2 \
  --pin_memory true \
  --persistent_workers true \
  --prefetch_factor 2 \
  --chunk_cache_size 8 \
  --chunk_aware_sampler \
  --max_runtime_hours 11.2 \
  --save_before_exit_minutes 20
)

if [[ -n "${RESUME_FROM}" ]]; then
  TRAIN_ARGS+=(--resume_from "${RESUME_FROM}")
fi

python "${TRAIN_ARGS[@]}"

python scripts/check_d15_from_scratch_run.py --output_dir "${OUTPUT_DIR}"
python scripts/check_d15_resume_integrity.py --run_dir "${OUTPUT_DIR}"

ZIP_PATH="/kaggle/working/${RUN_NAME}_outputs.zip"
rm -f "${ZIP_PATH}"
cd /kaggle/working
zip -r "${ZIP_PATH}" "outputs/d15_from_scratch/${RUN_NAME}"
echo "[D15] wrote ${ZIP_PATH}"
