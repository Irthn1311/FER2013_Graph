#!/usr/bin/env bash
set -euo pipefail

RUN_KEY="${1:-}"

case "${RUN_KEY}" in
  ce_continue)
    CONFIG="configs/d13c/d13c_m16_ce_continue.yaml"
    RUN_NAME="d13c_m16_ce_continue"
    ;;
  supcon_l001)
    CONFIG="configs/d13c/d13c_m16_supcon_l001.yaml"
    RUN_NAME="d13c_m16_supcon_l001"
    ;;
  supcon_l002)
    CONFIG="configs/d13c/d13c_m16_supcon_l002.yaml"
    RUN_NAME="d13c_m16_supcon_l002"
    ;;
  supcon_l005)
    CONFIG="configs/d13c/d13c_m16_supcon_l005.yaml"
    RUN_NAME="d13c_m16_supcon_l005"
    ;;
  supcon_l010)
    CONFIG="configs/d13c/d13c_m16_supcon_l010.yaml"
    RUN_NAME="d13c_m16_supcon_l010"
    ;;
  supcon_l002_freeze)
    CONFIG="configs/d13c/d13c_m16_supcon_l002_freeze_backbone.yaml"
    RUN_NAME="d13c_m16_supcon_l002_freeze_backbone"
    ;;
  supcon_l002_proj128)
    CONFIG="configs/d13c/d13c_m16_supcon_l002_proj128.yaml"
    RUN_NAME="d13c_m16_supcon_l002_proj128"
    ;;
  m8_supcon_l002_control)
    CONFIG="configs/d13c/d13c_m8_supcon_l002_control.yaml"
    RUN_NAME="d13c_m8_supcon_l002_control"
    ;;
  *)
    echo "Unknown D13C run_key: ${RUN_KEY}" >&2
    echo "Supported: ce_continue supcon_l001 supcon_l002 supcon_l005 supcon_l010 supcon_l002_freeze supcon_l002_proj128 m8_supcon_l002_control" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="/kaggle/working/outputs/d13c_diagnostic/${RUN_NAME}"

echo "[D13C] diagnostic only"
echo "[D13C] no full D13C, no full SupCon, no prototype, no motif-level SupCon, no motif claim"
echo "[D13C] run_key=${RUN_KEY}"
echo "[D13C] config=${CONFIG}"
echo "[D13C] output_dir=${OUTPUT_DIR}"

python training/train_d13c.py \
  --config "${CONFIG}" \
  --environment kaggle \
  --output_dir "${OUTPUT_DIR}"

python scripts/check_d13c_diagnostic_run.py --output_dir "${OUTPUT_DIR}"

ZIP_PATH="/kaggle/working/${RUN_NAME}_outputs.zip"
rm -f "${ZIP_PATH}"
cd /kaggle/working
zip -r "${ZIP_PATH}" "outputs/d13c_diagnostic/${RUN_NAME}"
echo "[D13C] wrote ${ZIP_PATH}"
