#!/usr/bin/env bash
set -euo pipefail

RUN_KEY="${1:-}"
if [[ -z "${RUN_KEY}" ]]; then
  echo "Usage: bash kaggle/d13b/run_d13b_diagnostic_kaggle.sh <run_key>" >&2
  exit 2
fi

case "${RUN_KEY}" in
  m8_basic)
    CONFIG="configs/d13b/d13b_k144_m8_basic.yaml"
    RUN_NAME="d13b_k144_m8_basic"
    ;;
  dp_amp_smoke)
    CONFIG="configs/d13b/d13b_k144_m8_dp_amp_smoke.yaml"
    RUN_NAME="d13b_k144_m8_dp_amp_smoke"
    ;;
  m16_basic)
    CONFIG="configs/d13b/d13b_k144_m16_basic.yaml"
    RUN_NAME="d13b_k144_m16_basic"
    ;;
  m8_no_slot_reg)
    CONFIG="configs/d13b/d13b_k144_m8_no_slot_reg.yaml"
    RUN_NAME="d13b_k144_m8_no_slot_reg"
    ;;
  m8_strong_slot_reg)
    CONFIG="configs/d13b/d13b_k144_m8_strong_slot_reg.yaml"
    RUN_NAME="d13b_k144_m8_strong_slot_reg"
    ;;
  m16_deep_readout)
    CONFIG="configs/d13b/d13b_k144_m16_deep_readout.yaml"
    RUN_NAME="d13b_k144_m16_deep_readout"
    ;;
  m8_deep_region)
    CONFIG="configs/d13b/d13b_k144_m8_deep_region.yaml"
    RUN_NAME="d13b_k144_m8_deep_region"
    ;;
  k256_m8_score_control)
    CONFIG="configs/d13b/d13b_k256_m8_score_control.yaml"
    RUN_NAME="d13b_k256_m8_score_control"
    ;;
  m8_seed2)
    CONFIG="configs/d13b/d13b_k144_m8_seed2.yaml"
    RUN_NAME="d13b_k144_m8_seed2"
    ;;
  *)
    echo "Unknown run_key: ${RUN_KEY}" >&2
    echo "Supported: m8_basic dp_amp_smoke m16_basic m8_no_slot_reg m8_strong_slot_reg m16_deep_readout m8_deep_region k256_m8_score_control m8_seed2" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="outputs/d13b_diagnostic/${RUN_NAME}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-FER-GRAPH-D13B}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" || "${USE_WANDB}" == "true" || "${USE_WANDB}" == "TRUE" ]]; then
  WANDB_ARGS+=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_run_name "${RUN_NAME}_$(date +%Y%m%d_%H%M%S)")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
else
  WANDB_ARGS+=(--no_wandb)
fi

echo "[D13B] run_key=${RUN_KEY}"
echo "[D13B] config=${CONFIG}"
echo "[D13B] output_dir=${OUTPUT_DIR}"
echo "[D13B] wandb=${USE_WANDB} project=${WANDB_PROJECT} entity=${WANDB_ENTITY:-default}"
echo "[D13B] diagnostic only: no SupCon, no D13C, no motif claim"

python training/train_d13b.py \
  --config "${CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  "${WANDB_ARGS[@]}"

python scripts/check_d13b_diagnostic_run.py --output_dir "${OUTPUT_DIR}"

zip -r "${RUN_NAME}_outputs.zip" "${OUTPUT_DIR}"

if [[ -f "${OUTPUT_DIR}/d13b_diagnostic_check_summary.json" ]]; then
  python - <<PY
import json
from pathlib import Path
p = Path("${OUTPUT_DIR}") / "d13b_diagnostic_check_summary.json"
data = json.loads(p.read_text())
print("[D13B] checker_decision=" + str(data.get("decision")))
PY
fi

echo "[D13B] zip=${RUN_NAME}_outputs.zip"
