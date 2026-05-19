#!/usr/bin/env bash
set -euo pipefail

run_key="${1:-}"

case "${run_key}" in
  extended_baseline_k144)
    config="configs/d13/extended/d13a_edgeaware_lite_localpool_k144_ep100.yaml"
    source_run="d13a_edgeaware_lite_localpool_k144_outputs"
    ;;
  extended_k256)
    config="configs/d13/extended/d13a_edgeaware_lite_localpool_k256_ep100.yaml"
    source_run="d13a_edgeaware_lite_localpool_k256_outputs"
    ;;
  extended_seed3)
    config="configs/d13/extended/d13a_edgeaware_lite_localpool_k144_seed3_ep100.yaml"
    source_run="d13a_edgeaware_lite_localpool_k144_seed3_outputs"
    ;;
  extended_no_aux)
    config="configs/d13/extended/d13a_edgeaware_lite_localpool_k144_no_aux_ep100.yaml"
    source_run="d13a_edgeaware_lite_localpool_k144_no_aux_outputs"
    ;;
  extended_anneal_1to05)
    config="configs/d13/extended/d13a_edgeaware_lite_localpool_k144_anneal_1to05_ep100.yaml"
    source_run="d13a_edgeaware_lite_localpool_k144_anneal_1to05_outputs"
    ;;
  extended_compact_balance_x2)
    config="configs/d13/extended/d13a_edgeaware_lite_localpool_k144_compact_balance_x2_ep100.yaml"
    source_run="d13a_edgeaware_lite_localpool_k144_compact_balance_x2_outputs"
    ;;
  *)
    echo "Usage: bash kaggle/d13/run_d13a_extended_kaggle.sh <run_key>" >&2
    echo "Valid run_key: extended_baseline_k144 extended_k256 extended_seed3 extended_no_aux extended_anneal_1to05 extended_compact_balance_x2" >&2
    exit 2
    ;;
esac

source_dir="outputs/d13_hierarchical_reduction/${source_run}"
resume_checkpoint="${source_dir}/checkpoints/last.pt"
output_dir="outputs/d13_hierarchical_reduction/extended/${source_run}_ep100"
zip_path="${source_run}_ep100_outputs.zip"

echo "[D13A extended] run_key=${run_key}"
echo "[D13A extended] config=${config}"
echo "[D13A extended] source_dir=${source_dir}"
echo "[D13A extended] resume_checkpoint=${resume_checkpoint}"
echo "[D13A extended] output_dir=${output_dir}"

if [[ ! -f "${resume_checkpoint}" ]]; then
  echo "[D13A extended] missing resume checkpoint: ${resume_checkpoint}" >&2
  echo "[D13A extended] Make sure the 50-epoch output is available before launching the extension." >&2
  exit 3
fi

python training/train_d13.py \
  --config "${config}" \
  --output_dir "${output_dir}" \
  --resume_checkpoint "${resume_checkpoint}" \
  --environment kaggle

check_status=0
python scripts/check_d13_debug_run.py --output_dir "${output_dir}" || check_status=$?
if [[ "${check_status}" -ne 0 && "${check_status}" -ne 2 ]]; then
  echo "[D13A extended] checker failed unexpectedly with code ${check_status}" >&2
  exit "${check_status}"
fi

rm -f "${zip_path}"
zip -r "${zip_path}" "${output_dir}"

echo "[D13A extended] output_dir=${output_dir}"
echo "[D13A extended] zip_path=${zip_path}"
echo "[D13A extended] best_checkpoint_exists=$([[ -f "${output_dir}/checkpoints/best.pt" ]] && echo true || echo false)"
if [[ -f "${output_dir}/d13_debug_check_summary.json" ]]; then
  python - <<PY
import json
from pathlib import Path
summary = json.loads(Path("${output_dir}/d13_debug_check_summary.json").read_text(encoding="utf-8"))
print("[D13A extended] checker_decision=" + str(summary.get("final_decision")))
PY
fi
