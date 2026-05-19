#!/usr/bin/env bash
set -euo pipefail

run_key="${1:-}"

case "${run_key}" in
  k64)
    config="configs/d13/ablations/d13a_edgeaware_lite_localpool_k64.yaml"
    run_name="d13a_edgeaware_lite_localpool_k64"
    ;;
  k256)
    config="configs/d13/ablations/d13a_edgeaware_lite_localpool_k256.yaml"
    run_name="d13a_edgeaware_lite_localpool_k256"
    ;;
  temp07)
    config="configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_temp07.yaml"
    run_name="d13a_edgeaware_lite_localpool_k144_temp07"
    ;;
  temp05)
    config="configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_temp05.yaml"
    run_name="d13a_edgeaware_lite_localpool_k144_temp05"
    ;;
  anneal_1to05)
    config="configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_anneal_1to05.yaml"
    run_name="d13a_edgeaware_lite_localpool_k144_anneal_1to05"
    ;;
  no_aux)
    config="configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_no_aux.yaml"
    run_name="d13a_edgeaware_lite_localpool_k144_no_aux"
    ;;
  compact_balance_x2)
    config="configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_compact_balance_x2.yaml"
    run_name="d13a_edgeaware_lite_localpool_k144_compact_balance_x2"
    ;;
  seed2)
    config="configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_seed2.yaml"
    run_name="d13a_edgeaware_lite_localpool_k144_seed2"
    ;;
  seed3)
    config="configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_seed3.yaml"
    run_name="d13a_edgeaware_lite_localpool_k144_seed3"
    ;;
  lr1e4)
    config="configs/d13/ablations/d13a_edgeaware_lite_localpool_k144_lr1e4.yaml"
    run_name="d13a_edgeaware_lite_localpool_k144_lr1e4"
    ;;
  baseline)
    config="configs/d13/d13a_edgeaware_lite_localpool_k144.yaml"
    run_name="d13a_edgeaware_lite_localpool_k144"
    ;;
  gine)
    config="configs/d13/d13a_gine_localpool_k144.yaml"
    run_name="d13a_gine_localpool_k144"
    ;;
  *)
    echo "Usage: bash kaggle/d13/run_d13a_ablation_kaggle.sh <run_key>" >&2
    echo "Valid run_key: k64 k256 temp07 temp05 anneal_1to05 no_aux compact_balance_x2 seed2 seed3 lr1e4 baseline gine" >&2
    exit 2
    ;;
esac

output_dir="outputs/d13_hierarchical_reduction/ablations/${run_name}"
zip_path="${run_name}_outputs.zip"

echo "[D13A ablation] run_key=${run_key}"
echo "[D13A ablation] config=${config}"
echo "[D13A ablation] output_dir=${output_dir}"

python training/train_d13.py \
  --config "${config}" \
  --output_dir "${output_dir}" \
  --environment kaggle

check_status=0
python scripts/check_d13_debug_run.py --output_dir "${output_dir}" || check_status=$?
if [[ "${check_status}" -ne 0 && "${check_status}" -ne 2 ]]; then
  echo "[D13A ablation] checker failed unexpectedly with code ${check_status}" >&2
  exit "${check_status}"
fi

rm -f "${zip_path}"
zip -r "${zip_path}" "${output_dir}"

echo "[D13A ablation] output_dir=${output_dir}"
echo "[D13A ablation] zip_path=${zip_path}"
echo "[D13A ablation] best_checkpoint_exists=$([[ -f "${output_dir}/checkpoints/best.pt" ]] && echo true || echo false)"
if [[ -f "${output_dir}/d13_debug_check_summary.json" ]]; then
  python - <<PY
import json
from pathlib import Path
summary = json.loads(Path("${output_dir}/d13_debug_check_summary.json").read_text(encoding="utf-8"))
print("[D13A ablation] checker_decision=" + str(summary.get("final_decision")))
PY
fi

