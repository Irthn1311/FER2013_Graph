#!/usr/bin/env bash
set -euo pipefail

variant="${1:-edgeaware}"

case "${variant}" in
  edgeaware|edgeaware_lite)
    config="configs/d13/d13a_edgeaware_lite_localpool_k144.yaml"
    output_dir="outputs/d13_hierarchical_reduction/d13a_edgeaware_lite_localpool_k144"
    zip_name="d13a_edgeaware_lite_localpool_k144_outputs.zip"
    ;;
  gine)
    config="configs/d13/d13a_gine_localpool_k144.yaml"
    output_dir="outputs/d13_hierarchical_reduction/d13a_gine_localpool_k144"
    zip_name="d13a_gine_localpool_k144_outputs.zip"
    ;;
  *)
    echo "Usage: bash kaggle/d13/run_d13a_kaggle.sh {edgeaware|gine}" >&2
    exit 2
    ;;
esac

echo "[D13A] variant=${variant}"
echo "[D13A] config=${config}"
echo "[D13A] output_dir=${output_dir}"

python training/train_d13.py \
  --config "${config}" \
  --output_dir "${output_dir}" \
  --environment kaggle

python scripts/check_d13_debug_run.py --output_dir "${output_dir}" || true

rm -f "${zip_name}"
zip -r "${zip_name}" "${output_dir}"
echo "[D13A] zipped ${zip_name}"

