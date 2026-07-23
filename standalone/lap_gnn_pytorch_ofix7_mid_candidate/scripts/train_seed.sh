#!/usr/bin/env bash
set -euo pipefail
seed="$1"; fer_csv="$2"; prior_root="$3"; output_root="$4"; device="${5:-cuda:0}"; workers="${6:-2}"
python -m lap_gnn.cli.train --config "configs/fer2013_ofix7_mid_seed${seed}.yaml" --fer-csv "$fer_csv" --prior-root "$prior_root" --output-root "$output_root" --device "$device" --num-workers "$workers" --no-resume
