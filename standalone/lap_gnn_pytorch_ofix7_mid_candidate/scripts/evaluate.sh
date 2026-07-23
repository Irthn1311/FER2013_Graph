#!/usr/bin/env bash
set -euo pipefail
seed="$1"; checkpoint="$2"; fer_csv="$3"; prior_root="$4"; device="${5:-cuda:0}"
python -m lap_gnn.cli.evaluate --config "configs/fer2013_ofix7_mid_seed${seed}.yaml" --checkpoint "$checkpoint" --fer-csv "$fer_csv" --prior-root "$prior_root" --split test --device "$device" --num-workers 0
