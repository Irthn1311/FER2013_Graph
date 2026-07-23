#!/usr/bin/env bash
set -euo pipefail
python -m lap_gnn.cli.validate --config configs/fer2013_ofix7_mid_seed42.yaml --fer-csv "$1" --prior-root "$2" --device cpu --num-workers 0
