#!/usr/bin/env bash
set -euo pipefail
python -B tools/verify_no_torch_runtime.py
python -B tools/verify_no_parent_imports.py
python -B tools/verify_checksums.py
CUDA_VISIBLE_DEVICES="" python -B -m pytest -q
CUDA_VISIBLE_DEVICES="" python -B -m lap_gnn_tf.cli.compare_golden
