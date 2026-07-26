$ErrorActionPreference = "Stop"
python -B tools/verify_no_torch_runtime.py
python -B tools/verify_no_parent_imports.py
python -B tools/verify_checksums.py
$env:CUDA_VISIBLE_DEVICES = ""
python -B -m pytest -q
python -B -m lap_gnn_tf.cli.compare_golden
