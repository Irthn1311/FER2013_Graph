$ErrorActionPreference = "Stop"
if (-not $env:FER_CSV) { throw "Set FER_CSV" }
if (-not $env:PRIOR_ROOT) { throw "Set PRIOR_ROOT" }
if (-not $env:OUTPUT_ROOT) { throw "Set OUTPUT_ROOT" }
python -B -m lap_gnn_tf.cli.train `
  --config configs/fer2013_ofix7_mid_tensorflow_seed42.yaml `
  --fer-csv $env:FER_CSV `
  --prior-root $env:PRIOR_ROOT `
  --output-root $env:OUTPUT_ROOT `
  --device gpu `
  --graph-workers 2 `
  --batch-size 16 `
  --mixed-precision `
  --no-xla `
  --no-resume
