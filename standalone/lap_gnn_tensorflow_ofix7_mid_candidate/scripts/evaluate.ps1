$ErrorActionPreference = "Stop"
if (-not $env:RUN_DIR) { throw "Set RUN_DIR" }
if (-not $env:FER_CSV) { throw "Set FER_CSV" }
if (-not $env:PRIOR_ROOT) { throw "Set PRIOR_ROOT" }
python -B -m lap_gnn_tf.cli.evaluate `
  --run-dir $env:RUN_DIR `
  --fer-csv $env:FER_CSV `
  --prior-root $env:PRIOR_ROOT `
  --checkpoint best_val_macro_f1 `
  --split test
