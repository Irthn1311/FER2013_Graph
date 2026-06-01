#!/usr/bin/env bash
set -e

CONFIG="configs/d16/main_branch/d16r_a5b_edge_context_gnn_a4_ce_seed42.yaml"
PRIOR_DIR="outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
OUTPUT_DIR="outputs/d16_runs/main_branch/d16r_a5b_edge_context_gnn_a4_ce_seed42"
BENCH_SUMMARY="outputs/d16_analysis/main_branch/a5b_heavy_optimized_benchmark/a5b_heavy_optimized_benchmark_summary.json"

if [ "${FORCE_A5B_HEAVY_FULL:-0}" != "1" ]; then
  python - <<'PY'
import json
from pathlib import Path
path = Path("outputs/d16_analysis/main_branch/a5b_heavy_optimized_benchmark/a5b_heavy_optimized_benchmark_summary.json")
if not path.exists():
    raise SystemExit("Missing benchmark summary. Run kaggle/run_d16r_a5b_heavy_opt_benchmark_kaggle.sh first.")
decision = json.loads(path.read_text()).get("decision")
if decision != "HEAVY_OPT_SPEED_OK_FULL_RUN":
    raise SystemExit(
        f"Refusing full A5b-heavy run because benchmark decision is {decision!r}. "
        "Set FORCE_A5B_HEAVY_FULL=1 only after explicit user approval."
    )
print("benchmark decision permits full run:", decision)
PY
fi

test -f "$CONFIG"
test -d "$PRIOR_DIR"

python d16/training/train_d16.py \
  --config "$CONFIG" \
  --prior_dir "$PRIOR_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --disable_graph_cache
