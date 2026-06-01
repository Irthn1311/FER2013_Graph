#!/usr/bin/env bash
set -e

echo "[A5b-heavy-opt] Python/PyTorch/CUDA"
python - <<'PY'
import sys
import torch
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY

CONFIG="configs/d16/main_branch/d16r_a5b_edge_context_gnn_a4_ce_seed42.yaml"
PRIOR_DIR="outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
PROFILE_OUT="outputs/d16_analysis/main_branch/a5b_heavy_profile"
BENCH_OUT="outputs/d16_analysis/main_branch/a5b_heavy_optimized_benchmark"

test -f "$CONFIG"
test -d "$PRIOR_DIR"

python d16/scripts/profile_d16_a5b_heavy_step.py \
  --config "$CONFIG" \
  --prior_dir "$PRIOR_DIR" \
  --output_dir "$PROFILE_OUT" \
  --num_warmup_batches 5 \
  --num_profile_batches 20 \
  --num_workers 2

python d16/scripts/benchmark_d16_a5b_heavy_optimized.py \
  --config "$CONFIG" \
  --prior_dir "$PRIOR_DIR" \
  --output_dir "$BENCH_OUT" \
  --num_warmup_batches 5 \
  --num_benchmark_batches 30 \
  --num_workers 2

echo "[A5b-heavy-opt] Benchmark complete. Do not full train unless decision permits it:"
cat "$BENCH_OUT/A5B_HEAVY_OPTIMIZED_BENCHMARK.md"
