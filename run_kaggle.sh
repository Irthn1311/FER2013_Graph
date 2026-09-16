#!/usr/bin/env bash
set -e

echo "============================================================"
echo "🚀 [RUNNER] FER2013 Pixel Architecture Pipeline"
echo "============================================================"

# Config file (default: pixel_neighbor_motif)
CONFIG="${1:-standalone/pixel_gnn/configs/fer2013_pixel_neighbor_motif_fast_seed42.yaml}"

# Output directory: use /kaggle/working if on Kaggle, otherwise ./outputs
if [ -d "/kaggle/working" ]; then
    DEFAULT_OUTPUT="/kaggle/working/outputs/pixel_neighbor_motif"
else
    DEFAULT_OUTPUT="outputs/pixel_neighbor_motif"
fi
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT}"

echo "📋 Config: $CONFIG"
echo "📂 Output: $OUTPUT_DIR"

# Step 1: Quick smoke test
echo ""
echo "🔍 [STEP 1/2] Running quick smoke verification..."
python train.py --config "$CONFIG" --smoke --synthetic

# Step 2: Full Training + Val per epoch + Final Test evaluation
echo ""
echo "🏋️ [STEP 2/2] Launching Training + Validation + Test Pipeline..."
python train.py --config "$CONFIG" --output-root "$OUTPUT_DIR" "${@:3}"

echo ""
echo "============================================================"
echo "🎉 [COMPLETED] Pipeline finished successfully!"
echo "Outputs saved in: $OUTPUT_DIR"
echo "============================================================"
