#!/usr/bin/env bash
set -e

echo "============================================================"
echo "🚀 [RUNNER] FER2013 Hierarchical Pixel-to-Motif Graph Pipeline"
echo "============================================================"

# Config file (default: pixel_motif_graph_aug_fast)
CONFIG="${1:-standalone/pixel_gnn/configs/fer2013_pixel_motif_graph_aug_fast_seed42.yaml}"

# Output directory: use /kaggle/working if on Kaggle, otherwise ./outputs
if [ -d "/kaggle/working" ]; then
    DEFAULT_OUTPUT="/kaggle/working/outputs/pixel_motif_graph_aug"
else
    DEFAULT_OUTPUT="outputs/pixel_motif_graph_aug"
fi
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT}"
VIS_DIR="$OUTPUT_DIR/visualizations"

echo "📋 Config: $CONFIG"
echo "📂 Output: $OUTPUT_DIR"
echo "🎨 Visualizations: $VIS_DIR"

# STEP 1: Quick smoke verification (forward + backward + non-NaN + shape checks)
echo ""
echo "🔍 [STEP 1/3] Running quick smoke verification..."
python train.py --config "$CONFIG" --smoke --synthetic

# STEP 2: Full Training + Validation per epoch + Final Test evaluation
echo ""
echo "🏋️ [STEP 2/3] Launching Training + Validation + Test Pipeline..."
python train.py --config "$CONFIG" --output-root "$OUTPUT_DIR" "${@:3}"

# STEP 3: Test Set Graph Visualizations (Original, Pixel Graph, Motif Map, Overlay, Motif Graph, Attention)
echo ""
echo "🎨 [STEP 3/3] Generating Test Set Graph Visualizations..."
python standalone/pixel_gnn/visualize_test_graph.py \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIR/best_val_accuracy.weights.h5" \
    --output-dir "$VIS_DIR" \
    --num-samples 7

echo ""
echo "============================================================"
echo "🎉 [COMPLETED] Full Pipeline finished successfully from start to finish!"
echo "📦 Model Checkpoints & Metrics: $OUTPUT_DIR"
echo "🖼️ Graph Visualizations:        $VIS_DIR"
echo "============================================================"
