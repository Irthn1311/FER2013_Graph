#!/usr/bin/env bash
set -e

echo "============================================================"
echo "🚀 [RUNNER] FER2013 Pixel-to-Motif Graph + DATA AUGMENTATION"
echo "============================================================"

CONFIG="${1:-standalone/pixel_gnn/configs/fer2013_pixel_motif_graph_aug_fast_seed42.yaml}"

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

# STEP 1: Quick smoke verification
echo ""
echo "🔍 [STEP 1/3] Running quick smoke verification..."
python train.py --config "$CONFIG" --smoke --synthetic

# STEP 2: Full Training with Data Augmentation + Validation + Final Test evaluation
echo ""
echo "🏋️ [STEP 2/3] Launching Training with Data Augmentation..."
python train.py --config "$CONFIG" --output-root "$OUTPUT_DIR" "${@:3}"

# STEP 3: Test Set Graph Visualizations
echo ""
echo "🎨 [STEP 3/3] Generating Test Set Graph Visualizations..."
python standalone/pixel_gnn/visualize_test_graph.py \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIR/best_val_accuracy.weights.h5" \
    --output-dir "$VIS_DIR" \
    --num-samples 7

echo ""
echo "============================================================"
echo "🎉 [COMPLETED] Full Augmented Pipeline finished successfully!"
echo "📦 Model Checkpoints & Metrics: $OUTPUT_DIR"
echo "🖼️ Graph Visualizations:        $VIS_DIR"
echo "============================================================"
