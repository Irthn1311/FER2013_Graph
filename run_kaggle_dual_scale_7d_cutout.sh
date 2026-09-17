#!/usr/bin/env bash
set -e

echo "============================================================"
echo "🚀 [RUNNER] FER2013 Dual-Scale GNN with 7D Features + Random Cutout"
echo "           - 7D Pixel Features: [I, x, y, gx, gy, grad_mag, laplacian]"
echo "           - Random Cutout (Erasing) Augmentation: p=0.3, size=[8, 14]"
echo "           - Multi-Head Motif Attention (4 Heads, K=36 Motifs)"
echo "           - Test-Time Augmentation (TTA: Original + Horizontal Flip)"
echo "============================================================"

CONFIG="${1:-standalone/pixel_gnn/configs/fer2013_pixel_motif_dual_scale_7d_cutout_fast_seed42.yaml}"

if [ -d "/kaggle/working" ]; then
    DEFAULT_OUTPUT="/kaggle/working/outputs/pixel_motif_dual_scale_7d_cutout"
else
    DEFAULT_OUTPUT="outputs/pixel_motif_dual_scale_7d_cutout"
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

# STEP 2: Full Training with 7D Features, Cutout & TTA
echo ""
echo "🏋️ [STEP 2/3] Launching Training + Validation + TTA Test Pipeline..."
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
echo "🎉 [COMPLETED] 7D Features + Cutout Dual-Scale GNN Pipeline finished!"
echo "📦 Model Checkpoints: $OUTPUT_DIR"
echo "🖼️ Graph Visualizations: $VIS_DIR"
echo "============================================================"
