#!/usr/bin/env bash
set -e

echo "============================================================================"
echo "🚀 [RUNNER] FER2013 Pure Pixel Adaptive Multi-Scale Motif GNN"
echo "           - 9D Features: [I, x, y, gx, gy, grad_mag, laplacian, local_var, lbp]"
echo "           - 2-Tier Hierarchy: 2,304 Pixels -> 49 Micro Motifs -> 9 Macro Landmarks"
echo "           - Adaptive Anisotropic Covariance Spreads (sigma_x, sigma_y)"
echo "           - InfoNCE Motif Contrastive Loss & Expression Regularization"
echo "============================================================================"

CONFIG="${1:-standalone/pixel_gnn/configs/fer2013_pixel_motif_adaptive_multiscale_9d_seed42.yaml}"

if [ -d "/kaggle/working" ]; then
    DEFAULT_OUTPUT="/kaggle/working/outputs/pixel_motif_adaptive_multiscale_180ep"
else
    DEFAULT_OUTPUT="outputs/pixel_motif_adaptive_multiscale_180ep"
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

# STEP 2: Full Training
echo ""
echo "🏋️ [STEP 2/3] Launching Pure Pixel Adaptive Multi-Scale Training..."
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
echo "============================================================================"
echo "🎉 [COMPLETED] Pure Pixel Adaptive Multi-Scale Pipeline finished!"
echo "📦 Model Checkpoints: $OUTPUT_DIR"
echo "🖼️ Graph Visualizations: $VIS_DIR"
echo "============================================================================"
