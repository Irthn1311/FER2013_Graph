#!/usr/bin/env bash
set -e

echo "============================================================================"
echo "🔥 [RUNNER] FER2013 HEAVY Dual-Scale GNN (~3.2M Params, 200 Epochs)"
echo "           - 7D Features: [I, x, y, gx, gy, grad_mag, laplacian]"
echo "           - Model Width & Depth: Hidden Dim 192 (6 Heads), 4 Pixel Layers"
echo "           - Transformer FFN Blocks: 4x Expansion (192 -> 768 -> 192, GELU)"
echo "           - Dense Facial Motif Grid: K=49 Motifs (7x7), 3 Motif GNN Layers"
echo "           - Random Cutout Augmentation (p=0.35, size 8-16)"
echo "           - Top-5 Checkpoints for val_accuracy in checkpoints/best/"
echo "           - Top-5 Checkpoints for val_loss in checkpoints/best_loss/"
echo "           - Validation-Tuned TTA Sweep & Combinatorial Ensemble (FER2013_SGU Parity)"
echo "============================================================================"

CONFIG="${1:-standalone/pixel_gnn/configs/fer2013_pixel_motif_dual_scale_heavy_200ep_seed42.yaml}"

if [ -d "/kaggle/working" ]; then
    DEFAULT_OUTPUT="/kaggle/working/outputs/pixel_motif_dual_scale_heavy"
else
    DEFAULT_OUTPUT="outputs/pixel_motif_dual_scale_heavy"
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

# STEP 2: 200-Epoch Heavy Training + Top-5 Checkpoints + Validation-Tuned TTA & Ensemble Sweep
echo ""
echo "🏋️ [STEP 2/3] Launching Heavy Training Pipeline (200 Epochs)..."
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
echo "🎉 [COMPLETED] Heavy Dual-Scale GNN Pipeline & TTA Sweep finished!"
echo "📦 Model Checkpoints: $OUTPUT_DIR/checkpoints"
echo "📊 TTA & Ensemble Report: $OUTPUT_DIR/all_checkpoints_tta_results.json"
echo "🖼️ Graph Visualizations: $VIS_DIR"
echo "============================================================================"
