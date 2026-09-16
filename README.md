# Pixel GNN: Pure Pixel Graph Models for FER2013

This repository contains TensorFlow/Keras implementations of pure pixel graph neural architectures for Facial Expression Recognition on FER2013, free of heavy landmark extraction, CNN backbones, or global transformers.

## Architectures

All models are unified under `standalone/pixel_gnn/`:

1. **`pixel_neighbor_motif`** (Pure Pixel Neighbor Attention + Learned Motif Prototypes):
   - **Input**: FER2013 48x48 pixel images.
   - **Nodes**: 2,304 pixel nodes with 5 basic features: $[I, x, y, g_x, g_y]$.
   - **Edges**: 8-neighbor directional edges with 4 edge attributes: $[\Delta x, \Delta y, \text{dist}, |\Delta I|]$.
   - **Neighbor Attention**: Local multi-head neighbor attention without deep message passing.
   - **Global Aggregation**: Learned motif prototype attention pooling (e.g. 16 prototypes) aggregating pixel tokens into an image representation.
   - **Classifier**: Final Dense projection to 7 emotion classes.
   - **Parameters**: Ultra-compact (~28,201 parameters).

2. **`pixel_gnn_only`** (Pure Pixel Graph Neural Network):
   - **Message Passing**: Gated edge-conditioned layers operating directly on pixel grids.
   - **Parameters**: ~189,319 parameters.

---

## Unified Execution Entrypoint (`train.py`)

All models and configurations can be run using the single root entrypoint `train.py`.

### 1. Fast Smoke Test (No external data needed)

Run on synthetic data to verify model forward pass, backward pass, loss, and optimization:

```bash
# Smoke test Neighbor Attention + Motif Prototypes
python train.py --config standalone/pixel_gnn/configs/fer2013_pixel_neighbor_motif_fast_seed42.yaml --smoke --synthetic

# Smoke test Pixel GNN
python train.py --config standalone/pixel_gnn/configs/fer2013_pixel_gnn_only_fast_seed42.yaml --smoke --synthetic
```

### 2. Training on FER2013 (Local or Kaggle)

```bash
# Train pixel_neighbor_motif on FER2013 CSV
python train.py --config standalone/pixel_gnn/configs/fer2013_pixel_neighbor_motif_kaggle_fast_seed42.yaml --fer-csv /path/to/train.csv --output-root outputs/pixel_neighbor_motif
```

The training harness automatically:
- Evaluates validation metrics (loss, accuracy, macro F1, weighted F1) at each epoch.
- Checkpoints the best model based on validation loss.
- Evaluates the best checkpoint on the test set upon training completion.

### 3. Running Unit Tests

```bash
python standalone/pixel_gnn/tests/test_pixel_gnn.py
```
