# LAP-GNN: Landmark-Aware Pixel Graph Neural Network for FER2013

This repository contains the standalone TensorFlow/Keras implementation of **LAP-GNN OFIX7-mid**, designed for Facial Expression Recognition on the FER2013 dataset.

## Core Package

The self-contained, frozen reference package is located at:
`standalone/lap_gnn_tensorflow_ofix7_mid_candidate/`

### Model Specifications
- **Input**: FER2013 48x48 pixel images with MediaPipe landmark priors.
  - 2,304 pixel nodes with 37 node channels (intensity, spatial gradient/detail, 32 MediaPipe prior channels).
  - ~17,860 edges with 8 edge attributes (relative deltas, distance, gradient/intensity differences, part similarities).
- **Architecture**:
  - `PixelEncoder`: Pointwise Dense (37 -> 96).
  - `EdgeContextEncoder`: 3 Gated Edge-Context Message Passing layers (hidden size 96).
  - `part_pool` & `MicroMotifSupportReadout`: 5 anatomical facial parts (Mouth, Eye, Brow, Nose/Cheek, Global) with 20 micro-motif tokens producing a 480-dimensional image representation.
  - `Classifier`: Dense (480 -> 7 classes).
  - **Trainable Parameters**: Exactly 1,061,192.

## Kaggle Execution (2x GPU)

The repository provides configurations and notebooks for running on Kaggle with 2x Tesla T4 GPUs:

- **2-GPU Configuration**:
  `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/configs/fer2013_ofix7_mid_tensorflow_kaggle_2gpu.yaml`
  - Global batch size: 32 (16 per GPU).
  - `mixed_precision: true`, `memory_growth: true`.
  - Multi-worker prefetching (`graph_workers: 4`, `tf_data_prefetch: 4`).
- **Main Kaggle Notebook**:
  `notebooks/kaggle-end-to-end.ipynb`
- **Validation Notebook**:
  `notebooks/kaggle-issue7-validation-only.ipynb`

### Kaggle Input Datasets
1. FER2013 split CSVs (`train.csv`, `val.csv`, `test.csv`):
   `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`
2. Verified D16 MediaPipe priors:
   `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`
3. Clean graph cache:
   `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`

## Local Validation & Testing

Run commands from the package directory:

```bash
cd standalone/lap_gnn_tensorflow_ofix7_mid_candidate

# Inspect environment
python -m lap_gnn_tf.cli.inspect_environment

# Compare golden weights and parity
python -m lap_gnn_tf.cli.compare_golden --package-root .

# Validate baseline configuration
python -m lap_gnn_tf.cli.validate --config configs/fer2013_ofix7_mid_tensorflow_baseline.yaml --golden

# Run test suite
pytest -q
```
