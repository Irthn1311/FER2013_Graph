# Historical Compatibility Decision for Pure-GNN v3.1 Scientific Line

## 1. Executive Summary

Based on direct evidence from repository lineages (LAP OFIX7-mid, Gen2 CF-HPG v1.1, and Gen3 WS-HPG v1.0), this document records the compatibility decisions for the upcoming Pure-GNN v3.1 scientific evaluation.

Where historical evidence proves an unambiguous shared convention across all lineages, the convention is recorded as **SOURCE_CONFIRMED**. Where lineages diverge on choices that materially affect fairness or capacity (e.g. batch size, learning rate scheduler, weight decay, label smoothing, augmentation), the configuration field is marked **REQUIRES_REVIEW** and kept `null` in the scientific configuration.

Scientific execution remains strictly **DISABLED** pending independent source and configuration review.

---

## 2. Unambiguous Shared Conventions (SOURCE_CONFIRMED)

The following parameters are identical across LAP, Gen2, and Gen3:

1. **Dataset Split Strategy**:
   - `train.csv`: Training set ($N = 28,709$).
   - `val.csv`: Development / Model Selection set ($N = 3,589$).
   - `test.csv`: Strictly locked holdout final evaluation set ($N = 3,589$).
   - *Status*: **SOURCE_CONFIRMED**. No internal train split or synthetic research dev split is used.
2. **Test Set Isolation**:
   - Official `test.csv` is never opened, read, evaluated, or hashed during model selection or hyperparameter evaluation.
   - *Status*: **SOURCE_CONFIRMED**.
3. **Pixel Input Normalization**:
   - Raw 8-bit intensity values divided by $255.0 \implies [0.0, 1.0]$.
   - *Status*: **SOURCE_CONFIRMED**.
4. **Base Optimizer Family & Initial Learning Rate**:
   - Optimizer: AdamW.
   - Initial Learning Rate: $3 \times 10^{-4}$ ($0.0003$).
   - *Status*: **SOURCE_CONFIRMED**.
5. **Validation Frequency & Model Selection Metric**:
   - Evaluation frequency: Exactly every 1 epoch.
   - Selection metric: Validation accuracy (`val_accuracy`), mode `max`.
   - Tie-break rule: Earliest strict epoch improvement (`earliest_strict_max_val_accuracy`).
   - Save-best-only: True (`best_val_accuracy.keras`).
   - *Status*: **SOURCE_CONFIRMED**.
6. **Early Stopping Metric & Patience**:
   - Monitor: `val_loss`, mode `min`.
   - Patience: 15 epochs.
   - *Status*: **SOURCE_CONFIRMED**.
7. **Canonical Reproducibility Seed**:
   - Primary registered seed across all lineages: `42`.
   - *Status*: **SOURCE_CONFIRMED**.

---

## 3. Divergent Choices Requiring Independent Review (REQUIRES_REVIEW)

The lineages diverge on the following substantive training recipe choices:

### 3.1 Batch Size (LAP 16 vs Gen2/Gen3 64)
- **LAP OFIX7-mid**: Uses batch size **16** (`configs/fer2013_ofix7_mid_tensorflow_baseline.yaml:11`).
- **Gen2 CF-HPG / Gen3 WS-HPG**: Use batch size **64** (`train_validation_only.py:45`, `train_validation_only.py:66`).
- **Impact**: Batch size 16 produces $4\times$ more optimizer updates per epoch (1,794 steps vs 448 steps), fundamentally changing gradient noise and effective learning dynamics.
- **Decision**: `batch_size = null`, `status = "REQUIRES_REVIEW"`.

### 3.2 Learning Rate Scheduler (ReduceLROnPlateau vs WarmupCosine)
- **LAP OFIX7-mid**: Uses `ReduceLROnPlateau` monitoring `val_loss` (patience 5, factor 0.5, min lr 3e-5) (`configs/fer2013_ofix7_mid_tensorflow_baseline.yaml:190-196`).
- **Gen2 / Gen3**: Use `WarmupCosine` (5 warmup epochs linear ramp, followed by cosine decay to 1e-6 over 100 epochs) (`train_validation_only.py:63-100`).
- **Impact**: Plateau scheduling reacts dynamically to validation loss stall, whereas WarmupCosine enforces smooth deterministic decay.
- **Decision**: `lr_scheduler = null`, `status = "REQUIRES_REVIEW"`.

### 3.3 Weight Decay (0.001 vs 0.0005)
- **LAP OFIX7-mid**: Uses $1 \times 10^{-3}$ ($0.001$) (`configs/fer2013_ofix7_mid_tensorflow_baseline.yaml:163`).
- **Gen2 / Gen3**: Use $5 \times 10^{-4}$ ($0.0005$) (`train_validation_only.py:43`, `train_validation_only.py:64`).
- **Impact**: Affects regularization pressure on node projection, relational gate weights, and readout MLP.
- **Decision**: `weight_decay = null`, `status = "REQUIRES_REVIEW"`.

### 3.4 Max Training Epochs (90 vs 100)
- **LAP OFIX7-mid**: 90 epochs (`configs/fer2013_ofix7_mid_tensorflow_baseline.yaml:156`).
- **Gen2 / Gen3**: 100 epochs (`train_validation_only.py:46`, `train_validation_only.py:67`).
- **Decision**: `max_epochs = null`, `status = "REQUIRES_REVIEW"`.

### 3.5 Label Smoothing (0.0 vs 0.05)
- **LAP OFIX7-mid**: `label_smoothing: 0.0` (`configs/fer2013_ofix7_mid_tensorflow_baseline.yaml:148`).
- **Gen2 / Gen3**: `label_smoothing: 0.05` (`train_validation_only.py:50`, `train_validation_only.py:70`).
- **Impact**: Label smoothing prevents overconfident logit saturation, altering calibrated cross-entropy.
- **Decision**: `label_smoothing = null`, `status = "REQUIRES_REVIEW"`.

### 3.6 Data Augmentation Policy
- **LAP**: MediaPipe prior corruption schedules + random cropping/flips.
- **Gen2 / Gen3**: Standardized stateless geometric transformation (rotation $\pm 10^\circ$, translation $\pm 4\text{px}$, flip) + photometric perturbation + random erase (`augmentation.py`).
- **Impact**: In a raw pixel graph (Pure-GNN), pixel perturbations directly alter node input values and edge differences.
- **Decision**: `augmentation_policy = null`, `status = "REQUIRES_REVIEW"`.

---

## 4. Summary Table of Compatibility Status

| Configuration Field | Historical Consensus | Scientific Screen v1 Configuration | Compatibility Status |
|---|---|---|---|
| `scientific_execution_authorized` | N/A | `false` | **LOCKED_DISABLED** |
| `train_rows` | 28,709 | 28709 | **SOURCE_CONFIRMED** |
| `val_rows` | 3,589 | 3589 | **SOURCE_CONFIRMED** |
| `test_access_authorized` | Forbidden | `false` | **SOURCE_CONFIRMED** |
| `optimizer_type` | AdamW | "AdamW" | **SOURCE_CONFIRMED** |
| `learning_rate` | 0.0003 | 0.0003 | **SOURCE_CONFIRMED** |
| `checkpoint_monitor` | val_accuracy | "val_accuracy" | **SOURCE_CONFIRMED** |
| `checkpoint_mode` | max | "max" | **SOURCE_CONFIRMED** |
| `checkpoint_tie_break` | earliest_strict | "earliest_strict_max_val_accuracy" | **SOURCE_CONFIRMED** |
| `early_stopping_monitor` | val_loss | "val_loss" | **SOURCE_CONFIRMED** |
| `early_stopping_patience` | 15 | 15 | **SOURCE_CONFIRMED** |
| `validation_frequency_epochs` | 1 | 1 | **SOURCE_CONFIRMED** |
| `seed` | 42 | 42 | **SOURCE_CONFIRMED** |
| `batch_size` | Divergent (16 vs 64) | `null` | **REQUIRES_REVIEW** |
| `lr_scheduler` | Divergent (Plateau vs Cosine) | `null` | **REQUIRES_REVIEW** |
| `weight_decay` | Divergent (1e-3 vs 5e-4) | `null` | **REQUIRES_REVIEW** |
| `max_epochs` | Divergent (90 vs 100) | `null` | **REQUIRES_REVIEW** |
| `label_smoothing` | Divergent (0.0 vs 0.05) | `null` | **REQUIRES_REVIEW** |
| `global_clipnorm` | Divergent (None vs 1.0) | `null` | **REQUIRES_REVIEW** |
| `augmentation` | Divergent | `null` | **REQUIRES_REVIEW** |
