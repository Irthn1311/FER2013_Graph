# Historical Train / Val / Test Protocol Audit

## 1. Overview and Lineage Mapping

This audit extracts the exact empirical training, validation, and evaluation conventions across the three primary research lineages in the FER2013_Graph repository:
1. **LAP (Landmark-Aware Pixel GNN / OFIX7-mid)**: The locked TensorFlow reference in `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/` and historical PyTorch candidates.
2. **Gen2 (CF-HPG / RA-HPG Lineage)**: Cascaded Filtered Hierarchical Pixel Graph and Relation-Aware variants (`research/candidates/tf_cf_hpg_v1_1_resolution/`, `research/candidates/tf_ra_hpg_v1_relation_aware/`).
3. **Gen3 (WS-HPG Lineage)**: Weak-Support Hierarchical Pixel Graph (`research/candidates/tf_ws_hpg_v1_training/`).

---

## 2. Evidence-Supported Historical Protocols

| Protocol Field | Lineage 1: LAP (OFIX7-mid) | Lineage 2: Gen2 (CF-HPG v1.1) | Lineage 3: Gen3 (WS-HPG v1.0) | Agreement Status |
|---|---|---|---|---|
| **Train Source** | `train.csv` | `train.csv` | `train.csv` | **SHARED CONVENTION** |
| **Train Row Count** | 28,709 | 28,709 | 28,709 | **SHARED CONVENTION** |
| **Val Source** | `val.csv` (model selection) | `val.csv` (model selection) | `val.csv` (model selection) | **SHARED CONVENTION** |
| **Val Row Count** | 3,589 | 3,589 | 3,589 | **SHARED CONVENTION** |
| **Test Source** | `test.csv` (locked final) | `test.csv` (locked final) | `test.csv` (locked final) | **SHARED CONVENTION** |
| **Test Row Count** | 3,589 | 3,589 | 3,589 | **SHARED CONVENTION** |
| **Test Accessed in Selection** | **NO** | **NO** | **NO** | **SHARED CONVENTION** |
| **Pixel Normalization** | Raw pixels $/ 255.0 \in [0, 1]$ | Raw pixels $/ 255.0 \in [0, 1]$ | Raw pixels $/ 255.0 \in [0, 1]$ | **SHARED CONVENTION** |
| **Optimizer** | AdamW | AdamW | AdamW | **SHARED CONVENTION** |
| **Initial Learning Rate** | $3 \times 10^{-4}$ ($0.0003$) | $3 \times 10^{-4}$ ($0.0003$) | $3 \times 10^{-4}$ ($0.0003$) | **SHARED CONVENTION** |
| **Weight Decay** | $1 \times 10^{-3}$ ($0.001$) | $5 \times 10^{-4}$ ($0.0005$) | $5 \times 10^{-4}$ ($0.0005$) | **DIVERGENT** |
| **Batch Size** | 16 | 64 | 64 | **DIVERGENT** |
| **Max Epochs** | 90 | 100 | 100 | **DIVERGENT** |
| **LR Scheduler** | ReduceLROnPlateau (factor 0.5, pat 5) | WarmupCosine (5 warmup, cosine 100) | WarmupCosine (5 warmup, cosine 100) | **DIVERGENT** |
| **Early Stopping Metric** | `val_loss` (min) | `val_loss` (min) | `val_loss` (min) | **SHARED CONVENTION** |
| **Early Stopping Patience** | 15 epochs (min stop 30) | 15 epochs | 15 epochs | **SHARED CONVENTION** |
| **Validation Frequency** | Every 1 epoch | Every 1 epoch | Every 1 epoch | **SHARED CONVENTION** |
| **Checkpoint Monitor** | `val_accuracy` (max) | `val_accuracy` (max) | `val_accuracy` (max) | **SHARED CONVENTION** |
| **Checkpoint Tie-Break** | Earliest strict improvement | Earliest strict improvement | Earliest strict improvement | **SHARED CONVENTION** |
| **Selected Checkpoint** | `best_val_accuracy` | `best_val_accuracy` | `best_val_accuracy` | **SHARED CONVENTION** |
| **Loss Objective** | Categorical Cross-Entropy | Categorical Cross-Entropy | Categorical Cross-Entropy | **SHARED CONVENTION** |
| **Label Smoothing** | 0.0 (None) | 0.05 | 0.05 | **DIVERGENT** |
| **Global Clipnorm** | None / Unspecified | 1.0 | 1.0 | **DIVERGENT** |
| **Augmentation** | Prior corruption + crop/flip | Geometric + photometric + erase | Paired geometric + photometric + erase | **DIVERGENT** |
| **Canonical Seed** | 42 | 42 | 42 | **SHARED CONVENTION** |

---

## 3. Direct Source Evidence Citations

### 3.1 Lineage 1: LAP (OFIX7-mid)
- `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/configs/fer2013_ofix7_mid_tensorflow_baseline.yaml`:
  - `data.batch_size`: 16 (line 11)
  - `training.max_epochs`: 90 (line 156)
  - `training.lr`: 0.0003 (line 162)
  - `training.weight_decay`: 0.001 (line 163)
  - `training.optimizer.type`: adamw (line 165)
  - `training.eval_every_epoch`: true (line 171)
  - `training.checkpoint_monitor`: val_accuracy (line 174)
  - `training.checkpoint_policy.tie_break`: earliest_epoch_strict_improvement (line 204)
  - `training.early_stopping`: enabled, patience 15, metric val_loss (lines 183-188)
  - `training.scheduler`: plateau, monitor val_loss, factor 0.5, patience 5 (lines 189-196)
  - `loss.label_smoothing`: 0.0 (line 148)
  - `locked.dataset_split_signature`: fer2013_train28709_val3589_test3589 (line 264)

### 3.2 Lineage 2: Gen2 (CF-HPG v1.1 Resolution)
- `research/candidates/tf_cf_hpg_v1_1_resolution/train_validation_only.py`:
  - `DATASET_SIGNATURE`: fer2013_train28709_val3589_test3589 (line 27)
  - `TRAIN_SAMPLES`: 28_709 (line 29)
  - `VALIDATION_SAMPLES`: 3_589 (line 30)
  - `TRAINING_CONFIG.optimizer`: AdamW (line 41)
  - `TRAINING_CONFIG.learning_rate`: 3e-4 (line 42)
  - `TRAINING_CONFIG.weight_decay`: 5e-4 (line 43)
  - `TRAINING_CONFIG.global_clipnorm`: 1.0 (line 44)
  - `TRAINING_CONFIG.batch_size`: 64 (line 45)
  - `TRAINING_CONFIG.max_epochs`: 100 (line 46)
  - `TRAINING_CONFIG.warmup_epochs`: 5 (line 47)
  - `TRAINING_CONFIG.cosine_final_learning_rate`: 1e-6 (line 48)
  - `TRAINING_CONFIG.label_smoothing`: 0.05 (line 50)
  - `TRAINING_CONFIG.checkpoint`: earliest_strict_max_val_accuracy (line 51)
  - `TRAINING_CONFIG.early_stopping_monitor`: val_loss, patience 15 (lines 52-53)

### 3.3 Lineage 3: Gen3 (WS-HPG v1.0 Training Preparation)
- `research/candidates/tf_ws_hpg_v1_training/train_validation_only.py`:
  - `TRAIN_SAMPLES`: 28_709 (line 31)
  - `VALIDATION_SAMPLES`: 3_589 (line 32)
  - `TRAINING_CONFIG.seed`: 42 (line 61)
  - `TRAINING_CONFIG.optimizer`: AdamW (line 62)
  - `TRAINING_CONFIG.learning_rate`: 3e-4 (line 63)
  - `TRAINING_CONFIG.weight_decay`: 5e-4 (line 64)
  - `TRAINING_CONFIG.global_clipnorm`: 1.0 (line 65)
  - `TRAINING_CONFIG.batch_size`: 64 (line 66)
  - `TRAINING_CONFIG.max_epochs`: 100 (line 67)
  - `TRAINING_CONFIG.warmup_epochs`: 5 (line 68)
  - `TRAINING_CONFIG.cosine_final_learning_rate`: 1e-6 (line 69)
  - `TRAINING_CONFIG.training_label_smoothing`: 0.05 (line 70)
  - `TRAINING_CONFIG.validation_every_epochs`: 1 (line 71)
  - `TRAINING_CONFIG.checkpoint`: earliest_strict_max_val_accuracy (line 72)
  - `TRAINING_CONFIG.early_stopping_monitor`: val_loss, patience 15 (lines 73-75)
- `research/candidates/tf_ws_hpg_v1_training/augmentation.py`:
  - Deterministic stateless augmentation with horizontal flip, rotation $\pm 10^\circ$, translation $\pm 4\text{px}$, contrast $[0.85, 1.15]$, brightness $\pm 0.10$, random erase (lines 21-42).
