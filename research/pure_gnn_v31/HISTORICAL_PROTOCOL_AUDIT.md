# Historical Train / Val / Test Protocol Audit

## 1. Overview and Lineage Mapping

This audit extracts the exact empirical training, validation, and evaluation conventions across the three primary research lineages in the FER2013_Graph repository:
1. **LAP (Landmark-Aware Pixel GNN / OFIX7-mid)**: The frozen TensorFlow reference in `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/` and historical PyTorch candidates.
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
| **Pixel Normalization** | `image / 255.0 if image.max() > 1.0 else image` | Raw $[0, 255] \to [-1, 1]$ (`/ 127.5 - 1.0`) | `raw_image / 255.0` $\to [0, 1]$ | **DIVERGENT (NOT SHARED)** |
| **Image Augmentation** | **UNKNOWN** (no executable crop/flip found) | Horizontal flip + rot $\pm 10^\circ$ + trans $\pm 4\text{px}$ + contrast + brightness + erase | Paired horizontal flip + rot $\pm 10^\circ$ + trans $\pm 4\text{px}$ + contrast + brightness + erase | **DIVERGENT** |
| **Structural Prior Corruption** | **SOURCE_CONFIRMED** (`attenuate`, `shuffle`, `zero`, `fallback`) | N/A (no MediaPipe priors) | N/A (no MediaPipe priors) | **LAP-SPECIFIC** |
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
| **Canonical Seed** | 42 | 42 | 42 | **SHARED CONVENTION** |

---

## 3. Direct Source Evidence Citations

### 3.1 Lineage 1: LAP (OFIX7-mid)
- **Pixel Normalization**:
  - `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/src/lap_gnn_tf/graph/builder.py:569`:
    `image_norm = image / 255.0 if image.max() > 1.0 else image`
  - Note: `src/lap_gnn_tf/data/fer2013.py` only validates CSV headers and row count; it does not perform pixel transformation.
- **Image Augmentation**:
  - `image_augmentation`: **UNKNOWN**. No executable image crop or flip logic exists in `lap_gnn_tf` or baseline configs.
- **Structural Prior Corruption**:
  - **SOURCE_CONFIRMED**: `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/configs/fer2013_ofix7_mid_tensorflow_baseline.yaml:58-77`:
    `prior_corruption`: `attenuate_prior` (0.55), `shuffle_prior` (0.25), `zero_prior` (0.12), `forced_fallback` (0.08).
- **Training Hyperparameters**:
  - `data.batch_size`: 16 (`baseline.yaml:11`, `line 157`)
  - `training.max_epochs`: 90 (`baseline.yaml:156`)
  - `training.lr`: 0.0003 (`baseline.yaml:162`)
  - `training.weight_decay`: 0.001 (`baseline.yaml:163`)
  - `training.optimizer.type`: adamw (`baseline.yaml:165`)
  - `training.eval_every_epoch`: true (`baseline.yaml:171`)
  - `training.checkpoint_monitor`: val_accuracy (`baseline.yaml:174`)
  - `training.checkpoint_policy.tie_break`: earliest_epoch_strict_improvement (`baseline.yaml:204`)
  - `training.early_stopping`: enabled, patience 15, metric val_loss (`baseline.yaml:183-188`)
  - `training.scheduler`: plateau, monitor val_loss, factor 0.5, patience 5 (`baseline.yaml:189-196`)
  - `loss.label_smoothing`: 0.0 (`baseline.yaml:148`)
  - `locked.dataset_split_signature`: fer2013_train28709_val3589_test3589 (`baseline.yaml:264`)

### 3.2 Lineage 2: Gen2 (CF-HPG v1.1 Resolution)
- **Pixel Normalization**:
  - `research/candidates/tf_cf_hpg_v1_1_resolution/data.py`: Loads raw pixels without `/255`.
  - `research/candidates/tf_cf_hpg_v1_1_resolution/model.py` (`patchify_and_scale()`):
    `images = images / 127.5 - 1.0` $\implies$ effective normalization is $[-1.0, 1.0]$.
- **Image Augmentation**:
  - `research/candidates/tf_cf_hpg_v1_1_resolution/data.py:51-78`:
    Horizontal flip, rotation $\pm 10^\circ$, translation $\pm 4\text{px}$, contrast $[0.85, 1.15]$, brightness $\pm 0.10$, random erase.
- **Training Hyperparameters**:
  - `DATASET_SIGNATURE`: fer2013_train28709_val3589_test3589 (`train_validation_only.py:27`)
  - `TRAIN_SAMPLES`: 28_709 (`train_validation_only.py:29`)
  - `VALIDATION_SAMPLES`: 3_589 (`train_validation_only.py:30`)
  - `TRAINING_CONFIG.optimizer`: AdamW (`train_validation_only.py:41`)
  - `TRAINING_CONFIG.learning_rate`: 3e-4 (`train_validation_only.py:42`)
  - `TRAINING_CONFIG.weight_decay`: 5e-4 (`train_validation_only.py:43`)
  - `TRAINING_CONFIG.global_clipnorm`: 1.0 (`train_validation_only.py:44`)
  - `TRAINING_CONFIG.batch_size`: 64 (`train_validation_only.py:45`)
  - `TRAINING_CONFIG.max_epochs`: 100 (`train_validation_only.py:46`)
  - `TRAINING_CONFIG.warmup_epochs`: 5 (`train_validation_only.py:47`)
  - `TRAINING_CONFIG.cosine_final_learning_rate`: 1e-6 (`train_validation_only.py:48`)
  - `TRAINING_CONFIG.label_smoothing`: 0.05 (`train_validation_only.py:50`)
  - `TRAINING_CONFIG.checkpoint`: earliest_strict_max_val_accuracy (`train_validation_only.py:51`)
  - `TRAINING_CONFIG.early_stopping_monitor`: val_loss, patience 15 (`train_validation_only.py:52-53`)

### 3.3 Lineage 3: Gen3 (WS-HPG v1.0 Training Preparation)
- **Pixel Normalization**:
  - `research/candidates/tf_ws_hpg_v1_training/data.py:35`: Validation path uses `image / 255.0` $\implies [0, 1]$.
  - `research/candidates/tf_ws_hpg_v1_training/augmentation.py:59`: Training augmentation starts with `raw_image / 255.0` $\implies [0, 1]$.
- **Image Augmentation**:
  - `research/candidates/tf_ws_hpg_v1_training/augmentation.py:21-42`:
    Paired stateless horizontal flip, rotation $\pm 10^\circ$, translation $\pm 4\text{px}$, contrast $[0.85, 1.15]$, brightness $\pm 0.10$, erase area $[0.02, 0.10]$, aspect $[0.5, 2.0]$.
- **Training Hyperparameters**:
  - `TRAIN_SAMPLES`: 28_709 (`train_validation_only.py:31`)
  - `VALIDATION_SAMPLES`: 3_589 (`train_validation_only.py:32`)
  - `TRAINING_CONFIG.seed`: 42 (`train_validation_only.py:61`)
  - `TRAINING_CONFIG.optimizer`: AdamW (`train_validation_only.py:62`)
  - `TRAINING_CONFIG.learning_rate`: 3e-4 (`train_validation_only.py:63`)
  - `TRAINING_CONFIG.weight_decay`: 5e-4 (`train_validation_only.py:64`)
  - `TRAINING_CONFIG.global_clipnorm`: 1.0 (`train_validation_only.py:65`)
  - `TRAINING_CONFIG.batch_size`: 64 (`train_validation_only.py:66`)
  - `TRAINING_CONFIG.max_epochs`: 100 (`train_validation_only.py:67`)
  - `TRAINING_CONFIG.warmup_epochs`: 5 (`train_validation_only.py:68`)
  - `TRAINING_CONFIG.cosine_final_learning_rate`: 1e-6 (`train_validation_only.py:69`)
  - `TRAINING_CONFIG.training_label_smoothing`: 0.05 (`train_validation_only.py:70`)
  - `TRAINING_CONFIG.validation_every_epochs`: 1 (`train_validation_only.py:71`)
  - `TRAINING_CONFIG.checkpoint`: earliest_strict_max_val_accuracy (`train_validation_only.py:72`)
  - `TRAINING_CONFIG.early_stopping_monitor`: val_loss, patience 15 (`train_validation_only.py:73-75`)

---

## 4. Pure-GNN Specific Input Contract

Because pixel normalization is **DIVERGENT** across historical lineages:
- Pure-GNN scientific loader uses `raw_pixel / 255.0` $\implies [0.0, 1.0]$.
- This is designated as **`PURE_GNN_SPECIFIC_INPUT_CONTRACT`** (derived from `research/pure_gnn_v31/src/pure_gnn_v31/scientific/dataset.py:46`).
- It is explicitly NOT a shared historical consensus.
