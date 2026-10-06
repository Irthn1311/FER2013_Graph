# MPG-FER A0 ? Exact Code Lineage, Checkpoint, and Instrumentation Audit (Revised A0-R)

**Audit Date**: September 2026  
**Auditor**: Opencode CLI Agent  
**Repository Working Directory**: `D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5`  
**Current Git Branch**: `research/mpg-fer-v2-1-issue93`  
**Current Git HEAD**: `4967cc5dac3495be2300210215f72422f6f97aa4`  

---

## A0 Correction Log

The following factual corrections from the preliminary report are verified directly against source code and frozen execution artifacts:

1. **`supcon_temperature` in v2.1**:
   - Preliminary: `0.07` (incorrect).
   - Corrected from `mpg_fer_v2_1/config.py`: `supcon_temperature = 0.10`.
2. **`early_stop_patience` in v2**:
   - Preliminary: `10` (incorrect).
   - Corrected from `mpg_fer_v2/config.py`: `early_stop_patience = 20`.
3. **v2 Early-Stopping Semantics**:
   - Corrected: v2 does **not** have an `early_stop_monitor_start_epoch` field.
   - The validation checkpoint comparator is active from epoch 1. Patience increments on every non-improving epoch.
   - Early stopping triggers **only** when `epoch >= min_epochs` (`min_epochs = 50`) **and** `patience >= 20`.
4. **v2.1 Early-Stopping Semantics**:
   - Corrected: v2.1 defines `early_stop_monitor_start_epoch = 85` and `early_stop_patience = 15`.
   - The checkpoint comparator is active from epoch 1, but before epoch 85 `patience` is held/reset at zero.
   - Starting from epoch 85, non-improvement increments `patience`. Early stopping triggers when `epoch >= 85` and `patience >= 15`.
5. **v2 vs v2.1 Scheduler Semantics**:
   - v2: Warmup epochs 1?5, then cosine decay horizon tied to `max_epochs = 120`.
   - v2.1: Warmup epochs 1?5, then cosine decay horizon tied to independent `lr_decay_end_epoch = 85`, followed by a constant floor of `1e-6` up to `max_epochs`.
6. **V1 Provenance Status**:
   - Preliminary: `V1_PROVENANCE_BLOCKED` (based only on Git tree absence).
   - Corrected: **`PROVENANCE_BOUND_BY_EXECUTION_ARTIFACT`**. The official submitted Kaggle notebook `MPG_FER_final_one_shot_kaggle_T4.ipynb` (SHA-256: `296e118ebbea3a77cc9cdca51fe8ec2317f3d247b800ebad8b277aafe444c069`) contains the exact embedded source files in cell 1 (`EMBEDDED_SOURCES`), which compute the exact source tree hash `bf88bce5cf2223816e3708a6bf6b3da6120674c9535863f57110217d6664cac6`. Furthermore, all 11 downloaded embedded Python files from Kaggle run `irthn1311/mpg-fer-v1-final-t4-2026-09-20` match the local `research/mpg_fer_v1/src/mpg_fer_v1` files byte-for-byte.

---

## 1. Repository State & Working Tree Verification

- **Branch**: `research/mpg-fer-v2-1-issue93`
- **HEAD Commit SHA**: `4967cc5dac3495be2300210215f72422f6f97aa4`
- **Tracked Working Tree**: Fully clean. Zero uncommitted modifications to tracked files.
- **Untracked Directories**:
  - Legacy experiment scratch folders: `.codex-tmp-*`
  - Uncommitted local research baseline: `research/mpg_fer_v1/`
  - Local Kaggle execution artifacts: `research/mpg_fer_v2_1/official_runs/`, `research/mpg_fer_v2_1/staged/`
  - Canonical notebooks and audit scratch: `notebooks/MPG_FER_final_one_shot_kaggle_T4.ipynb`, `research/mpg_fer_audit/`

---

## 2. Exact Source Identities

Package source-tree hashes computed over all `*.py` files in each package root:

| Version | Package Path | Origin Binding | Reviewed / Expected Source Hash (SHA-256) | Actual Computed Source Hash (SHA-256) | Provenance Status |
|---|---|---|---|---|---|
| **MPG-FER v1** | `research/mpg_fer_v1/src/mpg_fer_v1` | Official submitted notebook & Kaggle embedded outputs | `bf88bce5cf2223816e3708a6bf6b3da6120674c9535863f57110217d6664cac6` | `bf88bce5cf2223816e3708a6bf6b3da6120674c9535863f57110217d6664cac6` | **BOUND BY EXECUTION ARTIFACT** |
| **MPG-FER v2** | `research/mpg_fer_v2/src/mpg_fer_v2` | Git Commit `96e5aec27cde315038965ee525b14043cc5ab29a` | `f9a06cd4f7482c6a6e37d58844c1a30022602e9fc825eff73240f71740b0af1c` | `f9a06cd4f7482c6a6e37d58844c1a30022602e9fc825eff73240f71740b0af1c` | **BOUND BY GIT COMMIT** |
| **MPG-FER v2.1** | `research/mpg_fer_v2_1/src/mpg_fer_v2_1` | Git Commit `4967cc5dac3495be2300210215f72422f6f97aa4` | `d86d93655c83810d36c89a632baee0745f1de0f0e701b762b719d44445a6e679` | `d86d93655c83810d36c89a632baee0745f1de0f0e701b762b719d44445a6e679` | **BOUND BY GIT COMMIT** |

---

## 3. V1 Provenance Closure

### Definitive Evidence from Execution Artifacts
1. **Official Notebook Embedding**:
   - Notebook `research/mpg_fer_v1/notebooks/MPG_FER_final_one_shot_kaggle_T4.ipynb` (SHA-256: `296e118ebbea3a77cc9cdca51fe8ec2317f3d247b800ebad8b277aafe444c069`) was pushed to Kaggle kernel `irthn1311/mpg-fer-v1-final-t4-2026-09-20`.
   - Cell 1 defines `EMBEDDED_SOURCES` containing all 11 modular Python files.
   - Parsing `EMBEDDED_SOURCES` and computing the canonical package hash produces:
     `bf88bce5cf2223816e3708a6bf6b3da6120674c9535863f57110217d6664cac6`.
2. **Kaggle Execution Output**:
   - The official downloaded artifacts in `research/mpg_fer_v1/outputs/kaggle_t4_final/mpg_fer_v1_embedded/mpg_fer_v1` contain all 11 Python files extracted during the official run.
   - Every file matches the local `research/mpg_fer_v1/src/mpg_fer_v1` source byte-for-byte.
3. **Execution Manifest & Checkpoint**:
   - `execution_manifest.json` confirms `selected_checkpoint_sha256 = 548325add48fc87a1f5011875c1f005123bbb82d8dfce5e7ba23285ad3f64a52` at epoch 72.
   - The physical checkpoint matches this SHA-256 exactly and strict-loads into the model with zero errors.
- **Conclusion**: V1 source and checkpoint are definitively bound by immutable execution artifacts: **`PROVENANCE_BOUND_BY_EXECUTION_ARTIFACT`**.

---

## 4. V1 -> V2 Known / Verifiable Changes

Comparing the verified v1 code with the git-locked v2 codebase (`96e5aec`):
1. **Spatial Supports**:
   - v1: Single spatial scale (12x12 windows, stride 6, 49 windows total).
   - v2: Three concentric multiscale spatial supports (8x8, 12x12, 16x16; 64, 144, 256 pixels per window) with learned scale saliency and scale gating.
2. **Trainable Parameters**:
   - Increased from `2,219,788` (v1) to `2,238,609` (v2), an increase of `+18,821` parameters located entirely in `motif_composer` (scale saliency networks and scale gate).
3. **Loss Formulation**:
   - v1: Diversity loss + cluster utilization loss.
   - v2: Mutual information loss (L_MI) balancing local cluster entropy and global prototype coverage + horizontal flip consistency loss (symmetric JS divergence).
4. **Checkpointing & Selection**:
   - v1: Online weights, selected via PublicTest TTA accuracy.
   - v2: Model EMA (decay 0.999), selected via EMA PublicTest flip-TTA accuracy.
5. **Resume & Continuation**:
   - v1: Single one-shot script, no atomic continuation.
   - v2: Full atomic state-dict continuation with SHA-256 verification and dataloader/sampler resume.

---

## 5. Exact V2 -> V2.1 Source Delta

Local byte-by-byte and AST comparison between v2 (`96e5aec`) and v2.1 (`4967cc5`):

### 5.1 Byte-Identical Files (Confirmed Locally)
- `features.py`
- `graph.py`
- `data.py`
- `evaluate.py`
- `utils.py`

### 5.2 Exact Symbol Differences & Semantic Classification

| File | Exact Field / Symbol | Old Behavior (v2) | New Behavior (v2.1) | Classification | Parameter Effect | Inference Effect | Repr Effect |
|---|---|---|---|---|---|---|---|
| `config.py` | `pixel_dropout` | 0.15 | 0.10 | REGULARIZATION | 0 | No | No (eval) |
| `config.py` | `pixel_drop_path_max` | 0.05 | 0.03 | REGULARIZATION | 0 | No | No (eval) |
| `config.py` | `motif_dropout` | 0.15 | 0.10 | REGULARIZATION | 0 | No | No (eval) |
| `config.py` | `motif_drop_path_max` | 0.10 | 0.05 | REGULARIZATION | 0 | No | No (eval) |
| `config.py` | `classifier_dropout` | 0.30 | 0.25 | REGULARIZATION | 0 | No | No (eval) |
| `config.py` | `lambda_mi` | 0.05 | 0.025 | LOSS | 0 | No | No (eval) |
| `config.py` | `consistency_probability` | 0.20 | 0.50 | REGULARIZATION | 0 | No | No (eval) |
| `config.py` | `lambda_consistency` | 0.05 | 0.15 | LOSS | 0 | No | No (eval) |
| `config.py` | `d_supcon` | N/A | 128 | ARCHITECTURE | +65,920 | No | Direct output |
| `config.py` | `lambda_supcon` | N/A | 0.05 | LOSS | 0 | No | No (eval) |
| `config.py` | `supcon_temperature` | N/A | **0.10** | LOSS | 0 | No | No (eval) |
| `config.py` | `lr_decay_end_epoch` | N/A (tied to max_epochs=120) | 85 | SCHEDULE | 0 | No | No |
| `config.py` | `early_stop_monitor_start_epoch`| N/A (gated by min_epochs=50) | 85 | SCHEDULE | 0 | No | No |
| `config.py` | `early_stop_patience` | **20** | **15** | SCHEDULE | 0 | No | No |
| `motif.py` | `raw_tau` | `nn.Parameter(torch.tensor(0.0))` | Replaced by `register_buffer("current_tau", ...)` | ARCHITECTURE / SCHEDULE | -1 | Yes | Yes |
| `motif.py` | `scheduled_motif_temperature` | N/A | Cosine decay helper ($0.70 	o 0.30$ through epoch 35) | SCHEDULE | 0 | No | Yes |
| `model.py` | `supcon_head` | None | `Sequential(Linear(512, 128), ReLU, Linear(128, 128))` | ARCHITECTURE / TRAINING_ONLY | +65,920 | No | Direct output |
| `model.py` | `forward()` | Returns `logits, aux` | Returns `logits, aux` with `supcon_embeddings` | LOSS / TRAINING_ONLY | 0 | No | Direct output |
| `losses.py` | `supervised_contrastive_loss` | None | Supervised contrastive loss function | LOSS | 0 | No | No |
| `ema.py` | `update()` | Exponential moving average on all floats | Copies non-parameter buffers directly without averaging | EMA | 0 | Yes (EMA) | Yes (EMA) |
| `checkpoint.py` | `build_resume_bundle` | Schema version 2 | Schema version 3 (tracks `scheduled_tau`, supcon state) | RESUME/LIFECYCLE | 0 | No | No |
| `train.py` | `train_epoch()` | Step LR decay at 120 | Step LR decay at 85; floor 1e-6; SupCon training loop | TRAINING_ONLY | 0 | No | No |
| `kaggle.py` | `resolve_resume_artifact` | Resolves `mpg-fer-v2-resume` | Resolves `mpg-fer-v2-1-resume` | KAGGLE_ROUTING | 0 | No | No |

---

## 6. Parameter & Component Reconciliation

| Component | Submodules | v1 Trainable Params | v2 Trainable Params | v2.1 Trainable Params | Delta (v2 -> v2.1) |
|---|---|---|---|---|---|
| **Pixel Projection** | `pixel_proj` | 3,360 | 3,360 | 3,360 | 0 |
| **Pixel GNN** | `pixel_gnn` (4 blocks) | 301,536 | 301,536 | 301,536 | 0 |
| **Pixel Readout** | `pixel_attn_pool`, `pixel_readout_proj` | 37,345 | 37,345 | 37,345 | 0 |
| **Aux Pixel Head** | `aux_pixel_head` | 903 | 903 | 903 | 0 |
| **Motif Composer** | `assignment_query`, `prototype_key`, `scale_saliency`, `type_proj`, `occurrence_proj`, `scale_gate`, `raw_tau` | 32,451 | 51,272 | 51,271 | -1 (`raw_tau` removed) |
| **Motif GNN** | `motif_gnn` (5 blocks) | 1,485,330 | 1,485,330 | 1,485,330 | 0 |
| **Motif Readout** | `motif_attn_pool`, `motif_readout_proj` | 222,529 | 222,529 | 222,529 | 0 |
| **Aux Motif Head** | `aux_motif_head` | 2,695 | 2,695 | 2,695 | 0 |
| **FER Classifier** | `classifier` | 133,639 | 133,639 | 133,639 | 0 |
| **SupCon Head** | `supcon_head` | 0 | 0 | 65,920 | +65,920 (new training head) |
| **Total Trainable** | | **2,219,788** | **2,238,609** | **2,304,528** | **+65,919** |

---

## 7. Checkpoint Identities & Strict Loading

| Version | Checkpoint File Path | File SHA-256 | Expected SHA-256 | Top-Level Keys | Selected Epoch | Weights Type | Strict Load Result |
|---|---|---|---|---|---|---|---|
| **v1** | `research/mpg_fer_v1/outputs/kaggle_t4_final/mpg_fer_v1_run/best_val_acc.pt` | `548325add48fc87a1f5011875c1f005123bbb82d8dfce5e7ba23285ad3f64a52` | `548325add48fc87a1f5011875c1f005123bbb82d8dfce5e7ba23285ad3f64a52` | `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_metrics`, `config`, `scheduler_state_dict`, `selection_metric`, `selection_tiebreak` | 72 | Online | **PASS** (0 missing, 0 unexpected) |
| **v2** | `research/mpg_fer_v2/outputs/kaggle_v2_final/final_run/best_val_acc.pt` | `f3cda72fc4d791e7017e2e0374f83ef22e9f22f03e8172b7389e53f8cbc6dc1c` | `f3cda72fc4d791e7017e2e0374f83ef22e9f22f03e8172b7389e53f8cbc6dc1c` | `checkpoint_type`, `weights_type`, `epoch`, `model_state_dict`, `ema_state_dict`, `val_metrics`, `config`, `source_hash`, `selection_metric`, `selection_tiebreak` | 62 | EMA | **PASS** (0 missing, 0 unexpected) |
| **v2.1** | `research/mpg_fer_v2_1/official_runs/segment_02/mpg_fer_v2_1_run/best_val_acc.pt` | `4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75` | `4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75` | `checkpoint_type`, `weights_type`, `epoch`, `model_state_dict`, `ema_state_dict`, `scheduled_tau`, `val_metrics`, `config`, `source_hash`, `selection_metric`, `selection_tiebreak` | 57 | EMA | **PASS** (0 missing, 0 unexpected) |

---

## 8. Dynamic Forward Shape & Non-Invasive Hook Audit

Audited across all three models using synthetic batch $[B=2, 1, 48, 48]$ in evaluation mode:

| Target Representation | Target Module & Hook Point | v1 Shape | v2 Shape | v2.1 Shape | Hook Non-Interference Proof |
|---|---|---|---|---|---|
| **Final Pixel GNN** | `model.pixel_gnn[-1]` (forward hook) | `[2, 2304, 96]` | `[2, 2304, 96]` | `[2, 2304, 96]` | Diff = 0.0 (Bit-Identical) |
| **Occurrence Raw** | `model.motif_composer.occurrence_proj` (forward pre-hook) | 1 call of `[2, 49, 133]` (12x12) | 3 calls of `[2, 49, 133]` (8x8, 12x12, 16x16) | 3 calls of `[2, 49, 133]` (8x8, 12x12, 16x16) | Diff = 0.0 (Bit-Identical) |
| **Occurrence Sub-Split** | Slices: `WHAT[:96]`, `TYPE[96:128]`, `WHERE[128:133]` | WHAT: `96`<br>TYPE: `32`<br>WHERE: `5` | WHAT: `96`<br>TYPE: `32`<br>WHERE: `5` | WHAT: `96`<br>TYPE: `32`<br>WHERE: `5` | Verified exact slice shapes |
| **Scale Candidates** | `model.motif_composer.scale_gate` (forward pre-hook) | N/A (single scale) | `[2, 49, 3, 192]` | `[2, 49, 3, 192]` | Diff = 0.0 (Bit-Identical) |
| **Pre-Motif GNN** | `model.motif_composer` (forward hook output[0]) | `[2, 49, 192]` | `[2, 49, 192]` | `[2, 49, 192]` | Diff = 0.0 (Bit-Identical) |
| **Post-Motif GNN** | `model.motif_gnn[-1]` (forward hook) | `[2, 49, 192]` | `[2, 49, 192]` | `[2, 49, 192]` | Diff = 0.0 (Bit-Identical) |
| **Pixel Readout** | Direct output `aux['h_pixel_readout']` | `[2, 128]` | `[2, 128]` | `[2, 128]` | Exact match |
| **Motif Readout** | Direct output `aux['h_motif_readout']` | `[2, 384]` | `[2, 384]` | `[2, 384]` | Exact match |
| **Fusion** | Direct output or Concat | `[2, 512]` (concat) | `[2, 512]` (concat) | `[2, 512]` (direct) | Exact match |
| **Assignments** | Direct output `aux['motif_assignments']` | `[2, 2304, 48]` | `[2, 2304, 48]` | `[2, 2304, 48]` | Exact match |

---

## 9. Final Audit Verdict

Because V1 provenance was recovered and cryptographically bound via official Kaggle execution artifacts and embedded source code in `MPG_FER_final_one_shot_kaggle_T4.ipynb`, and V2 / V2.1 are bound by Git commits:

**`A0_PASS_THREE_VERSION_PROVENANCE`**
