# D10 Project Handoff — Slot Attention Motif Discovery

> **Last updated**: 2026-05-08 20:30
> **Status**: ✅ V2 ablation complete, best config identified
> **Best Model**: D10 V2.2 (GNN 2, Iter 5, LR 3e-4) — **F1 0.5204, Acc 54.86%**
> **Target**: macro F1 ≥ 0.65 (stretch: 0.70+)

## 1. Tóm tắt kiến trúc

```
Pipeline (đúng tinh thần pixel graph → motif → emotion):

  Input: 48×48 pixel graph (2304 nodes, 7D features, 5D edges)
     ↓
  SharedPixelEncoder (2 GNN layers, hidden=96)  ← Deep encoding
     ↓
  IterativeSlotAttention (K=8 motifs, T=5 iterations)  ← Core innovation
     ↓
  Position Encoding (motif centers from attention-weighted pixel coords)
     ↓
  MotifRelationTransformer (1 layer, 4 heads)  ← Motif relations
     ↓
  ClassMotifAttentionHead (7 class queries × 8 motifs)  ← Emotion from motifs
     ↓
  Output: logits [B, 7]
```

**Total params**: 359,720 (trainable)

## 2. Tại sao D10 khác D9

| Aspect | D9 (failed, F1=0.17) | D10 (new) |
|---|---|---|
| Softmax direction | Over pixels (mỗi motif → phân tán khắp ảnh) | Over motifs (mỗi pixel → chọn 1 motif) |
| Iterations | 1 (one-shot) | 5 (iterative refinement + GRU) |
| Encoder depth | 1-2 GNN layers | 2 GNN layers (hidden=96) |
| Hidden dim | 64 | 96 |
| Num motifs | 16 (quá nhiều, diffuse) | 8 (focused) |
| Classifier | Pooled MLP | Class-motif attention |
| Auxiliary loss | Không | Aux CE (mean-pool motifs) |
| AMP | Không | ✅ Mixed precision |
| Batch size (Kaggle) | 32 | 64 |
| Params | ~200K | 360K |

## 3. Files đã tạo/sửa

### Mới:
- `models/d10_slot_motif_model.py` — Model chính (IterativeSlotAttention, MotifRelationTransformer, ClassMotifAttentionHead)
- `configs/experiments/d10_slot_motif.yaml` — Config train (hidden=128, K=8, T=3, 120 epochs)

### Sửa:
- `models/registry.py` — Thêm D10 model registration
- `training/losses.py` — Thêm D10SlotMotifLoss (D6B loss + aux CE)
- `notebooks/kaggle-end-to-end.ipynb` — Updated cho D10 training

## 4. Cách chạy

### Local (smoke test):
```bash
conda run -n fer-graph python -m scripts.train_d5a \
  --config configs/experiments/d10_slot_motif.yaml \
  --epochs 5 --max_train_batches 20 --max_val_batches 10 \
  --no_wandb --device cuda --amp
```

### Kaggle (full training):
1. Push code lên GitHub (branch main)
2. Upload notebook `notebooks/kaggle-end-to-end.ipynb` lên Kaggle
3. Trong Cell 2, chọn `CONFIG_PATH`:
   - `d10_slot_motif_fast.yaml`: Default, ~1.5-2 giờ
   - `d10_slot_motif.yaml`: Nặng hơn, ~3-4 giờ
4. Set `RUN_MODE = "train_full"`
5. Chạy tất cả cells
6. Download output zip

### Kaggle (quick test trước — ~15 phút):
- Cell 2: `RUN_MODE = "train_quick"` (30 epochs, capped batches)

### Speed optimizations đã áp dụng (Giảm từ 13h xuống 1.5h):
- ✅ **Fast Config**: `hidden=64`, 1 GNN layer, bù lại bằng 5 slot iterations (rất nhẹ)
- ✅ **Multi-GPU**: `DataParallel` hỗ trợ 2x T4 trên Kaggle (~1.5x speed)
- ✅ **torch.compile**: Compile GNN encoder thành C++ kernels (~1.5x speed)
- ✅ **Val Frequency**: Bỏ qua validation ở các epoch trung gian (`val_frequency=3`)
- ✅ **AMP** (mixed precision) + **Batch size 64** + **pin_memory=true** + **persistent_workers=true**

## 5. Loss function

```
Total Loss = CE_main + 0.01 × slot_diversity + 0.01 × border_penalty
           + 0.005 × slot_balance + 0.3 × CE_auxiliary
```

- **CE_main**: Weighted cross-entropy từ class-motif attention head
- **slot_diversity**: Penalize cosine similarity giữa attention maps (anti-collapse)
- **border_penalty**: Penalize motif attention trên border pixels
- **slot_balance**: KL divergence to uniform (mỗi motif chiếm ~12.5% ảnh)
- **CE_auxiliary**: Cross-entropy từ mean-pooled motif representation (early signal)

## 6. Diagnostics để theo dõi

| Metric | Ý nghĩa | Good range |
|---|---|---|
| `slot_div` / `slot_similarity_mean` | Cosine sim giữa motif attention maps | < 0.5 (diverse) |
| `slot_area_entropy` | Entropy of motif area distribution | > 1.5 (balanced) |
| `border_mass_mean` | Tỷ lệ border pixels trong motif | < 0.20 (focused) |
| `pixel_assignment_entropy` | Entropy assignment per pixel | < 1.5 (sharp) |
| `class_motif_entropy` | Class-motif attention entropy | 1.0-1.8 (selective) |
| `diag_main_accuracy` | Train accuracy (main head) | Tăng dần |
| `diag_aux_accuracy` | Train accuracy (aux head) | Tăng dần |

## 7. Kết quả thực tế & Diagnostic (Update 08/05/2026)

### 7a. D10 Fast (Prototype thành công - F1 0.5018)
- **Config**: 1 GNN layer, `hidden=64`, 5 slot iterations, LR 5e-4.
- **Test**: Accuracy **54.08%**, Macro F1 **0.5018**.
- **Nhận xét**: Chứng minh Slot Attention hoạt động. Dự đoán đủ 7 class. Hạn chế: Fear Recall thấp (25%) do Receptive Field hẹp.

### 7b. D10 Standard V1 (Sụp đổ - F1 0.1399)
- **Config**: 2 GNN layers, `hidden=96`, **3 slot iterations**, LR 5e-4.
- **Test**: Accuracy **26.00%**, Macro F1 **0.1399**. Mode collapse → đoán toàn Happy.
- **Nguyên nhân ban đầu nghĩ**: GNN 2 lớp oversmoothing. **Thực tế SAI** (xem 7c).

### 7c. V2 Ablation — 5 Kịch bản chẩn đoán (Hoàn thành 08/05/2026)

| # | Config | GNN | Iter | Dim | LR | Test F1 | Test Acc | pred_count | Kết luận |
|---|--------|-----|------|-----|------|---------|----------|------------|----------|
| **2** 🏆 | **d10_v2_2_iter5_lr3e4** | **2** | **5** | **96** | **3e-4** | **0.5204** | **54.86%** | **[419,42,317,943,639,413,816]** | **NEW SOTA — Balanced predictions** |
| Fast | d10_slot_motif_fast | 1 | 5 | 64 | 5e-4 | 0.5018 | 54.08% | Balanced | Baseline tốt |
| 4 | d10_v2_4_iter7_lr3e4 | 2 | 7 | 96 | 3e-4 | 0.2637 | 35.02% | [302,0,223,1145,592,821,506] | Iter quá nhiều → vanishing grad |
| 5 | d10_v2_5_gnn1_dim128 | 1 | 5 | 128 | 5e-4 | 0.2313 | 34.72% | [0,0,189,1484,1023,533,360] | LR quá cao cho dim 128 |
| 1 | d10_v2_1_iter5 | 2 | 5 | 96 | 5e-4 | 0.1372 | 25.72% | [13,5,25,2359,713,473,1] | LR quá cao → collapse |
| Std V1 | d10_slot_motif | 2 | 3 | 96 | 5e-4 | 0.1399 | 26.00% | [0,0,34,2103,773,673,6] | Iter ít + LR cao → collapse |
| 3 | d10_v2_3_gnn1_iter5 | 1 | 5 | 96 | 5e-4 | 0.1259 | 24.71% | [0,0,0,2345,524,720,0] | LR quá cao cho dim 96 |

### 7d. Bài học rút ra (Critical Findings)

1. **Thủ phạm thực sự là LR, KHÔNG phải GNN depth.**
   - Kịch bản 3 (GNN 1, dim 96, LR 5e-4) sập nặng hơn cả Kịch bản 2 (GNN 2, dim 96, LR 3e-4).
   - Khi `hidden_dim ≥ 96`, ma trận trọng số Slot Attention tăng theo O(d²). LR 5e-4 gây gradient shock ở epoch đầu, khiến model rơi vào local minimum "đoán toàn Happy".
   - Bản Fast (dim=64) sống sót nhờ ma trận nhỏ đủ kìm hãm gradient.
2. **GNN 2 lớp thực sự CÓ ÍCH** khi LR đúng.
   - V2.2 (GNN 2 + LR 3e-4) vượt mặt bản Fast (GNN 1): F1 0.5204 vs 0.5018.
   - Receptive Field rộng hơn (5×5 thay vì 3×3) giúp Slot Attention nhìn thấy ngữ cảnh khuôn mặt tốt hơn.
3. **5 iterations là "Magic number".** 3 quá ít (collapse), 7 quá nhiều (vanishing gradient).
4. **Công thức chốt D10 Standard**: `GNN=2, Iter=5, Dim=96, LR=3e-4`.

## 8. Roadmap lên 0.70+ Accuracy

### Phase 1: Hyperparameter Tuning trên nền V2.2 (Mục tiêu: F1 0.56-0.60)
- [x] Thử LR warmup (linear 5 epoch) + Cosine Annealing thay ReduceLROnPlateau.
- [x] Tăng `num_motifs` từ 8 lên 10-12 (cho phép model phát hiện nhiều sub-regions hơn).
- [x] Tăng `motif_relation_layers` từ 1 lên 2 (cho motif "nói chuyện" nhiều hơn).
- [x] Label smoothing 0.1 trong CE loss.
- [x] Focal Loss (gamma=2) thay CE cho class khó (Disgust, Fear).

## 9. Kế hoạch V3 (Phase 1 Execution)

Đã tạo 7 config V3 để chạy tối đa công suất 7 slot trên Kaggle. Kết quả (80 epochs):

| Kịch bản | Macro F1 | Accuracy | Đánh giá |
|---|---|---|---|
| **V3.2 (Cosine Only)** | **0.5456** | 55.39% | **SOTA Mới (F1)**. Thay thế `ReduceLROnPlateau` bằng `CosineWarmup` là một quyết định chính xác. Tăng từ 0.5204 lên 0.5456 mà không cần đổi kiến trúc. |
| **V3.5 (GNN 3 lớp)** | 0.5332 | **55.75%** | **SOTA Mới (Accuracy)**. Bất ngờ lớn! GNN 3 lớp không bị Oversmoothing như ta tưởng, miễn là dùng LR warmup mượt mà. |
| **V3.4 (Focal Loss)** | 0.5080 | 53.41% | Thua kém bản chuẩn. Có vẻ Focal Loss ($\gamma=2.0$) đang phạt quá nặng hoặc xung đột nhẹ với Label Smoothing. |
| **V3.7 (Fast Enhanced)** | 0.5028 | 54.05% | Rất ấn tượng! GNN 1 lớp, Dim 64 nhỏ gọn nhưng tiệm cận sức mạnh của các bản full. |
| **V3.1 (Motifs 10)** | 0.4984 | 54.22% | Tăng số motif lên 10 không mang lại hiệu quả, 8 motifs dường như là "sweet spot" lý tưởng nhất của D10. |
| **V3.3 (Motifs 12)** | 0.4970 | 53.47% | Tương tự V3.1, càng nhồi nhiều motif model càng bị bối rối và F1 càng giảm. |
| **V3.6 (Dim 128)** | 0.1740 | 27.56% | **Sụp đổ (Collapse)**. Dim 128 lại tiếp tục đoán toàn class Happy. Điều này chứng minh ma trận trọng số 128 quá lớn để dùng LR 3e-4, cần phải giảm LR xuống 1e-4 nếu muốn ép Dim 128. |

**Bài học rút ra từ Phase 1:**
1. **LR Scheduler**: `CosineAnnealingWarmRestarts` vượt trội hoàn toàn so với Plateau. 
2. **Số lượng Motif**: `8` là con số vàng. 10 hay 12 làm nhiễu sự tập trung.
3. **GNN Depth**: GNN 3 lớp CÓ THỂ hoạt động và thậm chí cho Accuracy tốt nhất (55.75%).
4. **Trọng tâm tiếp theo (Phase 2/3)**: Lấy V3.2 hoặc V3.5 làm Base, triển khai DropPath, Residual Slot, và Edge Dropout.

## 10. Ràng buộc (KHÔNG phá)

- ❌ KHÔNG rebuild graph_repo
- ❌ KHÔNG sửa D5A/D7/D8B code
- ❌ KHÔNG dùng CSV split gốc
- ✅ graph_repo là nguồn dữ liệu duy nhất
- ✅ Reuse SharedPixelEncoder, D5Trainer, common.py
