# D10 Project Handoff — Slot Attention Motif Discovery

> **Last updated**: 2026-05-08
> **Status**: ✅ Code complete, ready to train
> **Model**: D10SlotMotifModel (360K params)
> **Target**: macro F1 ≥ 0.55 (stretch: 0.60+)

## 1. Tóm tắt kiến trúc

```
Pipeline (đúng tinh thần pixel graph → motif → emotion):

  Input: 48×48 pixel graph (2304 nodes, 7D features, 5D edges)
     ↓
  SharedPixelEncoder (2 GNN layers, hidden=96)  ← Deep encoding
     ↓
  IterativeSlotAttention (K=8 motifs, T=3 iterations)  ← Core innovation
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
| Iterations | 1 (one-shot) | 3 (iterative refinement + GRU) |
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

### D10 Fast (Thành công bước đầu - F1 0.5018)
- **Config**: 1 GNN layer, `hidden=64`, 5 slot iterations.
- **Test Metric**: Accuracy **54.08%**, Macro F1 **0.5018** (Vượt xa D9 F1=0.17 và D5A F1=0.29).
- **Nhận xét**: Chạm mốc 0.50 Macro F1 với kiến trúc nhẹ (132K params). Chứng minh Slot Attention đã tự động gom pixel thành các vùng ý nghĩa và dự đoán được đủ 7 class (không chết Disgust/Angry như các bản trước).
- **Hạn chế**: Điểm của class Fear còn thấp (Recall 25%). Nguyên nhân do 1 lớp GNN có Receptive Field quá hẹp (3x3), không bắt được sự phối hợp toàn mặt (mắt + miệng).

### D10 Standard (Sụp đổ - F1 0.1399)
- **Config**: 2 GNN layers, `hidden=96`, 3 slot iterations.
- **Test Metric**: Accuracy **26.00%**, Macro F1 **0.1399** (Sập hoàn toàn).
- **Nhận xét**: Mode collapse, mô hình đoán Happy tới 2103 lần, bỏ qua hoàn toàn Angry và Disgust (0 prediction).
- **Nguyên nhân**: 2 lớp GNN trộn features pixel quá sâu (oversmoothing), làm các vùng mặt mất sự khác biệt. Trong khi đó Slot Attention chỉ chạy 3 vòng (3 iterations) nên không đủ thời gian để tách ngược các motif bị rối này ra. Cùng với LR=0.0005 khá cao khiến model tìm đường tắt (đoán toàn class lớn).

## 8. Kế hoạch V2 (Diagnostic & Fix Oversmoothing)

Đã tạo 5 config V2 trong thư mục `configs/experiments/` để chạy phân tải song song trên Kaggle nhằm tìm ra điểm cân bằng giữa GNN Depth và Slot Iterations:

1. `d10_v2_1_iter5.yaml`: GNN 2, Iter 5, LR 0.0005 (Tăng vòng lặp slot lên 5 để đủ sức gỡ rối features từ GNN 2 lớp).
2. `d10_v2_2_iter5_lr3e4.yaml`: GNN 2, Iter 5, LR 0.0003 (Tăng vòng lặp + hạ LR tránh sốc).
3. `d10_v2_3_gnn1_iter5.yaml`: **[Ưu tiên cao nhất]** GNN 1, Iter 5, `hidden=96` (Về lại kiến trúc 1 lớp GNN thành công của bản Fast, nhưng buff sức mạnh hidden_dim lên 96 để có capacity mạnh hơn).
4. `d10_v2_4_iter7_lr3e4.yaml`: GNN 2, Iter 7, LR 0.0003 (Ép Slot Attention chạy tới 7 vòng).
5. `d10_v2_5_gnn1_dim128.yaml`: GNN 1, Iter 5, `hidden=128` (GNN 1 lớp nhưng buff width lên cực lớn).

**Hành động tiếp theo**: Phân tích kết quả của 5 config này để chốt cấu hình D10 Single-model mạnh nhất. Nếu điểm tiếp cận ngưỡng ~60%, tiến hành Ensemble với D7.

## 9. Ràng buộc (KHÔNG phá)

- ❌ KHÔNG rebuild graph_repo
- ❌ KHÔNG sửa D5A/D7/D8B code
- ❌ KHÔNG dùng CSV split gốc
- ✅ graph_repo là nguồn dữ liệu duy nhất
- ✅ Reuse SharedPixelEncoder, D5Trainer, common.py
