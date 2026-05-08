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

### Kaggle (full training — ~6-9 giờ):
1. Push code lên GitHub (branch main)
2. Upload notebook `notebooks/kaggle-end-to-end.ipynb` lên Kaggle
3. Trong Cell 2, set `RUN_MODE = "train_full"`
4. Chạy tất cả cells
5. Download output zip

### Kaggle (quick test trước — ~1-2 giờ):
- Cell 2: `RUN_MODE = "train_quick"` (30 epochs, capped batches)

### Speed optimizations đã áp dụng:
- ✅ **AMP** (mixed precision): `amp: true` trong config
- ✅ **Batch size 64** trên Kaggle (vs 32 trên local)
- ✅ **pin_memory=true** trên Kaggle
- ✅ **persistent_workers=true** + **prefetch_factor=2**
- ✅ **Model nhẹ**: hidden=96, 2 GNN layers (360K params)
- ✅ **profile_batches=3**: log tốc độ thật để biết thời gian chính xác

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

## 7. Kỳ vọng kết quả

| Phase | Macro F1 | Ghi chú |
|---|---|---|
| Epoch 1-5 | 0.12-0.18 | Model warming up, motifs chưa sắc |
| Epoch 10-20 | 0.25-0.35 | Motifs bắt đầu specialize |
| Epoch 30-50 | 0.35-0.50 | Training stabilizes |
| Epoch 80-120 | 0.50-0.60+ | Best performance |

**Nếu epoch 30 mà F1 < 0.20**: cần xem diagnostics, có thể motif vẫn collapse.

## 8. Ablation plan (sau khi có baseline)

1. **7D vs 3D features**: Nếu 7D gây noise, thử 3D (giống D9)
2. **Hidden dim**: 96 vs 128 vs 192
3. **Num motifs**: 6 vs 8 vs 12
4. **Slot iterations**: 2 vs 3 vs 5
5. **GNN layers**: 2 vs 3 vs 4
6. **Cosine scheduler** thay ReduceLROnPlateau

## 9. Ràng buộc (KHÔNG phá)

- ❌ KHÔNG rebuild graph_repo
- ❌ KHÔNG sửa D5A/D7/D8B code
- ❌ KHÔNG dùng CSV split gốc
- ✅ graph_repo là nguồn dữ liệu duy nhất
- ✅ Reuse SharedPixelEncoder, D5Trainer, common.py
