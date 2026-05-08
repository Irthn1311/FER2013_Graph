# D10 Fast — Phân tích kết quả & So sánh D7/D8B

## I. Bảng so sánh tổng thể

| Model | Accuracy | Macro F1 | Weighted F1 | Params | Hướng tiếp cận |
|---|---:|---:|---:|---:|---|
| D5A original | 40.54% | 0.2927 | 0.3587 | ~200K | Random motif |
| D9 (best) | ~22% | 0.167 | — | ~200K | Soft-cluster motif (collapse) |
| **D10 Fast** | **54.08%** | **0.5018** | **0.5295** | 132K | **Slot Attention motif** |
| D7A seed44 | 58.96% | 0.5694 | — | ~200K | Graph-Swin (không motif) |
| D7A standalone | 59.35% | 0.5699 | 0.5874 | ~200K | Graph-Swin region transformer |
| D8B border020 | 60.85% | 0.5907 | 0.6068 | ~200K | Face-aware Graph-Swin |
| D7+D8B ensemble | 64.61% | 0.6342 | 0.6421 | 5×~200K | 5-model prob average |

## II. D10 Fast đã đạt được gì

### ✅ Thành công lớn

1. **Phá vỡ hoàn toàn curse D9**: Từ macro F1 = 0.167 → **0.5018** (+200%). Slot Attention đúng hướng.
2. **Vượt xa D5A**: Từ 40.54% → **54.08%** accuracy, macro F1 từ 0.29 → **0.50**.
3. **7 class đều được predict**: pred_count `[426, 68, 295, 944, 598, 506, 752]` — Disgust có 68 predictions (D5A: 0).
4. **Chỉ 132K params** mà gần bằng D7 (200K), chứng minh Slot Attention hiệu quả.
5. **Train chỉ 70 phút** — rất nhanh.

### ❌ Khoảng cách với D7/D8B

| Gap | Giá trị |
|---|---|
| D10 Fast vs D7A seed44 (accuracy) | -4.88% |
| D10 Fast vs D7A seed44 (macro F1) | -0.0676 |
| D10 Fast vs D8B border020 (accuracy) | -6.77% |
| D10 Fast vs D8B border020 (macro F1) | -0.0889 |

## III. Phân tích per-class: Điểm yếu cụ thể

| Class | D10 Fast F1 | Vấn đề |
|---|---:|---|
| **Fear** | **0.3208** | Yếu nhất. Recall chỉ 25% — 75% Fear bị predict sai (→ Sad, Neutral, Angry) |
| Angry | 0.4035 | Recall thấp (37.7%). Bị nhầm nhiều sang Sad (80), Neutral (85) |
| Sad | 0.4077 | Bị phân tán: 142 → Neutral, 66 → Angry, 64 → Happy |
| Disgust | 0.4390 | Recall tốt (49%) nhưng precision thấp (39.7%) — quá ít mẫu (55) |
| Neutral | 0.5196 | Tạm ổn nhưng bị nhầm sang Sad (104) và Happy (61) |
| Surprise | 0.6659 | Tốt |
| Happy | 0.7559 | Tốt nhất — vẫn là class dễ nhất |

### Pattern confusion chính
```
Fear → phân tán khắp nơi (recall chỉ 25%)
Angry ↔ Sad ↔ Neutral: tam giác nhầm lẫn lớn nhất
Happy vẫn bị over-predict (944 predictions vs 879 actual)
```

## IV. Chẩn đoán: Tại sao D10 Fast chưa bằng D7/D8B

### 1. Pixel Encoder quá nông (Root cause chính)
- D10 Fast: **1 GNN layer, hidden=64** — receptive field chỉ 8 pixels
- D7/D8B: Swin Transformer có **global receptive field** qua window attention
- → Pixel embeddings của D10 Fast thiếu context toàn cục → motifs không đủ discriminative cho Fear/Angry/Sad

### 2. Fear collapse là bằng chứng
- Fear (sợ hãi) cần nhận diện tổ hợp toàn mặt: mắt mở to + miệng há + cơ mặt căng
- Với 1 GNN layer, mỗi pixel chỉ thấy 8 lân cận → không thể nhận ra tổ hợp features xa nhau
- D7 với Swin Transformer nhìn toàn bộ window 6×6 hoặc 4×4 → bắt được pattern toàn cục

### 3. Bản Fast hy sinh quá nhiều encoder capacity
- Đây là bản trade-off tốc độ. Standard config (2 GNN, hidden=96, 360K params) chưa được test
- Dự kiến Standard sẽ cải thiện **5-10%** macro F1

## V. Chiến lược tiếp theo

### Hướng A: Chạy D10 Standard (nhanh nhất, khả thi cao)
```
Config: d10_slot_motif.yaml (360K params, 2 GNN layers, hidden=96)
Dự kiến: Macro F1 ~0.55-0.58, Accuracy ~58-62%
Thời gian: ~3-4 giờ trên Kaggle
```
- Encoder sâu hơn (2 layers) → receptive field 16 pixels
- Hidden lớn hơn (96 vs 64) → embeddings discriminative hơn
- Có thể bằng hoặc vượt D7A single model

### Hướng B: D10 Enhanced (cải thiện kiến trúc)
Nếu Standard vẫn chưa đủ, có thể cải thiện:
1. **3-4 GNN layers** → receptive field 24-32 pixels
2. **Hidden=128** → mạnh hơn
3. **Cosine scheduler** thay ReduceLROnPlateau (ổn định hơn)
4. **Label smoothing** để giảm overfit trên Happy
5. **Focal loss** thay CE cho Fear/Angry (class khó)

### Hướng C: D10 + D7 Ensemble (cao nhất)
- D10 motif-based + D7 Swin-based = **complementary** (motif khác Swin)
- Dự kiến ensemble D10+D7 có thể đạt **68-72%** accuracy

## VI. Kết luận

> [!IMPORTANT]
> D10 Fast (132K params, 1 GNN layer) đạt **54.08% accuracy, macro F1 = 0.50** — **đã chứng minh Slot Attention motif hoạt động** và vượt xa D9.
> Tuy nhiên bản Fast đang bị giới hạn bởi pixel encoder quá nông.
> **Bước tiếp theo ưu tiên**: chạy D10 Standard (2 GNN, hidden=96) để thu hẹp gap với D7/D8B.
