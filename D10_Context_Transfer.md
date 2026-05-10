# BẢN LƯU TRỮ NGỮ CẢNH DỰ ÁN CHI TIẾT (D10 COMPREHENSIVE CONTEXT TRANSFER)
*Tài liệu này đóng vai trò như một "Bộ não phụ" (Secondary Brain) lưu trữ toàn bộ các thông tin kỹ thuật sâu sắc nhất, lịch sử thử nghiệm, phân tích lỗi, và định hướng mã nguồn của dự án FER-D5 (D10 Model). Khi bắt đầu một phiên làm việc mới, AI cần đọc kỹ toàn bộ tài liệu này trước khi viết code.*

---

## 1. BỐI CẢNH DỰ ÁN & MỤC TIÊU
- **Tên dự án**: D10 Slot Motif Emotion Recognition (FER2013 Pixel Graph).
- **Mục tiêu**: Đạt mốc **Macro F1 từ 0.70 đến 0.80** và **Accuracy > 65%**.
- **Hiện trạng (Tính đến Phase 3 Sweep)**: F1 kịch trần ở mức **0.545**, Accuracy đạt **55.87%**. Mô hình hiện tại đang gặp nút thắt (bottleneck) về khả năng biểu diễn ngữ nghĩa của các Motif, dẫn đến nhầm lẫn giữa các class khó (Fear vs Sad, Disgust vs Angry).

## 2. BẢN ĐỒ KIẾN TRÚC MÃ NGUỒN (CODEBASE MAP)
Dự án được cấu trúc theo dạng Modular, các file quan trọng nhất nằm ở:
1. `models/d10_slot_motif_model.py`: Chứa class `D10SlotMotifModel` và `IterativeSlotAttention`. Đây là trái tim của dự án. Mọi thay đổi về Data Flow, Graph Augmentation, Slot Refinement đều thực hiện ở đây.
2. `models/edge_gnn.py`: Chứa `EdgeAwarePixelGNNEncoder` (Base của `SharedPixelEncoder`). Trích xuất đặc trưng từng pixel dựa trên Edge Index (8-neighbor grid).
3. `training/losses.py`: Chứa `BaseD6Loss`, `D10SlotMotifLoss`, `WeightedCrossEntropy` và `FocalLoss`. Loss function là sự kết hợp của CrossEntropy và các auxiliary losses (Border Loss, Slot Smoothness).
4. `training/optimizer.py`: Nơi khai báo `CosineWarmup` (CosineAnnealingWarmRestarts + Linear Warmup) - Scheduler đang cho kết quả tốt nhất.
5. `scripts/evaluate_d5a.py` & `scratch/eval_v3.py`: Các script đánh giá checkpoint. Cần chú ý file `metrics.json` nằm trong thư mục `evaluation/` của từng run.
6. `configs/experiments/d10_*.yaml`: Toàn bộ cấu hình siêu tham số (Hyperparameters) được định nghĩa ở đây.

## 3. CHI TIẾT VỀ KIẾN TRÚC D10 (D10 ARCHITECTURE DEEP DIVE)
### 3.1. Deep Pixel Encoder (GNN)
- Biến bức ảnh 48x48 thành đồ thị gồm `2304 nodes` (pixels).
- Liên kết bằng `edge_index` (cạnh) với các pixel lân cận, và có `edge_attr` (khoảng cách vật lý).
- **Cải tiến Multi-scale GNN (Phase 3)**: Đã thêm một nhánh `encoder_aux` (nhiều hơn 1 lớp GNN). Kết quả của nhánh GNN 2 lớp và GNN 3 lớp được `concat` và đi qua `combine_scale` (Linear + LayerNorm + GELU). Điều này đã đẩy Accuracy lên SOTA mới 55.87%.

### 3.2. Iterative Slot Attention (Motif Discovery)
- Khởi tạo $K$ Motif slots (Hiện tại $K=8$) bằng `torch.randn`.
- **Luồng dữ liệu**: Pixel feature làm Key/Value, Slots làm Query.
- **Sự khác biệt quan trọng**: Softmax được tính theo chiều của Slots (`dim=1`), nghĩa là các Slots phải **cạnh tranh** (compete) để giành lấy từng Pixel. Điều này ép các Motifs không bị đè lên nhau (non-overlapping).
- Các pixel features được gom lại bằng Attention Map và dùng GRU để cập nhật trạng thái của Slot.
- **Cải tiến Residual (Phase 3)**: Đã thêm `slots = slots + slots_prev` sau hàm GRU để chống hiện tượng triệt tiêu Gradient qua 3 vòng lặp.

### 3.3. Cross-Attention Slot Refinement (Transformer Decoder)
- Nằm ngay sau quá trình Slot Attention.
- Thay vì bị giới hạn bởi Softmax cạnh tranh, ta cho các Motif Slots đóng vai trò làm Query, và toàn bộ Pixel Feature ban đầu làm Key/Value qua một lớp `nn.MultiheadAttention` tiêu chuẩn.
- Bổ sung thông tin không gian (Global Context) một lần cuối trước khi đưa vào Classifier.

### 3.4. Motif Relation Transformer & Classifier
- Các Motifs giao tiếp với nhau qua Self-Attention để hiểu ngữ cảnh (VD: "Miệng cười" đi với "Mắt nhăn" -> Happy).
- Cuối cùng được tính tổng trọng số (`weighted sum`) bằng `class_queries` (Learnable Parameters) để đưa ra logits cho 7 cảm xúc.

## 4. HỒ SƠ CÁC LỖI NGHIÊM TRỌNG (BUGS MORTEM)
**Để tránh việc AI mất thời gian dò dẫm lại, đây là các lỗi cốt lõi đã được giải quyết:**
1. **Lỗi Memory Pinned / DataParallel với `edge_index`**: 
   - `DataParallel` tự động cắt Tensor ở chiều $B$ (Batch size). Nhưng `edge_index` của Pixel Graph ban đầu có shape `[2, E]`, không có chiều Batch, dẫn đến bị cắt sai bét -> Lỗi `IndexError: out of bounds`.
   - **Giải pháp**: Trong Collate Function của DataLoader, ta ép `edge_index` thành shape `[B, 2, E]` bằng `.unsqueeze(0).expand(bsz, -1, -1).contiguous()`. Sau đó vào `D10SlotMotifModel.forward`, ta "nhả" nó về lại `[2, E]` bằng `edge_index = edge_index[0]`.
2. **Lỗi Missing Manifest trong Hierarchical Cache (V3)**:
   - Các file `manifest.json` không tự sinh ra đúng thư mục Artifact Root, làm validation data loader bị crash. Fix bằng việc chuẩn hóa đường dẫn ở `experiment_runner.py`.
3. **Lỗi Argparse `--num_workers`**:
   - Truyền tham số vào script đánh giá bị lỗi do Parser chưa khai báo. Đã fix gọn gàng.

## 5. BÁCH KHOA TOÀN THƯ THỬ NGHIỆM (EMPIRICAL LEARNINGS)
Đây là dữ liệu vô giá được đúc kết từ hơn 20 lần train trên Kaggle:

### 🟢 Những gì đã chứng minh là HIỆU QUẢ:
- **CosineAnnealingWarmRestarts (`CosineWarmup`)**: Vượt trội hơn hẳn `ReduceLROnPlateau`. Graph Model cần sự thay đổi LR nhịp nhàng liên tục thay vì bậc thang.
- **8 Motifs (Slots)**: Đây là `Sweet Spot`. Thử nghiệm 10 hay 12 Motifs khiến F1 tụt xuống 0.49 do mô hình bị loạn thị, chia nhỏ khuôn mặt một cách không cần thiết.
- **Multi-scale GNN**: Đỉnh cao kiến trúc hiện tại, giúp model nhận thức được cả chi tiết nhỏ (GNN 2 lớp) và mảng lớn (GNN 3 lớp). Đạt SOTA Accuracy 55.87%.
- **Epochs = 80**: Mô hình hội tụ rất đẹp ở epoch 80. Kéo dài lên 120 không mang lại giá trị đột phá mà tốn tài nguyên.

### 🔴 Những gì THẤT BẠI thảm hại (Tử Hình Kỹ Thuật):
- **Graph Regularization (Edge Dropout & Node Noise)**: Ý tưởng ban đầu là chống Overfitting. Nhưng thực tế D10 bị **Underfitting** cấu trúc Pixel Graph. Việc đứt gãy 10% kết nối (Edge Dropout) hoặc nhiễu Gaussian ở Node làm F1 rớt thẳng đứng xuống 0.42 - 0.48. **TUYỆT ĐỐI KHÔNG SỬ DỤNG** regularization trực tiếp lên đồ thị trong dự án này.
- **Dim = 128 với Learning Rate 3e-4**: Dẫn tới **Mode Collapse** (Mô hình chỉ nhắm mắt đoán bừa class 3 - Happy). Nếu muốn dùng Hidden Dim 128, LR phải cực nhỏ (`<= 1e-4`).
- **Focal Loss ($\gamma=2.0$)**: Trái với lý thuyết trị mất cân bằng, Focal Loss trừng phạt class dễ quá tay, khiến tổng F1 rớt từ 0.54 xuống 0.50. Cần cẩn trọng khi tinh chỉnh weights.

## 6. QUY TRÌNH SWEEP ĐANG DIỄN RA (PHASE 3: 10 SLOTS)
Người dùng đang sử dụng 10 slots Kaggle để vét cạn mọi tiềm năng của kiến trúc `Multi-scale + Cross Attention Refinement`. Danh sách quét bao gồm:
1. `d10_p3_1_refinement`: Base chuẩn.
2. `d10_p3_2_refinement_sqrt_weights`: Căn bậc hai Class Weights.
3. `d10_p3_3_no_weights`: Bỏ hẳn Class Weights.
4. `d10_p3_4_focal_gamma1`: Focal mềm $\gamma=1.0$.
5. `d10_p3_5_iter5`: 5 vòng Slot Attention.
6. `d10_p3_6_no_dropout`: Thả rông Dropout = 0.
7. `d10_p3_7_motif6`: Giảm còn 6 Motifs.
8. `d10_p3_8_dim128_low_lr`: Dim to 128 + LR nhỏ 1e-4.
9. `d10_p3_9_deep_gnn`: GNN sâu 3+4 lớp.
10. `d10_p3_10_no_pos_enc`: Bỏ Position Encoding.

## 7. ĐỊNH HƯỚNG TƯƠNG LAI (PHASE 5: SUPCON & TWO-STAGE LEARNING)
*Lưu ý: Đã chính thức gạt bỏ Phase 4 (Knowledge Distillation) vì D7 (Teacher) chưa đủ mạnh để đẩy D10 lên 0.70+. Hướng đi duy nhất là tự đào tạo Motif thông qua Contrastive Learning.*

**Kế hoạch bứt phá mốc 0.70 F1:**
Thỏa mãn quy tắc vàng: *"Học motif trước, rồi mới học classifier/relation"*. Ta sẽ viết lại hàm `_contrast_loss` bằng InfoNCE Loss và chia làm 2 giai đoạn:

- **Stage 1: Ép Cụm Khái Niệm (SupCon) - 120 Epochs**
  - **Action**: Bổ sung cờ `freeze_classifier: true` trong config.
  - **Logic**: Đóng băng `MotifRelationTransformer` và `Classifier`. Mô hình chỉ tính Loss qua hàm `SupCon Loss` (ép các Motif Vector của cùng class dính chặt vào nhau và đẩy các class khác ra xa). Mô hình sẽ tập trung 100% tài nguyên để tạo ra không gian ngữ nghĩa (Semantic) hoàn hảo cho các Motif.

- **Stage 2: Học Phân Loại (Semantic Relation) - 120 Epochs**
  - **Action**: Load checkpoint tốt nhất của Stage 1. Đặt cờ `freeze_encoder: true`.
  - **Logic**: Khóa toàn bộ tầng GNN và Slot Attention. Mở khóa tầng `MotifRelation` và `Classifier` để train bằng `CrossEntropy`. Nhờ đầu vào Motif đã được tách biệt rõ rệt (cluster hóa), bộ phân loại sẽ ngay lập tức vẽ được ranh giới siêu phẳng giữa các cảm xúc khó như Fear và Disgust.

*Chỉ thị cho AI phiên tới:* Khi 10 bản Sweep có kết quả, chọn bản có siêu tham số tốt nhất (vd: Sqrt Weights hoặc Dim 128) để làm nền tảng viết code Phase 5!

## 8. BÁCH KHOA TOÀN THƯ THỬ NGHIỆM TỪ PHASE 5 (TWO-STAGE SUPCON)
Phase 5 là một hành trình dài và đầy biến động. Mục tiêu ban đầu là giải quyết 2 vấn đề:
1. **Slot Collapse**: Các slot bị trùng lặp, không học được đặc trưng riêng.
2. **Overfitting**: Gap giữa Train và Val F1 quá lớn (~0.27).

### 8.1 Lịch Sử Các Lần Chạy (Runs) & Kết Quả

| Run | Cấu hình & Tinh chỉnh | Kết quả (Test F1) | Phân tích cốt lõi |
|---|---|---|---|
| **Run 1** (Baseline Two-Stage) | Mean-pooled SupCon ở Stage 1.<br>Stage 2: `freeze_encoder: false`, không có anti-overfitting (dropout=0.2). | **0.6130** | Vượt mốc 0.61 nhưng bị overfitting rất nặng (Train F1 0.87 vs Val F1 0.60). GNN và Slot Attention thực chất bị train lại từ đầu do `freeze_encoder: false`. |
| **Run 2** (Per-Slot SupCon) | Code `losses.py` đổi thành **Per-slot SupCon**. Thêm `slot_div: 0.1`.<br>Stage 2: `freeze_encoder: true`, heavy reg (dropout=0.4, label_smoothing=0.1, weight_decay=0.01). | **0.4289** | **Thảm họa**. Việc `freeze_encoder: true` khiến classifier không thể nhận được đặc trưng tốt từ encoder. Anti-overfit quá mạnh bóp nghẹt mô hình. |
| **Run 2** (Joint) | Train 1 mạch CE + Per-slot SupCon đồng thời. Heavy reg tương tự Run 2. Bị crash do `torch.compile` + GRU. | **0.3541** | Sụp đổ hoàn toàn do regularization quá đà và xung đột quá nhiều loại loss cùng lúc. |
| **Run 3** (Standard & MinReg) | Stage 1 giữ nguyên Per-slot SupCon.<br>Stage 2: Quay lại `freeze_encoder: false`. Thử reg nhẹ (dropout=0.3) và MinReg (giống hệt Run 1). | **0.4377** - **0.4417** | **Bằng chứng chí mạng**. Run 3 MinReg giống hệt Run 1 về mọi mặt ở Stage 2, chỉ khác là checkpoint Stage 1 dùng Per-Slot SupCon thay vì Mean-Pooled. F1 rớt từ 0.6130 xuống 0.4377. |

### 8.2 Chẩn Đoán Cốt Lõi: Tại Sao Per-Slot SupCon Phá Hủy Encoder?
Ý tưởng Per-Slot SupCon là ép *từng slot* (từng vùng của khuôn mặt) phải gom cụm theo biểu cảm cảm xúc (label).
Tuy nhiên, điều này **sai về mặt ngữ nghĩa hình ảnh**:
- Không phải mọi vùng trên khuôn mặt đều chứa cảm xúc. Ví dụ: Slot 1 focus vào *trán* hay *cổ*. Trán/cổ của người buồn và người vui trông giống hệt nhau.
- Khi ép Slot 1 phải gom cụm theo cảm xúc, Contrastive Loss đã sinh ra các gradient "rác", ép encoder tìm ra các pattern ảo không tồn tại.
- Hậu quả: Encoder rơi vào local minimum xấu. Representations bị phá vỡ hoàn toàn, khiến Stage 2 (dù có unfreeze) cũng không thể cứu vãn được.

**Khác biệt với Mean-Pooled SupCon (Run 1):**
Mean-pooled (trung bình 8 slot) nhìn vào *tổng thể khuôn mặt*. Loss chỉ ép tổng thể này gom cụm, do đó mạng nơ-ron tự do chọn vùng nào để focus mà không bị ép uổng.

### 8.3 Con Đường Khắc Phục (Hướng Tới Run 4)
Dựa trên bằng chứng sắt đá từ Run 3 MinReg, chúng ta đã đưa ra 3 quyết định kỹ thuật tối quan trọng:
1. **REVERT Code**: Quay về `Mean-Pooled SupCon` trong `losses.py`. Per-slot SupCon chính thức bị loại bỏ.
2. **Khôi phục Checkpoint**: Stage 1 của Run 1 (Mean-Pooled) là checkpoint tốt nhất hiện tại. Các lần train Stage 2 tiếp theo **BẮT BUỘC** phải dùng checkpoint này.
3. **Bài toán duy nhất còn lại - Anti-Overfitting**: Vì Run 1 bị overfit, hướng đi duy nhất là chạy Stage 2 (dựa trên checkpoint Run 1) với Regularization **vừa phải** (nằm giữa sự lỏng lẻo của Run 1 và sự khắt khe của Run 2):
   - `dropout`: 0.25 - 0.30 (thay vì 0.2 hoặc 0.4)
   - `label_smoothing`: 0.05 - 0.08 (thay vì 0.0 hoặc 0.1)
   - `weight_decay`: 0.0005 - 0.001 (thay vì 1e-4 hoặc 1e-2)
   - `freeze_encoder`: false (vẫn cho phép GNN/Slot fine-tune).
