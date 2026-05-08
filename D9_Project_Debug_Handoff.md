# FER-2013 Pixel Graph / Motif-GNN — D9 Debug & Research Handoff

> Mục đích tài liệu: cung cấp đầy đủ bối cảnh hiện tại cho AI coding agent/Codex để hiểu dự án, các thí nghiệm đã chạy, các lỗi/bottleneck đang gặp, và hướng xử lý tiếp theo.  
> Trọng tâm hiện tại: **khai thác D9** theo đúng ý tưởng pixel graph → motif → emotion, nhưng các run D9 hiện đang thấp và cần debug có cơ sở.

---

## 1. Tổng quan dự án

### 1.1. Chủ đề

Dự án nghiên cứu nhận diện biểu cảm khuôn mặt trên FER-2013 bằng hướng **Pixel Graph + Motif-Guided Learning / GNN**.

Ý tưởng nghiên cứu cốt lõi:

```text
Ảnh 48x48 grayscale
→ chuyển thành pixel graph 2304 nodes
→ học motif biểu cảm như vùng mắt, lông mày, miệng, khóe miệng, nếp nhăn / biên sáng tối
→ học quan hệ giữa các motif
→ dự đoán emotion toàn khuôn mặt
```

Mục tiêu không chỉ là tăng accuracy, mà còn muốn mô hình đúng tinh thần thầy hướng dẫn: **motif phải có ý nghĩa**, không chỉ là CNN/Swin-like classifier trá hình.

### 1.2. Dataset

- Dataset: **FER-2013**.
- Ảnh grayscale 48x48.
- 7 lớp emotion:
  1. Angry
  2. Disgust
  3. Fear
  4. Happy
  5. Sad
  6. Surprise
  7. Neutral

Split graph repo hiện dùng:

```text
train = 28709
val   = 3589
test  = 3589
```

### 1.3. Graph representation

Mỗi ảnh là một graph:

```text
nodes: 2304 pixels
edge_index: [2, E]
E khoảng 17860
```

Node features gốc 7D:

```text
0. intensity
1. x_norm
2. y_norm
3. gx
4. gy
5. grad_mag
6. local_contrast
```

Edge features 5D:

```text
0. dx
1. dy
2. dist
3. delta_intensity
4. intensity_similarity
```

Graph repo hiện có sẵn, **không rebuild**:

Local thường là:

```text
artifacts/graph_repo
```

Kaggle input thường là:

```text
/kaggle/input/datasets/irthn1311/graph-repo/graph_repo
```

Khi chạy Kaggle, nên copy sang:

```text
/kaggle/working/graph_repo
```

Rồi mọi train/evaluate/generate teacher probs dùng `/kaggle/working/graph_repo`.

---

## 2. Ràng buộc kỹ thuật hiện tại

### 2.1. Không được làm

```text
KHÔNG rebuild graph_repo nếu không thay đổi schema dữ liệu.
KHÔNG dùng CSV split gốc làm input train/evaluate hiện tại.
KHÔNG dùng CSV order để generate teacher probs.
KHÔNG sửa D7/D8B nếu không cần.
KHÔNG sửa train_d5a.py/evaluate_d5a.py nếu không cần.
KHÔNG train full 200 epoch mù khi D9 chưa có tín hiệu.
KHÔNG dùng val_accuracy làm best checkpoint cho classification.
KHÔNG dùng val_loss làm early stopping chính cho classification.
```

### 2.2. Metric policy

Với các run classification:

```text
best checkpoint = val_macro_f1
mode = max
early stopping nếu có cũng monitor val_macro_f1
```

Với motif discovery stage thuần:

```text
best checkpoint có thể monitor motif_quality_score / motif_quality
```

### 2.3. Lý do không dùng CSV hiện tại

CSV gốc đã dùng trước đây để build graph_repo. Hiện tại graph_repo là artifact trung tâm. Teacher probs, D9 student, D7/D8 teacher đều phải dùng cùng graph_repo split/order/index.

Sai lầm cần tránh:

```text
teacher_probs theo CSV order
D9 train theo graph_repo order
→ sample_idx lệch
→ distillation sai ngầm
```

---

## 3. Các hướng/bản đã triển khai trước D9

### 3.1. D6B

Một nhánh motif/class-part graph cũ, đạt khoảng:

```text
accuracy ≈ 54.08%
macro F1 ≈ 0.4985
Fear F1 ≈ 0.3698
```

Ý nghĩa: có performance khá hơn các prototype motif đầu, nhưng vẫn chưa đạt mục tiêu cao.

### 3.2. D7

Các bản D7 theo hướng graph + Swin/region transformer mạnh hơn. Một số kết quả quan trọng:

#### D7 window4

Run name:

```text
d7a_graph_swin_region_transformer_window4
```

Metric nhớ được:

```text
accuracy ≈ 0.6105
macro F1 ≈ 0.5837
weighted F1 ≈ 0.6049
```

#### D7 seed44

Run name:

```text
d7a_graph_swin_region_transformer_seed44
```

Metric:

```text
accuracy ≈ 0.6069
macro F1 ≈ 0.5883
```

#### D7 long150 resume

Run name:

```text
d7a_graph_swin_region_transformer_long150_resume
```

Được dùng như teacher candidate trong ensemble.

### 3.3. D8B

Các bản face-aware graph Swin / region-like.

#### D8B border020

Run name:

```text
d8b_face_aware_graph_swin_border020
```

Metric seed42 nhớ được:

```text
accuracy ≈ 0.6085
macro F1 ≈ 0.5907
weighted F1 ≈ 0.6068
```

Seed repeat:

```text
seed43: accuracy ≈ 0.5943, macro F1 ≈ 0.5751
seed44: accuracy ≈ 0.5940, macro F1 ≈ 0.5622
```

#### D8B area045

Run name:

```text
d8b_face_aware_graph_swin_area045
```

Có dùng trong ensemble mạnh nhất. Metric single riêng chưa chắc chắn trong memory.

### 3.4. D7/D8B ensemble

Ensemble graph-only tốt nhất đã có:

```text
D7 seed44
D7 long150_resume
D7 window4
D8B border020
D8B area045
```

Output name nhớ được:

```text
output/d7_d8b_ensemble_seed44_long150_window4_border020_area045_probavg
```

Metric:

```text
accuracy ≈ 0.6461
macro F1 ≈ 0.6342
weighted F1 ≈ 0.6421
```

Đây là hướng performance tốt nhất hiện tại, nhưng bị xem là lệch dần khỏi ý tưởng motif-GNN của thầy vì càng giống region/window/Swin-like classifier.

---

## 4. Mục tiêu riêng của D9

D9 được phát triển để quay lại đúng ý tưởng của thầy:

```text
pixel graph
→ motif biểu cảm có ý nghĩa
→ quan hệ motif
→ emotion
```

Mục tiêu D9 không chỉ là điểm cao, mà còn:

```text
motif phải có ý nghĩa
motif nhìn vào vùng mặt/mắt/miệng/lông mày/khoé miệng/nếp nhăn
motif không được chỉ là border/tóc/background
motif phải ảnh hưởng thật đến prediction
```

---

## 5. Feature audit D9-F0

Đã khảo sát feature masking cho motif discovery D9-F0:

```text
F0-A: intensity-only + full edge
F0-B: intensity + x/y
F0-C: intensity + gx/gy
F0-D: intensity + gx/gy + grad_mag
F0-E: no x/y
F0-F: full current feature
```

Kết luận:

- **B = intensity + x/y** có border/outer thấp nhất và foreground cao nhất trong nhóm D9-F0.
- **E = no x/y** clean-count tốt hơn trong D9, nhưng border/redundancy còn vấn đề.
- **F = full 7D current feature** không tự nhiên tốt hơn, selected_foreground thấp và clean thấp.

Vì vậy D9 hiện dùng Feature B để phát triển:

```yaml
feature_ablation:
  enabled: true
  node_indices: [0, 1, 2]
  edge_indices: [0, 1, 2, 3, 4]

model:
  node_dim: 3
  edge_dim: 5
```

Lưu ý:

```text
D7/D8B mạnh trước đó có thể dùng 7D.
D9 hiện dùng 3D vì 3D tốt hơn cho motif geometry/debug, không có nghĩa 3D chắc chắn tốt nhất cho mọi classifier.
```

---

## 6. D9 kiến trúc đã triển khai

### 6.1. D9-RG-MR-B

Run chính đầu tiên:

```text
D9-RG-MR-B
= Relation-aware Pixel Encoder
+ Motif Discovery
+ Motif Relation Classifier
+ Feature B
```

Pipeline:

```text
x [B,2304,7]
→ runtime mask [B,2304,3]
→ EdgeAwarePixelEncoder
→ h_pixel [B,2304,H]
→ MotifDiscoveryModule
→ motif_maps / motif_embeddings / selection_weights
→ D9MotifRelationClassifier KxK
→ logits [B,7]
```

Files đã tạo trong repo:

```text
models/d9_relation_encoder.py
models/d9_motif_relation_classifier.py
models/d9_relation_motif_model.py
scripts/train_d9_relation_motif.py
scripts/visualize_d9_motifs.py
configs/experiments/d9_rg_mr_b_intensity_xy_full_edge.yaml
```

### 6.2. D9-RG-MR-B diagnostic

Output:

```text
outputs/d9_rg_mr_b_diag20_cap200v100
```

Kết quả:

```text
best epoch = 5
best val_macro_f1 = 0.0552995
val_accuracy = 0.24
val_weighted_f1 = 0.0929032
```

Vấn đề:

```text
Toàn bộ 800 val samples predict Happy.
6/7 class F1 = 0.
Class collapse rất rõ.
```

Kết luận:

```text
D9-RG-MR-B gốc không học classification thật.
Motif Relation Classifier KxK là nghi phạm lớn.
```

---

## 7. D9 ablations sau khi D9-RG-MR-B collapse

### 7.1. Sanity global pool

Mục tiêu: kiểm data/training loop có chết không.

Kết quả:

```text
val_macro_f1 ≈ 0.1249
val_accuracy ≈ 0.1806
val_weighted_f1 ≈ 0.1295
```

Kết luận:

```text
Data/training loop không chết hoàn toàn.
D9 architecture/motif pipeline mới là vấn đề chính.
```

### 7.2. D9-RG-B no-MR pooled MLP cap200

Thay MR KxK bằng pooled MLP trên motif embeddings.

Kết quả:

```text
best epoch = 9
val_macro_f1 = 0.147594
val_accuracy = 0.2275
val_weighted_f1 = 0.169153
```

Pred distribution:

```text
{0: 74, 3: 111, 4: 383, 5: 231, 6: 1}
```

Per-class F1:

```text
Angry    0.1780
Disgust  0.0
Fear     0.0
Happy    0.2310
Sad      0.3228
Surprise 0.2866
Neutral  0.0147
```

Kết luận:

```text
Pooled MLP tốt hơn MR KxK rất rõ.
MR KxK không nên dùng làm head chính.
Nhưng model vẫn yếu, Disgust/Fear chết.
```

### 7.3. No-relation pooled MLP cap200

Bỏ EdgeAwarePixelEncoder, chỉ node projection → motif → pooled MLP.

Kết quả:

```text
val_macro_f1 = 0.1606
val_accuracy = 0.2125
val_weighted_f1 = 0.1628
```

Kết luận:

```text
Ở cùng cap200, no-relation thắng RG pooled MLP.
Relation encoder chưa chứng minh có ích; có thể gây nhiễu/oversmoothing.
```

### 7.4. RG pooled MLP diag20 cap300

Output:

```text
outputs/d9_rg_b_no_mr_pooled_mlp_diag20_cap300v100
```

Kết quả:

```text
best epoch = 18
val_macro_f1 = 0.1674
val_accuracy = 0.2213
val_weighted_f1 = 0.1984
```

Đây là baseline D9 pooled MLP no-teacher tốt nhất hiện tại.

### 7.5. MR-v2 residual

Ý tưởng:

```text
logits = pooled_logits + alpha * relation_logits
```

Kết quả:

```text
val_macro_f1 = 0.1347
val_accuracy = 0.2100
val_weighted_f1 = 0.1556
```

Kết luận:

```text
MR-v2 residual vẫn kém pooled MLP.
Relation giữa motif chưa giúp khi motif representation còn yếu.
```

---

## 8. D9-TGMS teacher-guided distillation

### 8.1. Mục tiêu

D9-TGMS-B:

```text
Teacher-Guided Motif Semantic Graph
```

Ý tưởng:

```text
D7/D8B teacher không dạy motif map.
Teacher chỉ dạy semantic soft labels/probabilities.
D9 student vẫn học bằng pixel graph → motif → pooled MLP.
```

Loss:

```text
total_loss = CE + alpha * KL(student || teacher) + motif_aux_weight * motif_loss
```

Teacher candidates:

```text
d7a_graph_swin_region_transformer_long150_resume
d7a_graph_swin_region_transformer_window4
d7a_graph_swin_region_transformer_seed44
d8b_face_aware_graph_swin_border020
d8b_face_aware_graph_swin_area045
```

Teacher checkpoints Kaggle pattern:

```text
/kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-long150-resume/best.pth
/kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-window4/best.pth
/kaggle/input/datasets/irthn1311/d7a-graph-swin-region-transformer-seed44/best.pth
/kaggle/input/datasets/irthn1311/d8b-face-aware-graph-swin-border020/best.pth
/kaggle/input/datasets/irthn1311/d8b-face-aware-graph-swin-area045/best.pth
```

### 8.2. Core distillation implementation

Files liên quan:

```text
scripts/generate_teacher_probs.py
scripts/train_d9_relation_motif.py
configs/experiments/d9_tgms_b_distill_a05_t2.yaml
configs/experiments/d9_tgms_b_distill_a02_t2.yaml
docs/d9_tgms_distillation.md
```

Đã thêm:

```text
sample_idx / global_index vào FullGraphDataset.
Teacher probs load theo sample_idx.
teacher_labels[sample_idx] == batch['y'] check bắt buộc.
Nếu mismatch thì raise.
```

Dummy smoke đã pass.

### 8.3. TGMS alpha 0.5

Output:

```text
outputs/d9_tgms_b_distill_a05_t2_cap300v100_outputs
```

Kết quả:

```text
best_epoch = 11
best_val_macro_f1 = 0.15991496495899107
val_accuracy = 0.23625
val_weighted_f1 = 0.1993192267114996
```

Per-class F1:

```text
Angry    0.0091324200913242
Disgust  0.0
Fear     0.10119047619047619
Happy    0.36007827788649704
Sad      0.1731958762886598
Surprise 0.28040540540540543
Neutral  0.19540229885057472
```

Pred distribution:

```text
[4, 0, 99, 639, 216, 401, 241]
```

Loss scale:

```text
CE ≈ 2.0059
KL ≈ 4.3486
alpha*KL ≈ 2.1743
(alpha*KL)/CE ≈ 1.084
```

Kết luận:

```text
alpha 0.5 quá mạnh, distillation pressure lớn hơn CE.
Angry gần như chết.
Không vượt no-teacher.
```

### 8.4. TGMS alpha 0.2

Output:

```text
outputs/d9_tgms_b_distill_a02_t2_cap300v100_outputs
```

Kết quả:

```text
best_epoch = 23
best_val_macro_f1 = 0.16418252761507132
val_accuracy = 0.241875
val_weighted_f1 = 0.20171634558782373
```

Per-class F1:

```text
Angry    0.13714285714285715
Disgust  0.0
Fear     0.06228373702422145
Happy    0.3781362007168459
Sad      0.1649048625792812
Surprise 0.2777777777777778
Neutral  0.12903225806451613
```

Pred distribution:

```text
[135, 0, 52, 733, 204, 385, 91]
```

Loss scale:

```text
CE ≈ 1.9546
KL ≈ 4.4192
alpha*KL ≈ 0.8838
(alpha*KL)/CE ≈ 0.452
```

Kết luận:

```text
alpha 0.2 tốt hơn alpha 0.5.
Nhưng vẫn không vượt no-teacher D9 pooled MLP 0.1674.
Disgust vẫn chết.
```

### 8.5. TGMS analysis report

Files:

```text
outputs/d9_tgms_distill_analysis_report.md
outputs/d9_tgms_distill_analysis_summary.csv
scripts/analyze_d9_tgms_distill_runs.py
```

Kết luận phân tích:

```text
Hai run hợp lệ.
Teacher probs thật sự được dùng vì train_distill_loss non-zero và train_teacher_conf_mean ≈ 0.72.
Teacher metrics không có trong extracted artifact vì teacher_probs folder không được bundle vào output run.
alpha 0.2 tốt hơn alpha 0.5.
Không run nào vượt no-teacher macro F1.
Distillation tăng nhẹ accuracy/weighted F1 nhưng không tăng macro F1.
Class khó không cải thiện, Disgust vẫn F1=0.
Motif metrics vẫn xấu: foreground=0, border≈0.30, entropy cao, effective count gần 16.
```

---

## 9. Vấn đề cốt lõi hiện tại của D9

Sau toàn bộ các thí nghiệm, nguyên nhân gốc nhất được xác định:

```text
D9 thấp vì motif discovery chưa tạo được motif emotion-discriminative.
```

Chuỗi nguyên nhân:

```text
Motif maps diffuse / foreground=0 / border cao / entropy cao
→ motif embeddings không đại diện rõ mắt-miệng-lông mày-khoé miệng
→ pooled MLP chỉ học được tín hiệu yếu cho class dễ
→ MR relation head học nhiễu và collapse
→ teacher semantic loss chỉ kéo logits, không sửa motif
→ class khó như Disgust/Fear/Neutral vẫn chết
→ macro F1 quanh 0.16
```

Các dấu hiệu định lượng quan trọng:

```text
selected_foreground = 0.0
selected_border ≈ 0.30
entropy ≈ 2.74–2.76
effective_count ≈ 15.5–15.7 / 16
motif_loss gần hằng số ≈ 0.025
Disgust F1 = 0.0 trong hầu hết các run D9
```

Ý nghĩa:

```text
Không phải D9 thiếu classifier mạnh.
Không phải thiếu teacher.
Không phải chỉ cần train lâu hơn.
D9 đang thiếu motif có ý nghĩa.
```

---

## 10. Những lỗi notebook/Kaggle đã gặp

### 10.1. Manifest thiếu node_dim/edge_dim

Graph repo đúng nhưng `manifest.pt` không có key trực tiếp:

```text
node_dim
edge_dim
```

Lỗi cũ:

```python
if int(manifest.get('node_dim', -1)) != 7: raise
```

Kết luận:

```text
Đây là false alarm do check metadata quá cứng.
Không rebuild graph_repo.
Sửa thành warning nếu manifest thiếu node_dim/edge_dim.
Runtime/config sẽ check shape thật.
```

### 10.2. reader undefined

Có lỗi:

```text
NameError: reader is not defined
```

Do cell verify dùng:

```python
reader.split_size(...)
```

nhưng chưa tạo `reader`.

Cách sửa:

```text
Không dùng reader trong verify cell.
Chỉ cần check train/val/test folder tồn tại.
Exact split size để Dataset/DataLoader kiểm sau.
```

### 10.3. Path variables lẫn lộn

Các biến cần thống nhất:

```python
GRAPH_REPO_INPUT   = Path('/kaggle/input/datasets/irthn1311/graph-repo/graph_repo')
GRAPH_REPO_WORKING = Path('/kaggle/working/graph_repo')
GRAPH_REPO_PATH    = GRAPH_REPO_WORKING
```

Logic đúng:

```text
Verify/copy từ GRAPH_REPO_INPUT.
Sau khi copy, mọi command train/generate dùng GRAPH_REPO_PATH = GRAPH_REPO_WORKING.
```

### 10.4. Notebook runner

Notebook đang dùng:

```text
kaggle-end-to-end.ipynb
```

Nên hỗ trợ 3 mode:

```text
teacher_probs
train_a05
train_a02
```

Hiện đã có prompt yêu cầu Codex sửa notebook theo hướng tối thiểu:

```text
không tạo notebook mới
không rewrite lớn
chỉ sửa variable/path/helper/verify logic
```

---

## 11. Hướng tiếp theo đang được cân nhắc: D9-SMR-B

Vì TGMS không cứu được D9, hướng có cơ sở nhất nếu vẫn muốn khai thác D9 là:

```text
D9-SMR-B
= Staged Motif Rescue with Pooled Classifier
```

Mục tiêu: sửa motif trước, không sửa head trước.

### 11.1. Ý tưởng

3 phase:

#### Phase 1 — Motif Warm-up

```text
Train motif discovery / pixel encoder bằng motif-quality losses.
Chưa train classifier chính hoặc cls_weight rất nhỏ.
Mục tiêu: giảm border, tăng foreground, giảm diffuse/collapse.
```

Monitor:

```text
motif_quality_score hoặc composite motif_score
```

#### Phase 2 — Frozen Motif Classifier

```text
Load best warmup.
Freeze encoder + motif discovery.
Train pooled MLP classifier trên motif embeddings.
Monitor val_macro_f1.
```

#### Phase 3 — Light Unfreeze

```text
Load best classifier.
Unfreeze nhẹ motif queries/projection.
Encoder freeze hoặc LR nhỏ.
Loss = CE + 0.01 * motif_loss.
Monitor val_macro_f1.
```

### 11.2. Lý do có cơ sở

Các hướng đã thử thất bại vì không sửa gốc:

```text
MR: thêm head phức tạp khi motif chưa tốt → fail.
TGMS: thêm semantic loss ở logits nhưng không sửa motif → không tăng macro.
Feature ablation: mới screening, chưa semantic.
```

D9-SMR hỏi đúng câu hỏi:

```text
Nếu motif được warm-up tốt hơn, classifier có học tốt hơn không?
```

### 11.3. Run đề xuất nếu triển khai

```bash
python -m scripts.train_d9_smr_staged \
  --config configs/experiments/d9_smr_b.yaml \
  --env kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --phase all \
  --output_root outputs/d9_smr_b_cap300v100 \
  --device cuda \
  --no_wandb
```

Budget gợi ý:

```text
warmup:    8 epochs, cap300v100
classifier: 20 epochs, cap300v100
finetune:  10–20 epochs, cap300v100
```

Không tăng budget trước khi thấy macro F1 > 0.20.

---

## 12. Quy tắc quyết định hiện tại

### Nếu mục tiêu là điểm cao trong 2 ngày

```text
Dừng D9 làm result chính.
Dùng D7/D8B ensemble làm performance result.
```

Lý do:

```text
D7/D8B ensemble đã macro F1 ≈ 0.6342.
D9 hiện quanh 0.16–0.17.
```

### Nếu mục tiêu là khai thác D9 đúng ý tưởng thầy

```text
Không chạy thêm MR/TGMS alpha lớn.
Triển khai D9-SMR staged motif rescue.
```

Success criteria:

```text
>0.20 macro F1: có tín hiệu D9 tốt hơn.
>0.30 macro F1: đáng phát triển nghiêm túc.
>0.40 macro F1: rất thành công trong thời gian ngắn.
```

Nếu vẫn quanh 0.16–0.17:

```text
D9 hiện cần thiết kế motif discovery mới, không thể cứu bằng head/loss ngắn hạn.
```

---

## 13. Các file quan trọng hiện tại

### D9 model/training

```text
models/d9_relation_encoder.py
models/d9_motif_relation_classifier.py
models/d9_relation_motif_model.py
models/d9_sanity_models.py
scripts/train_d9_relation_motif.py
scripts/visualize_d9_motifs.py
```

### Teacher distillation

```text
scripts/generate_teacher_probs.py
configs/experiments/d9_tgms_b_distill_a05_t2.yaml
configs/experiments/d9_tgms_b_distill_a02_t2.yaml
docs/d9_tgms_distillation.md
```

### Analysis scripts/reports

```text
scripts/analyze_d9_run_artifacts.py
scripts/analyze_d9_tgms_distill_runs.py
outputs/d9_tgms_distill_analysis_report.md
outputs/d9_tgms_distill_analysis_summary.csv
```

### Important outputs

```text
outputs/d9_rg_mr_b_diag20_cap200v100
outputs/d9_rg_b_no_mr_pooled_mlp_diag20_cap300v100
outputs/d9_tgms_b_distill_a05_t2_cap300v100_outputs
outputs/d9_tgms_b_distill_a02_t2_cap300v100_outputs
```

---

## 14. Current final technical diagnosis

```text
D9 is not failing because the code cannot train.
D9 is failing because the learned motifs are not semantic/useful enough.
```

Current evidence:

```text
Global pool sanity can learn weakly → data/training loop not totally broken.
Pooled MLP beats MR → MR head is not the main fix.
Teacher distillation does not beat no-teacher → output semantic guidance alone is insufficient.
Motif metrics are poor → representation bottleneck is at motif discovery.
Class difficult categories stay dead → learned features are not emotion-discriminative.
```

One-line conclusion:

```text
D9 không thấp vì thiếu classifier mạnh; D9 thấp vì motif đầu vào cho classifier chưa mang ý nghĩa cảm xúc.
```

---

## 15. Recommended next instruction for AI coding agent

If continuing D9, do **not** run another random variant. Implement/inspect D9-SMR staged training only if there is enough time.

Priority order:

```text
1. Verify current D9 motif outputs and metrics.
2. Implement staged motif warm-up if not already available.
3. Freeze motif and train pooled classifier.
4. Light unfreeze only if classifier gets signal.
5. Run motif deletion test to check whether motifs influence predictions.
```

Avoid:

```text
- MR KxK old head
- TGMS alpha 0.2/0.5 repeat
- full 200 epoch D9
- feature ablation sweep without motif fix
- CSV-based teacher order
```

