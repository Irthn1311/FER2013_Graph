Version / hướng
→ Mục tiêu
→ Cải tiến so với bản trước
→ Kết quả / tín hiệu
→ Nhược điểm
→ Bài học rút ra
1. Bức tranh tổng thể dự án

Dự án của bạn đang đi theo hướng:

Ảnh FER-2013 48x48 grayscale
→ biểu diễn thành pixel graph 2304 node
→ học motif biểu cảm
→ dùng motif/quan hệ motif để phân loại emotion

Ý tưởng gốc hợp với thầy là:

pixel-level graph
→ motif nhỏ vùng mắt / lông mày / miệng / khóe miệng / nếp nhăn
→ gộp thành cấu trúc motif toàn khuôn mặt
→ phân loại biểu cảm

Nhưng trong quá trình phát triển, có hai áp lực trái chiều:

1. Muốn điểm cao.
2. Muốn đúng tinh thần motif-GNN.

Các bản D7/D8B điểm cao hơn, nhưng dần lệch sang hướng region/window/Swin-like. Các bản D9 quay lại motif-GNN hơn, nhưng hiện tại điểm thấp vì motif chưa semantic.

2. Nhóm baseline / các bản trước D9
2.1. Baseline MLP trên descriptor motif
Mục tiêu

Dùng các descriptor/motif đã chọn, rồi đưa qua MLP để phân loại ảnh.

subgraph/motif descriptor
→ MLP
→ emotion logits
Cải tiến so với rất ban đầu

Đây là baseline đơn giản để kiểm tra:

motif descriptor có chứa tín hiệu emotion không?
Kết quả

Khoảng:

accuracy ≈ 39.20%
macro F1 ≈ 0.3174
Nhược điểm

MLP không mô hình hóa quan hệ giữa các motif.

Nó xem mỗi ảnh như một vector đặc trưng, không hiểu motif nào liên quan motif nào.

Ví dụ:

khóe miệng + mắt
lông mày + miệng dưới
mắt mở + miệng mở

các quan hệ này không được học rõ.

Bài học

Descriptor motif có một ít tín hiệu, nhưng chưa đủ. Nếu muốn đi đúng hướng thầy, phải học quan hệ giữa các motif, không chỉ dùng MLP đơn giản.

2.2. GNN trên motif descriptor
Mục tiêu

Thay vì MLP, dùng GNN để học quan hệ giữa các motif/subgraph descriptors.

motif descriptor nodes
→ motif-level GNN
→ image-level prediction
Cải tiến

So với MLP, bản này bắt đầu học:

motif-to-motif relation
Kết quả

Một bản motif_norm khoảng:

accuracy ≈ 45.11%
macro F1 ≈ 0.4196
weighted F1 ≈ 0.4380

Một số class cải thiện, ví dụ Disgust có tín hiệu hơn. Fear vẫn yếu.

Nhược điểm

Motif đầu vào vẫn phụ thuộc mạnh vào quá trình sinh/chọn motif. Nếu motif chọn ra chưa tốt, GNN phía sau cũng chỉ học trên dữ liệu nhiễu.

Ngoài ra, motif lúc này vẫn còn mang tính “descriptor/subgraph”, chưa chắc đã đúng là vùng mắt/miệng/lông mày có ý nghĩa.

Bài học

GNN ở tầng motif có ích hơn MLP, nhưng chất lượng motif đầu vào là nút thắt. Không thể chỉ tăng classifier nếu motif chưa semantic.

3. Nhóm D5/D6: hierarchical motif / class-part refinement
3.1. D5
Mục tiêu

Đi sâu hơn vào hướng motif theo tầng:

pixel/subgraph motif
→ part/motif representation
→ graph classifier
Cải tiến

So với GNN descriptor, D5 cố gắng đưa motif gần hơn với các bộ phận khuôn mặt.

Kết quả / tín hiệu

Một số bản D5 có vấn đề collapse, đặc biệt một vài class như Disgust có thể rất yếu hoặc bằng 0 ở một số biến thể.

Nhược điểm

Motif/slot chưa ổn định.

Có hiện tượng:

slot collapse
nhiều slot nhìn cùng vùng
slot học nền/tóc/border
class khó như Fear/Disgust yếu
Bài học

Muốn motif có ý nghĩa thì cần kiểm soát:

diversity giữa slots
foreground / border
class-wise motif usage

Nếu không, motif dễ trở thành các vùng nhiễu.

3.2. D6A — slot_pixel_part_graph_motif
Mục tiêu

Học motif theo cấu trúc hierarchical hơn:

pixel-level
→ part slots
→ part graph / motif graph
→ emotion
Cải tiến

D6A đưa motif gần hơn với ý tưởng:

mắt / miệng / vùng mặt

thay vì chỉ descriptor trừu tượng.

Kết quả

Khoảng:

accuracy ≈ 53.11%
macro F1 ≈ 0.4888

D6A có tín hiệu tốt hơn các bản trước, đặc biệt cứu được một số class từng rất yếu.

Nhược điểm

Slot collapse vẫn tồn tại. Một slot có thể chiếm tỷ trọng rất lớn, ví dụ khoảng hơn 50% mass.

Fear vẫn yếu:

Fear F1 ≈ 0.2966
Bài học

Hierarchical motif giúp tăng điểm, nhưng nếu slot không phân tán hợp lý thì model vẫn không học được đầy đủ biểu cảm. Cần loss/regularization cho diversity và foreground.

3.3. D6B — class-part refinement + border loss
Mục tiêu

Cải thiện D6A bằng:

class-part refinement
border loss tốt hơn
kiểm soát slot collapse
Cải tiến so với D6A

D6B xử lý trực tiếp hai vấn đề:

slot học border/nền
slot collapse
Kết quả

Khoảng:

accuracy ≈ 54.08%
macro F1 ≈ 0.4985
Fear F1 ≈ 0.3698

So với D6A:

accuracy tăng từ 53.11% → 54.08%
macro F1 tăng từ 0.4888 → 0.4985
Fear cải thiện rõ
Nhược điểm

Disgust có thể giảm ở một số trade-off. Slot collapse giảm nhưng chưa biến mất. Performance vẫn chưa đạt mức cao.

Bài học

Regularization hình học như border loss có tác dụng. Nhưng chỉ border loss không đủ để đảm bảo motif semantic. Cần vừa kiểm soát vị trí, vừa làm motif có liên hệ với class.

4. Nhóm D7: Graph-Swin / region transformer
4.1. D7 seed44 / long150 / window4
Mục tiêu

Tăng performance bằng kiến trúc mạnh hơn, kết hợp graph representation với region/window/Swin-like transformer.

pixel/region graph features
→ region/window modeling
→ transformer-like interaction
→ emotion prediction
Cải tiến

D7 tăng mạnh khả năng representation. Model không còn phụ thuộc quá nhiều vào motif discovery thủ công/yếu.

Các biến thể quan trọng:

d7a_graph_swin_region_transformer_seed44
d7a_graph_swin_region_transformer_long150_resume
d7a_graph_swin_region_transformer_window4
Kết quả

Một số mốc:

D7 window4:
accuracy ≈ 0.6105
macro F1 ≈ 0.5837
weighted F1 ≈ 0.6049

D7 seed44:
accuracy ≈ 0.6069
macro F1 ≈ 0.5883

D7-only ensemble:

accuracy ≈ 0.6436
macro F1 ≈ 0.6292
weighted F1 ≈ 0.6401
Nhược điểm

D7 điểm tốt hơn rõ, nhưng bắt đầu lệch khỏi ý tưởng thầy.

Nó thiên về:

region/window representation
Swin-like hierarchy
performance-oriented architecture

hơn là motif nhỏ có ý nghĩa như:

mắt
lông mày
khóe miệng
nếp nhăn
Bài học

Nếu mục tiêu là điểm, D7 rất mạnh. Nhưng nếu mục tiêu nghiên cứu motif-GNN đúng tinh thần thầy, D7 không phải bản đại diện tốt nhất.

5. Nhóm D8B: Face-aware Graph Swin
5.1. D8B border020
Mục tiêu

Tiếp tục tăng performance bằng hướng face-aware/Swin-like, đồng thời có thêm kiểm soát vùng mặt/border.

Cải tiến so với D7

D8B đưa thêm yếu tố face-aware và border control. Ý tưởng là giúp model tập trung hơn vào vùng khuôn mặt, giảm nhiễu nền/border.

Kết quả

Bản d8b_face_aware_graph_swin_border020 khoảng:

accuracy ≈ 0.6085
macro F1 ≈ 0.5907
weighted F1 ≈ 0.6068

Seed khác:

seed43:
accuracy ≈ 0.5943
macro F1 ≈ 0.5751

seed44:
accuracy ≈ 0.5940
macro F1 ≈ 0.5622
Nhược điểm

D8B cũng vẫn lệch về performance architecture. Nó mạnh hơn D9 hiện tại, nhưng không thể giải thích rõ motif nhỏ theo ý tưởng thầy.

Bài học

Face-aware/border-aware có lợi cho classification. Nhưng nó không tự động sinh motif semantic. Các bản điểm cao vẫn cần được phân biệt với hướng motif-GNN nghiên cứu.

5.2. D8B area045
Mục tiêu

Tạo thêm diversity cho ensemble, thay đổi area/border setting.

Cải tiến

Không nhất thiết single model tốt nhất, nhưng giúp ensemble đa dạng hơn.

Nhược điểm

Không có đủ thông tin single metric trong context, nhưng nó là member của ensemble mạnh nhất.

Bài học

Một model không cần là single tốt nhất vẫn có thể hữu ích cho ensemble nếu lỗi/prediction khác các model còn lại.

6. Ensemble D7 + D8B
6.1. D7/D8B graph-only ensemble
Mục tiêu

Lấy kết quả performance cao nhất bằng probability averaging.

Members:

D7 seed44
D7 long150_resume
D7 window4
D8B border020
D8B area045

Tên ensemble:

d7_d8b_ensemble_seed44_long150_window4_border020_area045_probavg
Kết quả

Khoảng:

accuracy ≈ 0.6461
macro F1 ≈ 0.6342
weighted F1 ≈ 0.6421
Cải tiến

Đây là kết quả mạnh nhất hiện tại. Ensemble tận dụng diversity giữa D7/D8B và các seed/config khác nhau.

Nhược điểm

Về nghiên cứu motif, ensemble này không phải bản giải thích tốt nhất. Nó là performance result.

Bài học

Nếu mục tiêu là điểm cao, ensemble là hướng đúng. Nếu mục tiêu là “bản đúng ý tưởng thầy”, cần D9 hoặc một biến thể motif-GNN khác.

7. Nhóm D9: quay lại motif-GNN đúng ý tưởng thầy

D9 được mở ra vì bạn thấy các bản D7/D8B tuy mạnh nhưng ngày càng lệch khỏi hướng thầy. D9 cố quay lại:

pixel graph
→ motif discovery
→ motif relation
→ emotion classification

Đây là hướng đúng hơn về mặt nghiên cứu, nhưng hiện tại yếu về metric.

8. D9-F0 Feature Audit
8.1. Mục tiêu

Khảo sát feature node nào giúp motif discovery tốt hơn.

Các bản:

F0-A: intensity-only + full edge
F0-B: intensity + x/y
F0-C: intensity + gx/gy
F0-D: intensity + gx/gy + grad_mag
F0-E: no x/y
F0-F: full current feature
Kết quả chính

So với Stage1G long24 baseline, các bản D9-F0 đều chưa vượt. Nhưng trong nội bộ D9:

B = intensity + x/y

có:

border/outer thấp nhất
foreground cao nhất

Còn:

E = no x/y

có clean-count tốt nhất trong nhóm D9, nhưng border/redundancy chưa ổn.

Full feature F:

selected_foreground thấp
clean rất thấp
Nhược điểm

Các run diagnostic ngắn, chưa kết luận classification. Chúng chỉ cho biết motif geometry ban đầu.

Bài học

3D [intensity, x, y] tốt hơn cho motif geometry hiện tại, nhưng không có nghĩa chắc chắn tốt nhất cho classification. Full 7D không tự nhiên tốt hơn vì gradient/contrast có thể kéo motif vào edge/tóc/border.

9. D9-RG-MR-B gốc
9.1. Mục tiêu

Triển khai D9 end-to-end đúng ý tưởng:

Pixel graph
→ EdgeAwarePixelEncoder
→ MotifDiscoveryModule
→ MotifRelationClassifier
→ emotion logits

Feature:

node_indices = [0,1,2]
edge_indices = [0,1,2,3,4]
node_dim = 3
edge_dim = 5
Cải tiến

Đây là bản D9 đầu tiên chạy end-to-end:

pixel-to-pixel relation
→ motif
→ motif-to-motif relation
→ classification

Nó đúng tinh thần thầy hơn D7/D8B.

Kết quả

Diagnostic:

best val_macro_f1 ≈ 0.0553
val_accuracy ≈ 0.24
val_weighted_f1 ≈ 0.0929

Vấn đề rất nặng:

toàn bộ val predict Happy
6/7 class F1 = 0
Nhược điểm

Motif Relation Classifier KxK quá phức tạp khi motif chưa tốt. Nó làm model collapse.

Bài học

Không thể học relation giữa motif nếu motif chưa có ý nghĩa. MR không phải bước đầu tiên nên làm. Phải có motif representation ổn trước.

10. D9 pooled MLP no-MR
10.1. Mục tiêu

Bỏ Motif Relation Classifier phức tạp, dùng head đơn giản hơn:

motif_embeddings
→ selection-weighted pooling
→ MLP classifier
Cải tiến

Giảm độ phức tạp head. Không ép học KxK motif relation khi motif chưa semantic.

Kết quả

Bản d9_rg_b_no_mr_pooled_mlp_cap200v100:

best val_macro_f1 ≈ 0.1476
val_accuracy ≈ 0.2275
val_weighted_f1 ≈ 0.1692

Bản dài hơn diag20 cap300:

best val_macro_f1 ≈ 0.1674
val_accuracy ≈ 0.2213
val_weighted_f1 ≈ 0.1984
Nhược điểm

Vẫn yếu. Disgust/Fear/Neutral còn rất thấp. Chưa vượt rõ Stage2 cũ.

Bài học

Pooled MLP tốt hơn MR rất rõ. Head đơn giản là baseline D9 ổn nhất hiện tại. Nhưng điểm vẫn thấp, nghĩa là bottleneck không chỉ ở head, mà ở motif representation.

11. D9 no-relation pooled MLP
11.1. Mục tiêu

Kiểm tra EdgeAwarePixelEncoder có giúp hay phá.

Pipeline:

x [B,2304,3]
→ node projection MLP
→ motif discovery
→ pooled MLP
→ logits

Không dùng relation encoder/message passing.

Kết quả

Cap200:

best val_macro_f1 ≈ 0.1606
val_accuracy ≈ 0.2125
val_weighted_f1 ≈ 0.1628
Cải tiến

Ở cùng cap200, no-relation thắng RG pooled MLP 0.1476.

Nhược điểm

Vẫn quanh 0.16. Chưa đủ để kết luận no-relation là hướng mạnh, vì RG pooled MLP với cap300 đạt 0.1674.

Bài học

Relation encoder chưa chứng minh có ích. Có thể nó đang gây oversmoothing hoặc nhiễu. Nhưng dù có bỏ relation encoder, motif representation vẫn yếu.

12. D9 MR-v2 residual
12.1. Mục tiêu

Thay MR KxK cũ bằng relation branch nhẹ hơn:

logits = pooled_logits + alpha * relation_logits

Ý tưởng là relation chỉ bổ sung, không thay thế pooled baseline.

Kết quả
best val_macro_f1 ≈ 0.1347
val_accuracy ≈ 0.2100
val_weighted_f1 ≈ 0.1556
Nhược điểm

Kém hơn pooled MLP. Relation branch vẫn không giúp.

Bài học

Ngay cả relation residual nhẹ cũng chưa có ích khi motif chưa semantic. Không nên thêm relation head lúc này.

13. D9-TGMS: Teacher-Guided Motif Semantic Graph
13.1. Mục tiêu

Dùng D7/D8B teacher để truyền semantic soft labels cho D9.

Loss:

CE(label, student_logits)
+ alpha * KL(student, teacher)
+ motif_aux_weight * motif_loss

Teacher không dạy motif map, chỉ dạy semantic distribution.

Cải tiến

So với D9 pooled MLP, TGMS thêm supervision mềm từ teacher mạnh hơn.

Chạy hai alpha:

alpha = 0.5
alpha = 0.2
T = 2
Kết quả

Alpha 0.5:

best_epoch = 11
val_macro_f1 = 0.1599
val_accuracy = 0.2363
val_weighted_f1 = 0.1993

Alpha 0.2:

best_epoch = 23
val_macro_f1 = 0.1642
val_accuracy = 0.2419
val_weighted_f1 = 0.2017

No-teacher pooled MLP reference:

val_macro_f1 ≈ 0.1674
val_accuracy ≈ 0.2213
val_weighted_f1 ≈ 0.1984
Nhược điểm

TGMS không vượt no-teacher về macro F1.

Alpha 0.5 quá mạnh:

(alpha * KL) / CE ≈ 1.084

Alpha 0.2 vừa hơn:

(alpha * KL) / CE ≈ 0.452

Nhưng vẫn chưa cải thiện class khó. Disgust vẫn chết:

Disgust F1 = 0.0
Disgust prediction = 0

Motif metrics vẫn xấu:

selected_foreground = 0.0
selected_border ≈ 0.307–0.309
entropy cao
effective_count ≈ 15.5/16
Bài học

Teacher semantic loss không sửa được motif. Nó chỉ kéo output distribution. Nếu motif representation yếu, teacher không đủ cứu model. Alpha lớn còn kéo student về class lớn như Happy/Surprise.

14. Vấn đề cốt lõi hiện tại của D9

Sau tất cả thí nghiệm, có thể chốt:

D9 không thấp vì thiếu classifier mạnh.
D9 thấp vì motif đầu vào cho classifier chưa mang ý nghĩa cảm xúc.

Chuỗi nguyên nhân:

Motif maps diffuse / border-heavy / foreground thấp
→ motif embeddings không semantic
→ pooled MLP chỉ học được tín hiệu yếu
→ MR relation học trên motif nhiễu nên collapse
→ teacher loss chỉ sửa logits, không sửa motif
→ class khó như Disgust/Fear/Neutral không học được
→ macro F1 quanh 0.16

Đây là vấn đề gốc.

15. So sánh vai trò các hướng
Hướng	Điểm mạnh	Điểm yếu	Vai trò hiện tại
D7/D8B ensemble	Điểm cao nhất	Lệch ý tưởng motif-GNN	Result performance chính
D6B	Motif/part hợp lý hơn D7/D8	Điểm chưa cao	Baseline motif/part tốt
D9-RG-MR	Đúng ý tưởng thầy	Collapse nặng	Negative ablation
D9 pooled MLP	Head ổn nhất D9 hiện tại	Vẫn yếu	D9 baseline chính
D9 no-relation	Kiểm tra relation encoder	Chưa vượt rõ	Ablation
D9 MR-v2	Thử relation nhẹ	Không giúp	Loại khỏi mainline
D9-TGMS	Thử teacher semantic	Không vượt no-teacher	Negative/neutral experiment
D9-SMR đề xuất	Sửa motif trước	Chưa chạy	Hướng D9 đáng thử nhất
16. Bài học tổng hợp theo từng cấp
16.1. Về feature

3D [intensity, x, y] tốt hơn cho motif geometry hiện tại. Full 7D có thể tốt cho classifier mạnh như D7/D8B, nhưng không chắc tốt cho motif discovery.

Bài học:

Feature tốt cho classifier chưa chắc tốt cho motif.
16.2. Về head classifier

MR phức tạp không giúp khi motif yếu. Pooled MLP ổn định hơn.

Bài học:

Đừng học relation khi node/motif chưa có nghĩa.
16.3. Về teacher distillation

Teacher chỉ dạy semantic output, không dạy motif.

Bài học:

Nếu muốn motif tốt, phải có loss/phase trực tiếp cải thiện motif.
Teacher logits không đủ.
16.4. Về relation encoder

Relation encoder chưa chứng minh giúp. No-relation có lúc thắng.

Bài học:

Message passing pixel-level có thể gây nhiễu/oversmoothing nếu chưa kiểm soát.
16.5. Về D9 nói chung

D9 đúng hướng nghiên cứu, nhưng chưa mature. Cần staged motif learning.

Bài học:

Học motif trước, rồi mới học classifier/relation.
