# Kiến Trúc D11: Global-Local Gated Motif Graph Network với FACS Spatial Prior

## 1. Tóm tắt điều hành
D11 là kiến trúc mạng nơ-ron đồ thị phân cấp (Hierarchical Graph Neural Network) được thiết kế để huấn luyện **1-Stage (End-to-End)**, khắc phục triệt để sự rườm rà của quy trình 2-Stage ở D10. D11 giải quyết hai điểm mù lớn nhất của đồ thị pixel: (1) Thiếu ngữ cảnh toàn cục và (2) Sự "vô kỷ luật" về không gian của thuật toán Slot Attention nguyên thủy. Bằng cách kết hợp **Virtual Node** (thu thập bối cảnh) và **FACS Spatial Prior** (kỷ luật không gian bằng Soft Loss), D11 đảm bảo nguyên tắc "Local làm Vua, Global làm Quân sư", duy trì các bằng chứng giải phẫu học rõ ràng mà không rơi vào rủi ro học lối tắt (shortcut learning).

## 2. Bối cảnh và động cơ
Bài toán nhận diện cảm xúc khuôn mặt trên tập dữ liệu FER-2013 đòi hỏi mô hình phải kết hợp cả bằng chứng cục bộ (mắt, miệng, lông mày) và bối cảnh toàn cục (bố cục, trạng thái chung). 

Ở kiến trúc D10, việc trích xuất Motif phụ thuộc hoàn toàn vào Slot Attention theo hướng không giám sát (unsupervised). Điều này dẫn đến sự mất ổn định cực lớn trong các epoch đầu tiên: hàm Loss Cross-Entropy liên tục ép mạng đoán cảm xúc khi các Slot chưa kịp định hình khái niệm khuôn mặt, dẫn đến hiện tượng Slot Collapse (mọi slot đều nhìn vào một chỗ). D10 buộc phải dùng phương pháp pre-train 2-Stage (SupCon) cồng kềnh.

Để tiến tới huấn luyện 1-Stage với mức F1 cao, D11 cần một kiến trúc "ổn định hóa" ngay từ đầu. Việc bổ sung Global Context giúp mạng có mỏ neo để đoán bối cảnh, và áp dụng FACS Spatial Prior ép các Slot vào đúng vị trí giải phẫu học, mở đường cho khả năng học tự nhiên và hội tụ cực nhanh.

## 3. Vấn đề của D10
Kiến trúc D10 gặp các rủi ro cấu trúc do thiếu bối cảnh và thiếu kỷ luật không gian:
*   **Slot collapse:** Nhiều slot tập trung vào cùng một vùng không gian nổi bật.
*   **Nhiễu nền:** Các slot bị thu hút bởi tóc hoặc background do thiếu định hướng toàn cục.
*   **Mất thông tin cấu trúc:** Ở bước candidate selection (chọn lọc top-k), các thông tin nền tảng về cấu trúc bị loại bỏ.
*   **Chậm chạp và tốn tài nguyên:** Phụ thuộc vào quá trình 2-Stage để ổn định hóa Slot Attention.

## 4. Triết lý thiết kế D11
Thiết kế của D11 tuân thủ 7 nguyên tắc:
1.  **Local Motif là trục chính:** Bằng chứng giải phẫu học là cơ sở trực tiếp cho dự đoán.
2.  **Kỷ luật không gian (FACS Priors):** Các Slot bị quản lý bằng hàm Loss không gian để tránh sụp đổ và đảm bảo tính giải phẫu học.
3.  **Global Context là tín hiệu điều chỉnh:** Đóng vai trò bổ trợ để diễn giải bằng chứng cục bộ.
4.  **Không nối trực tiếp:** Nhánh Global không được đưa thẳng vào Classifier.
5.  **Kiểm soát broadcast:** Tránh phát tán Global quá sớm xuống pixel nodes để hạn chế nhòe đặc trưng (over-smoothing).
6.  **Chứng minh qua Ablation:** Hiệu quả phải được kiểm định bằng phân tích nhân quả.
7.  **Sự sinh tồn của Motif:** Nếu metric tổng tăng nhưng Local Motif mất vai trò độc lập, D11 chưa đạt mục tiêu nghiên cứu.

## 5. Kiến trúc tổng quan

```text
Input image 48×48
→ Pixel Graph (2304 nodes, features có x_norm, y_norm)
→ GNN Encoder
→ Tách 2 nhánh song song:
   - Local Branch: pixel_node_embeddings → Slot Attention (với Soft Spatial Loss) → local_raw (8 local motifs)
   - Global Branch: attentive global gather (virtual node) → global_context
→ FiLM-style Modulation:
   local_refined = local_raw * (1 + tanh(W_gamma(global_context))) + W_beta(global_context)
→ Classifier
→ Emotion Prediction
```

**Mô tả khái niệm các khối dữ liệu:**
*   `pixel_nodes`: Đặc trưng tọa độ, cường độ tại từng vị trí.
*   `virtual_node`: Token đại diện bức tranh toàn cảnh.
*   `local_raw`: Đặc trưng motif cục bộ gốc (đã được "nắn" qua FACS Prior).
*   `global_context`: Bối cảnh toàn mặt (bottleneck 32 hoặc 64 dims).
*   `gamma`, `beta`: Các ma trận điều biến có cùng chiều không gian với `local_raw`.
*   `local_refined`: Đặc trưng motif sau khi được ngữ cảnh hóa.

*(Ghi chú kỹ thuật: Các shape như `[B, 2304, D]` là mô tả 개념. Khi triển khai trên PyTorch Geometric, dữ liệu có thể flatten thành `[B×N, D]`).*

## 6. Global Branch: Virtual Node D11A
Virtual Node là một graph-level token gom thông tin từ 2304 pixel nodes thông qua *gated attention* nhằm giảm nhiễu nền.
D11A triển khai theo hướng gather-only/read-only. Nếu Virtual Node phát tán (broadcast) thông tin ngược lại cho các pixel nodes ở các layer đầu tiên, các node cục bộ sẽ nhận chung một lượng tín hiệu toàn cục, gây ra hiện tượng *over-smoothing*, phá vỡ nỗ lực tách cụm của Slot Attention. Giai đoạn đầu ưu tiên kiểm thử D11A an toàn này trước khi nghĩ tới D11B (late gated broadcast).

## 7. Local Branch: Slot/Motif với FACS Spatial Prior (Soft Spatial Loss)
Local Branch tiếp nhận 2304 pixel nodes để tạo ra 8 local motifs. Điểm đột phá của D11 để chạy 1-Stage là áp dụng **Soft Spatial Loss** dựa trên Facial Action Coding System (FACS).

**Vấn đề:** Mạng nguyên thủy tự học rất mù mờ, dẫn đến Slot Collapse.
**Giải pháp (Soft Spatial Loss):**
1.  **Tính Trọng Tâm (Center of Mass):** Dựa vào ma trận Attention `A` của Slot Attention và tính chất không gian gốc `x_norm, y_norm` của pixel nodes, hệ thống sẽ tính tọa độ trọng tâm chú ý `(cx, cy)` cho từng Slot `k`:
    `cx_k = sum(A_{k, i} * x_i)` và `cy_k = sum(A_{k, i} * y_i)`.
2.  **Phân Vai Giải Phẫu Học (FACS Assignment):**
    *   **Slot 0 & 1 (Vùng Mắt/Mày Trái):** Bị phạt nặng nếu trọng tâm đi lệch khỏi góc `x < 0.5` và `y < 0.5`.
    *   **Slot 2 & 3 (Vùng Mắt/Mày Phải):** Bị phạt nặng nếu trọng tâm đi lệch khỏi góc `x > 0.5` và `y < 0.5`.
    *   **Slot 4 & 5 (Vùng Miệng/Cằm):** Bị phạt nặng nếu trọng tâm đi ngược lên nửa trên `y < 0.5`.
    *   **Slot 6 & 7 (Wildcards/Nếp nhăn):** Thả tự do, không bị phạt, cho phép mạng tự tìm các motif bổ sung ngẫu nhiên.
3.  **Ưu điểm của hàm Soft Loss:** Việc phạt (Penalty) linh hoạt hơn rất nhiều so với dùng Hard Mask (chặn tuyệt đối). Nếu ảnh bị crop lệch (miệng nằm lẹm lên trên), mạng vẫn có thể chấp nhận hy sinh một phần Loss để dịch chuyển trọng tâm đi tìm cái miệng. Soft Loss hoạt động như một "Dây thun" kéo các Slot về vị trí khuôn mẫu, cho phép mạng hội tụ an toàn ngay trong 1-Stage mà không bao giờ bị Slot Collapse.

## 8. Fusion bằng FiLM-style Modulation
D11 từ chối nối thẳng (Concatenation) để tránh mạng MLP học lối tắt. Phép điều biến kiểu FiLM được sử dụng:
`local_refined = local_raw * (1 + tanh(W_gamma(global_context))) + W_beta(global_context)`
*   `gamma` điều chỉnh thang đo (scale) của bằng chứng cục bộ.
*   `beta` cung cấp bias nhỏ dựa trên ngữ cảnh.
*   Hàm `tanh` trong thiết kế residual modulation cho phép tăng hoặc giảm nhẹ đặc trưng linh hoạt. Classifier chỉ nhận đầu vào từ `local_refined`, đảm bảo Global chỉ có chức năng cố vấn.

## 9. Anti-shortcut Mechanisms & Hàm Loss Tích Hợp
D11 áp dụng hệ sinh thái bảo vệ bao gồm:
*   **Global Bottleneck:** Giới hạn Global Context ở 32/64 dims.
*   **Global Dropout:** Khởi điểm 0.3 trên nhánh Global.
*   **Hàm Loss Tích Hợp Đa Nhiệm:**
    `L_total = L_fusion + lambda_1 * L_local_aux + lambda_2 * L_spatial_prior`
    - `L_fusion`: Đảm bảo hiệu năng dự đoán tổng thể.
    - `L_local_aux`: Ép `local_raw` tự duy trì năng lực phân loại độc lập.
    - `L_spatial_prior`: Hàm Soft Loss "Kỷ luật không gian", ép các Slot không được rời bỏ vị trí giải phẫu học, triệt tiêu nguy cơ sụp đổ, cho phép 1-Stage training diễn ra mượt mà.

## 10. Các rủi ro toàn pipeline

1.  **Dataset bias:** Global branch học đặc điểm nhiễu (ánh sáng, crop). *Kiểm tra:* Background Sensitivity Test.
2.  **Feature bottleneck:** D11/1-Stage cải thiện kiến trúc nhưng không giải quyết đặc trưng gốc nghèo (chỉ có intensity, gradient, x, y). Trần hiệu năng (ceiling) vẫn có thể bị kẹt ở mức 0.64-0.66 nếu không dùng CNN chiết xuất feature phức tạp.
3.  **Over-smoothing:** Virtual Node làm nhòe node embeddings. *Kiểm tra:* Đo độ tương đồng Cosine.
4.  **Local death:** Motif bị mất vai trò trung tâm. *Kiểm tra:* Zero-Local evaluation.
5.  **Gate shortcut:** `gamma`, `beta` vô tình trở thành mã class ngầm. *Kiểm tra:* Gate-only test.
6.  **Slot collapse:** Rủi ro này đã được khắc phục phần lớn nhờ `L_spatial_prior`, nhưng cần theo dõi xem hệ số `lambda_2` có đủ lớn để giữ kỷ luật không.
7.  **Interpretability ảo:** Attention map đẹp nhưng không có tác động nhân quả. *Kiểm tra:* Masking test.
8.  **Schema drift:** Thêm loại node/edge mới dễ gây lỗi index, mask, batch collate.
9.  **Class imbalance:** Chỉ cải thiện nhóm nhãn dễ học (Happy/Neutral).
10. **Label noise FER-2013:** Mô hình mạnh hơn có thể overfit nhãn sai.
11. **Baseline không công bằng:** D11 có số lượng tham số lớn hơn D10 nên thắng giả định. *Cách giảm thiểu:* Đánh giá với Local-only same-params.
12. **Evaluation thiếu hụt:** Thiếu chứng cứ ablation khoa học sẽ không chứng minh được giả thuyết thiết kế.

## 11. Evaluation Protocol / Test Suite
Để được xem là một cải tiến có cơ sở so với D10, D11 cần được đánh giá qua các kịch bản nhân quả nghiêm ngặt:

### 11.1 Full vs Zero-Global vs Zero-Local
*   **Zero-Global (`global_context = 0`):** Đo lường năng lực tự thân của nhánh Local.
*   **Zero-Local (`local_raw = 0`):** Kiểm tra hiện tượng học lối tắt qua Global.

### 11.2 Shuffle Global Test
*   Tráo đổi bối cảnh Global giữa các ảnh. Nếu Global cung cấp bối cảnh thực sự, hiệu năng cần phải giảm vừa phải.

### 11.3 Drop Top/Random/Replacement Slot
*   *(Vì Slot đã được gán vị trí không gian cụ thể qua FACS Prior, phép Shuffle Slot không còn phù hợp)*.
*   Sử dụng **Drop Top-gated Slot** và **Slot Replacement** (thay Slot miệng bằng Slot miệng của ảnh khác) để đo lường độ sụt giảm confidence, qua đó xác minh tính nhân quả giải phẫu học.

### 11.4 F1_Fusion vs F1_Raw_Local
*   Theo dõi cả hai chỉ số qua các epoch. `F1_Raw_Local` không được thụt lùi so với D10 gốc.

### 11.5 Per-class F1 và Confusion Delta
*   Kiểm tra sự thuyên giảm của các cặp nhầm lẫn cục bộ (Fear→Surprise, Sad→Neutral).

### 11.6 Gate Distribution Logger
*   Theo dõi biểu đồ `gamma`, `beta`. Kiểm tra xem các cổng có biến thiên hợp lý theo ngữ cảnh class hay không.

### 11.7 Gate-only Test
*   Sử dụng vector Gate huấn luyện linear probe. Nếu F1 quá cao, Global đang mã hóa class thông qua cổng điều biến.

### 11.8 Attention Heatmap + Background Sensitivity
*   Hiển thị vùng Attention của Virtual Node. Che vùng nền để xem kết quả dự đoán có thay đổi không.

### 11.9 Over-smoothing Detector
*   Đo độ tương đồng Cosine giữa các node embeddings. So sánh với D10.

### 11.10 Local-only Same-Params Baseline
*   Đối chứng công bằng với phiên bản mạng D10 được doping thông số tham số bằng D11.

### 11.11 Seed Robustness
*   Đánh giá sự ổn định của kết quả thông qua nhiều random seeds.

### 11.12 Global Helpfulness on Hard Samples
*   Kiểm tra xem Global Branch có giúp ích nhiều hơn ở các mẫu dự đoán Local khó/mơ hồ hay không.

### 11.13 Slot Consistency under Augmentation
*   Áp dụng biến đổi ảnh nhẹ (translation). Nhờ có FACS Prior, độ ổn định của hệ thống được kỳ vọng rất cao.

## 12. Tiêu chí thành công và thất bại

**D11 được xem là một hướng đi thành công nếu:**
*   Quá trình huấn luyện 1-Stage diễn ra ổn định, Loss hội tụ mượt mà không bị Slot Collapse.
*   Full D11 đánh bại D10 2-Stage về Macro F1.
*   F1 Raw Local vẫn giữ được năng lực dự đoán.
*   Visualization chứng minh các Slot định hình đúng vị trí giải phẫu học (nhờ FACS Prior) và Gate phân phối có ngữ nghĩa.
*   Confusion giảm ở các cặp class khó phân biệt.

**D11 có thể cần thiết kế lại nếu:**
*   Hàm Spatial Loss quá lỏng (vẫn sập Slot) hoặc quá chặt (Mạng không hội tụ được classification loss).
*   Zero-Local hoặc Gate-only F1 đạt mức cao bất thường (Shortcut).
*   Attention tập trung vào rìa ảnh hoặc tóc.
*   Local raw F1 tụt dốc mạnh.

## 13. Lưu ý triển khai kỹ thuật sau này
*   **Tính Center of Mass:** Tránh chia cho 0 khi Softmax Attention có trị số cực nhỏ. Cần áp dụng epsilon (e.g., `1e-9`).
*   **Cân chỉnh Hệ số Loss:** `lambda_2` cho Spatial Prior cần được warmup lớn từ đầu, sau đó có thể giữ nguyên hoặc giảm dần (decay) khi các Slot đã tìm được đúng vị trí.
*   **Virtual Node Indexing:** Quản lý chặt chẽ để không xuất hiện sự giao thoa (cross-graph) giữa các ảnh trong batch.
*   **Evaluation Mode:** Hàm forward cần cung cấp intermediate tensors để phục vụ chuỗi Test Suite.

## 14. Kế hoạch thực nghiệm đề xuất
Khuyến nghị thực hiện theo tuần tự lên cấp:
1.  **D10 Baseline:** Mốc chuẩn để so sánh.
2.  **D10 + Soft Spatial Loss (1-Stage):** Đánh giá xem riêng FACS Prior đã đủ sức thay thế 2-Stage SupCon hay chưa.
3.  **D11A Simple (1-Stage):** Thêm Virtual Node (Gather-only) + FiLM + Aux Loss + Soft Spatial Loss. Đây là "Hình thái tối thượng" hướng tới.
4.  **D11A Tuning:** Cân chỉnh `lambda_1`, `lambda_2`, bottleneck và dropout.
5.  **D11A Full Evaluation:** Quét toàn bộ Test Suite.

## 15. Bảng tổng hợp ablation

| Khảo sát | Mục đích | Thay đổi cấu hình | Metric cần xem | Kết quả kỳ vọng | Dấu hiệu đáng lưu ý |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D10 Baseline** | Mốc chặn dưới | 2-Stage mặc định | Macro F1 | Ổn định | - |
| **D10 + FACS Prior** | Đánh giá kỷ luật Slot | Áp dụng Soft Spatial Loss | Loss Curve, F1 | Hội tụ mượt ở 1-Stage | Loss hội tụ chậm |
| **D11 Full (1-Stage)**| Năng lực tổng | D11A hoàn chỉnh | Macro F1 | Thắng baseline | Accuracy tăng, Macro F1 chững |
| **D11 Zero-Global** | Vai trò Cục bộ | `global_context = 0` | F1 Raw/Fusion | Giảm vừa phải | Sụp đổ hoàn toàn |
| **D11 Zero-Local** | Kiểm soát Global | `local_raw = 0` | F1 Fusion | Giảm mạnh | F1 cao (Global shortcut) |
| **D11 Shuffle-Global**| Bối cảnh đặc thù | Tráo Global theo ảnh | Macro F1 | Giảm | Không giảm |
| **D11 Drop Slot** | Causal Check nhân quả | Drop slot / Replace slot | Confidence | Giảm mạnh | Confidence không đổi |
| **D11 Gate-Only** | Kiểm tra mã hóa ngầm | Probe bằng Gate vector | F1 Gate-only | F1 thấp | F1 cao bất thường |
| **D11 No Spatial Loss**| Vai trò FACS Prior | `lambda_2 = 0` | Attention Map | Slot Collapse | Mạng tự hội tụ dễ dàng (hiếm) |

## 16. Văn phong báo cáo nên dùng
Đoạn mô tả khái quát kiến trúc:
> *"D11A được đề xuất như một kiến trúc mạng nhận diện cảm xúc 1-Stage End-to-End, giải quyết đồng thời sự thiếu hụt ngữ cảnh toàn cục và tính bất ổn định của đồ thị cục bộ. Bằng cách áp dụng hàm tổn thất không gian (Soft Spatial Prior Loss) dựa trên giải phẫu học, mô hình định hình thành công các bó cơ cục bộ mà không phụ thuộc vào quy trình pre-train phức tạp. Song song đó, ngữ cảnh toàn cục (Virtual Node) được kiểm soát nghiêm ngặt qua cơ chế điều biến FiLM, tinh chỉnh bằng chứng cục bộ thay vì chi phối kết quả. Hiệu quả của hệ thống được thẩm định chuyên sâu bằng các phương pháp phân tích nhân quả."*

## 17. Kết luận
Sự kết hợp giữa Global Context và kỷ luật không gian (FACS Spatial Prior) mở ra cơ hội đột phá đưa cấu trúc D11 trở về quy trình huấn luyện 1-Stage siêu tốc và ổn định. Phương pháp tiếp cận này cung cấp một mỏ neo giải phẫu học bền vững để các mô-típ tự tin phát triển, đồng thời bổ sung thông tin diện rộng bị khuyết thiếu ở các lớp message passing thông thường. Tuy trần hiệu năng có thể bị giới hạn bởi đặc trưng gốc (feature bottleneck), hướng đi này cung cấp một nền tảng khoa học vững chắc và có cơ sở nhất để đẩy giới hạn GNN thuần trên FER-2013 lên cao nhất có thể.
