Được, giờ ta “mổ” riêng khối encoder mà D10/D11 đang dùng. Tên chính xác của khối này là:

SharedPixelEncoder
    └── EdgeAwarePixelMessageLayer × L

Nó nằm trong dual_branch_graph_swin_motif.py, còn lõi message passing thật sự nằm trong slot_pixel_part_graph_motif.py. D10 dùng nó làm “Deep Pixel Encoder”, còn D11 cũng gọi self.encoder = SharedPixelEncoder(...).

1. Encoder này nhận gì và trả gì?

Với D10 dạng chuẩn, input vào encoder là:

x:          [B, N, node_dim]
edge_index: [2, E]
edge_attr:  [E, edge_dim] hoặc [B, E, edge_dim]
node_mask:  [B, N] hoặc None

Trong bài của bạn thường là:

B = batch size
N = 2304 pixel nodes
node_dim = 7
edge_dim = 5
E ≈ 17860 cạnh
hidden_dim = 128 ở D10

D10 ghi rõ pipeline ban đầu là:

Pixel graph (N=2304, 7D node, 5D edge)
→ Deep EdgeAware GNN Encoder
→ Iterative Slot Attention
→ Motif Relation Transformer
→ Class-Motif Attention

Output của encoder là:

h_pixel: [B, N, hidden_dim]

Tức là mỗi pixel ban đầu từ vector 7 chiều được biến thành một embedding ẩn, ví dụ 128 chiều.

2. SharedPixelEncoder gồm những gì?

Trong code, SharedPixelEncoder có 2 phần:

1. input_proj
2. pixel_layers = EdgeAwarePixelMessageLayer × pixel_gnn_layers

Cụ thể, input_proj gồm:

nn.Linear(node_dim, hidden_dim)
nn.LayerNorm(hidden_dim)
nn.GELU()
nn.Dropout(dropout)

Sau đó pixel_layers là một danh sách nhiều EdgeAwarePixelMessageLayer, số lượng phụ thuộc pixel_gnn_layers.

Trong forward:

h_pixel = self.input_proj(x.float())

for layer in self.pixel_layers:
    h_pixel = layer(
        h_pixel,
        edge_index=edge_index,
        edge_attr=edge_attr.float(),
        node_mask=node_mask,
    )

return h_pixel

Nói đơn giản:

x raw pixel features
→ chiếu lên hidden_dim
→ truyền qua nhiều lớp message passing có nhận thức cạnh
→ h_pixel
3. Bước 1: input_proj làm gì?

Ban đầu mỗi pixel có feature thô, ví dụ:

[intensity, x_norm, y_norm, gradient, ...]

Nếu node_dim = 7, mỗi node là vector 7 chiều.

Nhưng GNN không làm việc trực tiếp trên vector 7 chiều. Nó đưa mỗi pixel vào không gian ẩn lớn hơn:

x:       [B, 2304, 7]
Linear
h_pixel: [B, 2304, 128]

Sau đó:

LayerNorm → ổn định phân phối feature
GELU      → phi tuyến
Dropout   → regularization

Ý nghĩa:

input_proj = biến mỗi pixel từ “dữ liệu thô” thành “embedding ban đầu”.

Ở thời điểm này, mỗi node vẫn chủ yếu biết thông tin của chính nó. Nó chưa biết hàng xóm.

4. Bước 2: EdgeAwarePixelMessageLayer làm gì?

Đây là phần quan trọng nhất.

Code định nghĩa nó là:

class EdgeAwarePixelMessageLayer(nn.Module):
    """Lightweight edge-gated message passing over the fixed pixel graph."""

Bên trong có:

msg_mlp   : biến embedding node nguồn thành message
edge_gate : biến edge_attr thành gate
agg_mlp   : xử lý message đã gom
norm_msg  : LayerNorm sau residual message
ffn       : feed-forward network
norm_ffn  : LayerNorm sau residual FFN

Ta phân rã forward từng bước.

5. Bước 2.1: lấy source và destination từ edge_index

Code:

src = edge_index[0].long()
dst = edge_index[1].long()

Nếu:

edge_index[:, k] = [12, 13]

thì:

src[k] = 12
dst[k] = 13

Nghĩa là:

node 12 gửi message
node 13 nhận message

Với graph 8-neighbor, mỗi pixel nhận message từ các pixel kề nó.

6. Bước 2.2: lấy embedding của node nguồn

Code:

h_src = h.index_select(dim=1, index=src)

Nếu:

h: [B, N, H]
src: [E]

thì:

h_src: [B, E, H]

Nghĩa là với mỗi cạnh, ta lấy embedding của node nguồn tương ứng.

Ví dụ:

edge 12 → 13
h_src[k] = h[:, 12, :]
7. Bước 2.3: tính gate từ edge_attr

Code:

gate = torch.sigmoid(self.edge_gate(edge_attr.float()))

edge_gate là MLP:

Linear(edge_dim, hidden_dim)
GELU()
Linear(hidden_dim, hidden_dim)

Nếu:

edge_attr: [E, 5]

thì sau edge_gate:

gate: [E, H]

hoặc broadcast thành:

gate: [B, E, H]

Giá trị sau sigmoid nằm trong:

0 → 1

Ý nghĩa:

gate gần 0: chặn message qua cạnh này
gate gần 1: cho message đi mạnh qua cạnh này
gate khoảng 0.5: cho message đi một phần

Đây là chỗ “edge-aware” nhất của encoder.

8. Bước 2.4: tạo message từ node nguồn, rồi nhân với gate

Code:

msg = self.msg_mlp(h_src) * gate

msg_mlp là:

Linear(hidden_dim, hidden_dim)
GELU()
Dropout()

Vậy:

h_src       → msg_raw
edge_attr   → gate
msg         = msg_raw * gate

Công thức:

g_ij = sigmoid(MLP_edge(e_ij))

m_ij = MLP_msg(h_i) ⊙ g_ij

Trong đó:

h_i  = embedding node nguồn
e_ij = edge_attr của cạnh i → j
g_ij = gate cạnh
m_ij = message sau khi gate
⊙    = nhân từng chiều

Điểm rất quan trọng:

Gate không phải một số scalar duy nhất.
Gate có thể là vector hidden_dim chiều.

Tức là cạnh có thể cho qua một số kênh đặc trưng, chặn một số kênh khác.

Ví dụ:

gate = [0.9, 0.1, 0.7, 0.0, ...]

Nghĩa là thông tin không bị đóng/mở toàn bộ, mà được lọc theo từng chiều feature.

9. Bước 2.5: gom message về node đích

Code:

agg = msg.new_zeros(h.shape)
agg.index_add_(dim=1, index=dst, source=msg)

Ban đầu:

agg: [B, N, H] toàn 0

Sau đó, với mỗi cạnh i → j, message m_ij được cộng vào:

agg[:, j, :] += m_ij

Ví dụ node 13 nhận từ nhiều node:

12 → 13
14 → 13
61 → 13
62 → 13
...

thì:

agg[:, 13, :] =
    msg_12_to_13
  + msg_14_to_13
  + msg_61_to_13
  + ...

Đây là quá trình aggregate.

10. Bước 2.6: chia trung bình theo degree

Code:

deg = msg.new_zeros(h.shape[1])
deg.index_add_(dim=0, index=dst, source=torch.ones_like(dst, dtype=msg.dtype))
agg = agg / deg.clamp_min(1.0).view(1, -1, 1)

Mục đích là biến tổng thành trung bình.

Vì pixel giữa ảnh có thể nhận 8 message, pixel góc chỉ nhận 3 message. Nếu chỉ cộng, node giữa có magnitude lớn hơn node góc chỉ vì nó nhiều hàng xóm hơn.

Chia degree giúp:

node giữa không bị lớn bất thường
node biên/góc không bị nhỏ bất thường

Công thức:

agg_j = mean_{i ∈ N(j)} m_ij
11. Bước 2.7: áp dụng node_mask nếu có

Code:

if node_mask is not None:
    mask = node_mask.to(dtype=h.dtype).unsqueeze(-1)
    agg = agg * mask

Nếu có node không hợp lệ, nó sẽ không nhận thông tin.

Với ảnh FER 48×48 full grid, thường tất cả 2304 node đều hợp lệ. Nhưng node_mask hữu ích khi pipeline có padding hoặc subgraph.

12. Bước 2.8: residual update lần 1

Code:

h = self.norm_msg(h + self.agg_mlp(agg))

agg_mlp là:

Linear(hidden_dim, hidden_dim)
GELU()
Dropout()

Ý nghĩa:

agg      = thông tin từ hàng xóm
agg_mlp  = xử lý thông tin hàng xóm
h + ...  = residual connection
LayerNorm = ổn định

Công thức:

h'_j = LayerNorm(h_j + MLP_agg(agg_j))

Residual rất quan trọng. Nó giúp node giữ thông tin gốc của chính nó, không bị hàng xóm ghi đè hoàn toàn.

13. Bước 2.9: feed-forward residual lần 2

Code:

h = self.norm_ffn(h + self.ffn(h))

ffn là:

Linear(H, 2H)
GELU()
Dropout()
Linear(2H, H)
Dropout()

Đây giống block Transformer kiểu:

message/update block
→ FFN block

Ý nghĩa:

Sau khi nhận thông tin hàng xóm, mỗi node tự xử lý lại embedding của mình bằng FFN.

Công thức:

h''_j = LayerNorm(h'_j + FFN(h'_j))
14. Toàn bộ một EdgeAwarePixelMessageLayer dưới dạng công thức

Một layer đầy đủ là:

src, dst = edge_index

h_src = h[src]

g_ij = sigmoid(MLP_edge(e_ij))

m_ij = MLP_msg(h_i) ⊙ g_ij

agg_j = mean_{i ∈ N(j)} m_ij

h'_j = LayerNorm(h_j + MLP_agg(agg_j))

h''_j = LayerNorm(h'_j + FFN(h'_j))

Output:

h_out = h''

Shape không đổi:

[B, 2304, hidden_dim] → [B, 2304, hidden_dim]
15. Encoder nhiều layer thì thông tin lan như thế nào?

Nếu pixel_gnn_layers = 3, D10 dùng 3 lớp edge-aware message passing. D10 config mặc định trong code đặt pixel_gnn_layers: int = 3.

Ý nghĩa receptive field:

1 layer: mỗi pixel biết hàng xóm 1 bước
2 layer: biết vùng cách 2 bước
3 layer: biết vùng cách 3 bước

Với grid 8-neighbor, sau 3 layer, một pixel có thể nhận ngữ cảnh từ vùng khoảng 7×7 quanh nó.

Nó vẫn là local receptive field, chưa thể nối mắt với miệng nếu khoảng cách lớn. Vì vậy D10 cần Slot Attention + Motif Relation Transformer phía sau, còn D11 thêm Global branch.

16. Encoder này đang làm “smoothing có điều kiện”

Một GNN thường có nguy cơ làm mượt node quá mức: pixel gần nhau cứ truyền thông tin qua lại, ranh giới bị nhòe.

EdgeAwarePixelMessageLayer giảm rủi ro đó bằng gate từ cạnh:

vùng đồng nhất:
edge_attr cho thấy intensity gần nhau
→ gate có thể mở
→ thông tin lan truyền mượt

vùng ranh giới:
edge_attr cho thấy delta_intensity lớn
→ gate có thể giảm
→ hạn chế nhòe biên

Nhưng phải nói cẩn thận: model có khả năng học hành vi này, không đảm bảo tự động làm đúng ngay từ đầu.

17. Đây có phải attention không?

Ở tầng encoder này: không phải GAT attention.

Nó là:

soft edge-gating

Khác biệt:

GAT:
alpha_ij = softmax(score(h_i, h_j))
các neighbor cạnh tranh với nhau

EdgeAware:
gate_ij = sigmoid(MLP(edge_attr_ij))
mỗi cạnh được mở/đóng độc lập theo edge feature

Vậy encoder hiện tại không hỏi:

node đích nên nghe node nguồn nào nhất?

mà hỏi:

cạnh này có cho message đi qua không, và cho qua kênh feature nào?

Đây là lựa chọn hợp lý cho pixel graph vì edge_attr chứa thông tin thị giác rất quan trọng.

18. Vai trò của encoder trong D10

Trong D10:

x, edge_index, edge_attr
→ SharedPixelEncoder
→ h_pixel
→ IterativeSlotAttention
→ motif_embeddings
→ MotifRelationTransformer
→ ClassMotifAttentionHead

D10 code mô tả encoder là bước 1 “Deep Pixel Encoder”, sau đó mới đến Slot Attention.

Nghĩa là Slot Attention không làm việc trên pixel raw nữa. Nó làm việc trên pixel embeddings đã được GNN làm giàu bằng ngữ cảnh cục bộ.

Nếu encoder yếu, slot sẽ hút những feature nghèo. Nếu encoder tốt, slot có cơ hội học motif mắt/miệng/lông mày rõ hơn.

19. Vai trò của encoder trong D11

Trong D11:

x, edge_index, edge_attr
→ SharedPixelEncoder
→ encoded_x / dense_x
→ Local Branch: SlotAttentionFACS
→ Global Branch: VirtualNodeGather

Code D11 gọi:

encoded_x = self.encoder(x, edge_index, edge_attr)

rồi chuyển sang dense representation để đưa vào Slot Attention và Gather.

Vậy trong D11, encoder là nguồn chung cho cả Local và Global:

Local nhìn h_pixel để tạo local_raw
Global nhìn h_pixel để tạo global_context

Điều này có nghĩa là nếu encoder học tốt cấu trúc pixel-level, cả hai nhánh đều được lợi.

20. Điểm mạnh của encoder này

Thứ nhất, nó tận dụng edge_attr, rất hợp với pixel graph. Không chỉ biết pixel nào nối với pixel nào, nó còn biết cạnh đó có hướng, khoảng cách, chênh lệch intensity, similarity.

Thứ hai, gate là vector theo hidden_dim, nên khả năng lọc thông tin khá tinh tế. Một cạnh có thể chặn kênh này nhưng cho kênh khác đi qua.

Thứ ba, residual + LayerNorm + FFN giúp layer ổn định hơn một mean aggregation đơn giản.

Thứ tư, nó nhẹ hơn GAT/Transformer toàn cục. Với graph 2304 node và khoảng 17860 edge, đây là lựa chọn hợp lý cho Kaggle/T4.

21. Điểm yếu của encoder này

Thứ nhất, gate chỉ dựa vào edge_attr, không dựa trực tiếp vào cặp h_src, h_dst. Nghĩa là gate ít “ngữ cảnh động” hơn GAT/GATv2.

Thứ hai, nó vẫn local. Ba layer chỉ mở rộng receptive field cục bộ, không giải quyết quan hệ xa mắt-miệng.

Thứ ba, nếu edge_attr scale sai hoặc nghèo, gate sẽ kém. Ví dụ delta_intensity quá nhỏ sau normalization, edge_gate khó học ranh giới.

Thứ tư, mean aggregation vẫn có thể làm smoothing. Gate giảm rủi ro nhưng không triệt tiêu.

Thứ năm, nếu dùng nhiều layer hơn, over-smoothing vẫn có thể xảy ra, nhất là với ảnh 48×48 nhỏ và node feature nghèo.

22. Hình dung trực quan

Hãy tưởng tượng mỗi pixel là một người trong lưới 48×48.

edge_index nói:

ai được nói chuyện với ai

edge_attr nói:

quan hệ giữa hai người là kiểu gì
gần/xa, hướng nào, giống/khác intensity ra sao

EdgeAwarePixelMessageLayer làm:

người nguồn chuẩn bị một tin nhắn
cạnh giữa hai người quyết định tin nhắn đó được truyền mạnh/yếu ở từng kênh
người đích gom tin nhắn từ các hàng xóm
người đích cập nhật hiểu biết của mình

Sau vài lớp, mỗi pixel không chỉ biết bản thân nó sáng/tối thế nào, mà biết vùng nhỏ xung quanh nó có cấu trúc gì.

23. Một câu mô tả chuẩn cho báo cáo

Bạn có thể viết:

Khối encoder của D10/D11 sử dụng SharedPixelEncoder, bao gồm một phép chiếu đặc trưng node ban đầu lên không gian ẩn và nhiều lớp EdgeAwarePixelMessageLayer. Ở mỗi lớp, thông điệp từ node nguồn được tạo bởi một MLP và được điều biến bởi cổng sigmoid học từ edge attributes. Các thông điệp sau đó được gom trung bình về node đích, đi qua MLP, residual connection, LayerNorm và FFN. Cơ chế này cho phép mô hình khai thác quan hệ cục bộ giữa các pixel dựa trên hướng cạnh, khoảng cách và chênh lệch cường độ sáng, phù hợp với biểu diễn đồ thị pixel của ảnh khuôn mặt.
24. Tóm tắt một dòng
SharedPixelEncoder = Linear projection + nhiều lớp EdgeAwarePixelMessageLayer.
EdgeAwarePixelMessageLayer = message từ node nguồn × gate học từ edge_attr → mean aggregate về node đích → residual + LayerNorm + FFN.

Nắm được câu này là bạn đã nắm được “móng nhà” của cả D10 và D11.

Có 2 câu trả lời quan trọng ở đây:

Một là: encoder hiện tại đúng là có điểm yếu “local”, nhưng không nên xem nó là sai. Nó đang làm đúng vai trò: trích đặc trưng cục bộ pixel-level.

Hai là: thêm Global vào D11 có giúp, nhưng chỉ giúp theo kiểu ngữ cảnh hóa / điều chỉnh motif, không tự động làm motif tốt nếu local encoder hoặc slot chưa học được bằng chứng tốt.

Nói cách khác:

Encoder local yếu về quan hệ xa.
Global branch giúp bù quan hệ xa ở tầng readout/fusion.
Nhưng Global không thay thế được việc Local Motif phải học được vùng mắt/miệng/lông mày có ý nghĩa.
1. Nhìn lại encoder hiện tại đang làm gì

Hiện tại D10/D11 dùng SharedPixelEncoder, gồm input_proj rồi nhiều lớp EdgeAwarePixelMessageLayer. Trong code, SharedPixelEncoder project node feature lên hidden dim rồi chạy từng edge-aware layer qua edge_index, edge_attr, node_mask.

Lõi EdgeAwarePixelMessageLayer làm:

src = edge_index[0]
dst = edge_index[1]
h_src = h.index_select(dim=1, index=src)
gate = sigmoid(edge_gate(edge_attr))
msg = msg_mlp(h_src) * gate
agg.index_add_(dim=1, index=dst, source=msg)
agg = agg / degree
h = norm_msg(h + agg_mlp(agg))
h = norm_ffn(h + ffn(h))

Đoạn này xác nhận encoder là edge-gated local message passing, không phải GAT/GCN thuần.

D10 sau đó đưa embedding pixel vào Slot Attention. D10 tự mô tả pipeline là:

Pixel graph
→ Deep EdgeAware GNN Encoder
→ Iterative Slot Attention
→ Motif Relation Transformer
→ Class-Motif Attention

D11 cũng dùng SharedPixelEncoder, rồi tách ra Local Branch và Global Branch.

Vì vậy encoder này là móng nhà chung, còn D10/D11 khác nhau ở tầng đọc embedding sau encoder.

2. Điểm yếu 1: encoder vẫn local

Đây là điểm yếu bản chất.

Trong graph 48×48 8-neighbor, một node chỉ nối với các pixel gần nó. Sau 1 layer, pixel chỉ biết hàng xóm 1 bước. Sau 3 layer, nó chỉ biết vùng quanh nó vài pixel. D10 có pixel_gnn_layers=3 mặc định, tức là receptive field cục bộ có mở rộng nhưng vẫn chưa đủ để mắt và miệng “giao tiếp trực tiếp” ở tầng pixel.

Điều này tạo ra vấn đề:

Pixel vùng mắt có thể hiểu mí mắt/lông mày gần nó.
Pixel vùng miệng có thể hiểu viền môi/khoé môi gần nó.
Nhưng mắt không biết miệng đang cười, há, mím hay trễ.

Với FER, nhiều class không thể quyết định bằng một vùng đơn lẻ:

Fear vs Surprise: đều có mắt mở, miệng mở.
Sad vs Neutral: khác biệt nhẹ ở mắt + khóe miệng + tổng thể mặt.
Angry vs Sad: đều có vùng mắt/lông mày căng, cần thêm miệng và toàn mặt.

Vậy local encoder tốt cho viền, nếp, vùng nhỏ, nhưng yếu cho quan hệ bộ phận xa.

Khắc phục thế nào?

Có 4 hướng, theo mức độ rủi ro tăng dần.

Hướng A: Giữ encoder local, thêm tầng motif relation

D10 đã làm hướng này: sau Slot Attention, các motif đi qua MotifRelationTransformer. Ý tưởng là:

Encoder học local pixel feature.
Slot Attention gom pixel thành motif.
Transformer học quan hệ giữa motif.

Đây là hợp lý, vì không nhất thiết pixel mắt phải nói chuyện trực tiếp với pixel miệng. Chỉ cần motif mắt nói chuyện với motif miệng sau khi được gom.

Vấn đề là nếu slot chưa gom đúng motif, Transformer phía sau cũng học trên motif yếu.

Hướng B: Thêm global ở readout như D11A

D11A dùng VirtualNodeGather: attention trên toàn bộ pixel embeddings để tạo global_context, rồi FiLM điều biến local_raw. Code hiện tại đúng theo hướng này: VirtualNodeGather softmax trên N node, gom thành global vector; FiLMFusion dùng global sinh gamma, beta để điều chỉnh local.

Hướng này giúp vì:

Local motif vẫn do encoder + slot tạo ra.
Global context nhìn toàn ảnh.
FiLM dùng toàn cảnh để tăng/giảm motif.

Ví dụ local thấy “miệng há”. Global có thể giúp điều chỉnh:

miệng há + mắt mở to toàn mặt → Surprise
miệng há + mặt căng + mắt sợ → Fear

Nhưng nó không sửa được tận gốc encoder local, vì pixel nodes vẫn không nhận global context trước khi thành slot. Nó chỉ sửa ở giai đoạn sau.

Hướng C: Late global broadcast

Cho global broadcast ngược lại pixel/motif ở layer cuối hoặc sau slot, nhưng rất nhẹ.

Ví dụ:

h_pixel_local = EdgeAwareEncoder(x)
global = Gather(h_pixel_local)
h_pixel_refined = h_pixel_local + small_gate * Project(global)

Cái này có thể giúp pixel embeddings biết context hơn trước khi slot attention. Nhưng rủi ro là over-smoothing: mọi pixel nhận cùng tín hiệu global, slot mất sắc nét. Tài liệu D11 đã đặt luật “không broadcast sớm” để tránh nhòe đặc trưng.

Hướng D: Encoder lai local-global

Thay encoder bằng Graph Transformer, GATv2, hoặc thêm long-range edges. Đây là hướng mạnh nhưng rủi ro cao vì thay đổi cả móng nhà. Không nên làm ngay khi D11 còn đang được kiểm chứng.

3. Vậy thêm Global có giúp thật không?

Có, nhưng phải hiểu đúng nó giúp ở tầng nào.

D11 hiện tại không làm cho pixel mắt và pixel miệng trao đổi trực tiếp trong encoder. D11A là read-only global gather, nên nó không mở rộng receptive field của từng pixel node. Nó giúp ở tầng sau:

local_raw = slot motifs từ pixel embeddings local
global_context = tóm tắt toàn mặt
local_refined = FiLM(local_raw, global_context)

Công thức thiết kế D11 là:

local_refined = local_raw * (1 + tanh(W_gamma(global_context))) + W_beta(global_context)

Tài liệu D11 cũng nhấn mạnh Global chỉ điều chỉnh Local, không đi thẳng vào classifier.

Vì vậy, Global giúp thật nếu local đã có tín hiệu cơ bản:

Local phát hiện được miệng/mắt/lông mày.
Global giúp diễn giải tổ hợp đó.

Nhưng Global không giúp nhiều nếu local motif rỗng:

Slot nhìn nền/tóc.
Slot collapse vào cùng một vùng.
Local_raw không phân biệt class.

Khi đó Global dễ thành shortcut hoặc chỉ vá lỗi hời hợt. Tài liệu D11 đã liệt kê rủi ro này: local death, gate shortcut, slot collapse, attention nhìn nền.

Câu chốt:

Global giúp giải quyết thiếu bối cảnh.
Global không thay thế được motif.
Nếu motif yếu, Global có thể che lỗi nhưng không làm hướng nghiên cứu mạnh hơn.
4. Điểm yếu 2: gate chỉ dựa vào edge_attr, không phụ thuộc động vào h_src/h_dst

Encoder hiện tại tính gate từ cạnh:

gate = sigmoid(edge_gate(edge_attr))
msg = msg_mlp(h_src) * gate

Điều này nghĩa là gate chủ yếu hỏi:

Cạnh này có đặc tính gì?
dx, dy, dist, delta_intensity, similarity ra sao?

Nhưng nó không trực tiếp hỏi:

Trong ngữ cảnh hiện tại, node nguồn và node đích đang biểu diễn gì?

Đây là khác biệt với attention động. Ví dụ cùng một kiểu cạnh “delta_intensity cao” có thể là:

viền môi có ích
nhiễu do ánh sáng
rìa tóc
biên nền

Nếu gate chỉ nhìn edge_attr, nó khó phân biệt đầy đủ ngữ cảnh.

Cách khắc phục

Không cần bỏ encoder. Có thể nâng cấp thành context-aware edge gate:

gate_ij = sigmoid(MLP([edge_attr_ij, h_src_i, h_dst_j, h_src_i - h_dst_j]))

Hoặc nhẹ hơn:

edge_gate = sigmoid(MLP_edge(edge_attr) + MLP_node([h_src, h_dst]))

Như vậy gate vẫn tận dụng edge_attr nhưng có thêm ngữ cảnh node.

Rủi ro: tăng compute và dễ overfit. Nên đây là ablation sau, không phải ưu tiên trước SupCon/slot.

5. Điểm yếu 3: mean aggregation có thể làm mất cấu trúc

Sau khi message được gate, layer gom về node đích bằng index_add_, rồi chia theo degree.

Về bản chất:

node đích nhận trung bình các message từ hàng xóm

Mean aggregation ổn định, nhưng có thể làm mất khác biệt giữa neighbor quan trọng và neighbor ít quan trọng. Gate có giúp, nhưng nếu nhiều cạnh gate tương tự nhau, cuối cùng vẫn là trung bình.

Vấn đề này đặc biệt rõ ở vùng cấu trúc mảnh:

khóe miệng
mí mắt
nếp nhăn nhỏ
rãnh mũi-miệng

Các chi tiết này dễ bị làm mượt.

Cách khắc phục

Có 3 hướng:

1. Dùng weighted mean theo gate magnitude thay vì degree mean.
2. Thêm residual mạnh hơn từ h gốc.
3. Thêm edge-aware attention nhẹ sau gate.

Ví dụ hiện tại degree mean chia theo số cạnh, không chia theo tổng gate. Một biến thể có thể là:

agg_j = sum_i gate_ij * msg_i / sum_i gate_ij

Cách này hợp lý hơn về mặt “cạnh nào mở nhiều thì ảnh hưởng nhiều”. Nhưng cần cẩn thận vì gate là vector hidden_dim, không scalar; phải quyết định normalize theo scalar gate hay theo channel.

6. Điểm yếu 4: over-smoothing nếu tăng layer

D10 muốn tăng receptive field nên tăng pixel_gnn_layers. Nhưng tăng layer làm node embeddings dễ giống nhau hơn. Encoder hiện có residual + LayerNorm + FFN, giúp ổn định, nhưng không triệt tiêu over-smoothing.

Bài toán rất khó cân bằng:

ít layer → thiếu ngữ cảnh cục bộ đủ rộng
nhiều layer → node bị nhòe, slot khó phân vùng
Cách khắc phục

Hướng hợp lý nhất không phải cứ tăng layer, mà là multi-scale.

D10 code đã có flag multi_scale_gnn: nếu bật, nó tạo thêm encoder_aux với nhiều hơn 1 layer và combine hai scale lại.

Triết lý này đúng:

encoder nông giữ chi tiết cạnh
encoder sâu lấy ngữ cảnh rộng hơn
combine lại

Nên nếu muốn khắc phục local-receptive-field mà không làm nhòe quá mạnh, multi-scale tốt hơn tăng layer đơn thuần.

Một hướng khác là dùng dilated / skip edges:

local 8-neighbor edges
+ edges cách 2 pixel
+ edges cách 4 pixel

Như vậy receptive field rộng hơn mà không cần quá nhiều layer. Nhưng thay đổi graph schema, cần rebuild artifact nếu edge_index/edge_attr đổi.

7. Điểm yếu 5: phụ thuộc mạnh vào chất lượng edge_attr

Encoder này sống nhờ edge_attr. Nếu edge_attr tốt, gate có ý nghĩa. Nếu edge_attr sai hoặc scale kém, gate trở thành nhiễu.

Ví dụ:

delta_intensity bị scale quá nhỏ → gate khó thấy biên
intensity_similarity thiết kế không tốt → không phân biệt vùng trơn/biên
dx/dy/dist áp đảo intensity → gate học hướng cạnh thay vì cấu trúc biểu cảm

Bạn từng nói thầy lo việc scale intensity 255 về 0–1 làm delta yếu đi. Đây là vấn đề thật với encoder này, vì edge_gate phụ thuộc trực tiếp vào edge_attr.

Cách khắc phục

Không nhất thiết “không scale tất cả”. Cách tốt hơn là scale có kiểm soát theo nhóm feature:

tọa độ/distance: normalize
intensity/delta: giữ dynamic range hoặc chuẩn hóa riêng
similarity: thiết kế sao cho phân biệt tốt

Quan trọng nhất là log phân phối:

histogram delta_intensity
histogram gate values
gate theo vùng biên vs vùng trơn

Nếu gate toàn quanh 0.5 thì edge_gate gần như không học được gì.

8. Điểm yếu 6: encoder không có face prior

Pixel graph chỉ biết lưới ảnh, không biết đâu là mặt, mắt, miệng. Nó sẽ học từ dữ liệu:

vùng nền/tóc cũng là node bình thường
rìa ảnh cũng là node bình thường
cạnh ở tóc có thể mạnh như cạnh ở mắt

Vì vậy encoder có thể tạo embedding tốt cho background, rồi slot/gather bị hút vào background.

D11 tài liệu cũng cảnh báo Global branch có thể học ánh sáng, crop, viền đen thay vì khuôn mặt.

Cách khắc phục

Có thể thêm face-aware bias nhẹ, không hard-code quá mạnh:

center prior cho node/gate
border penalty
mask giảm trọng số vùng góc/rìa
augmentation crop/translation để giảm shortcut nền

Với D11, Virtual attention phải kiểm tra bằng heatmap và background sensitivity. Đây đã nằm trong test suite.

9. Điểm yếu 7: encoder không tự tạo semantic cao cấp

Đây là điểm lớn.

Encoder hiện tại làm tốt pixel-level local structure. Nhưng từ intensity + edge_attr để ra semantic như “mắt mở”, “miệng cười”, “lông mày cau” là rất khó, nhất là ảnh 48×48.

D11 tài liệu cũng ghi rõ: D11 giải quyết thiếu global context, không giải quyết triệt để node/edge feature nghèo.

Cách khắc phục

Có 3 mức:

Mức 1: làm giàu node/edge handcrafted feature.
Mức 2: thêm local patch descriptor nhẹ quanh mỗi node.
Mức 3: hybrid CNN/Swin feature rồi đưa vào graph.

Mức 1 hợp hướng graph thuần nhất:

gradient magnitude
local mean/std
LBP-like texture
Sobel x/y
distance to center
border prior

Mức 2 mạnh hơn:

mỗi node có descriptor từ patch 3×3 hoặc 5×5

Mức 3 mạnh nhất nhưng làm hướng graph không còn thuần:

CNN/Swin tạo feature map
pixel graph dùng feature đó làm node feature

Nếu mục tiêu là metric cao, mức 3 có thể cần thiết. Nếu mục tiêu là nghiên cứu pixel graph/motif, mức 1-2 hợp hơn.

10. Điểm yếu 8: global có thể làm yếu motif không?

Có. Đây là rủi ro cực lớn.

Nếu thêm global sai cách, classifier có thể dựa vào global hơn local. Khi đó:

metric tăng
nhưng motif không có ý nghĩa

D11 hiện tại cố tránh bằng FiLM thay vì concat, bottleneck/dropout, auxiliary local loss. Tài liệu D11 đã ghi rõ các cơ chế này.

Nhưng dù FiLM, vẫn có shortcut gián tiếp:

gamma/beta có thể mã hóa class
local_refined bị global điều khiển quá mạnh
gate pattern trở thành đáp án

Vì vậy phải kiểm tra:

Zero-Global
Zero-Local
Gate-only
F1_Raw_Local
Shuffle-Global
Drop Top-Slot

Các tiêu chí này cũng đã nằm trong tài liệu D11.

Câu trả lời thẳng cho câu hỏi của bạn:

Có, Global có thể giúp quan hệ xa.
Nhưng nếu Local motif yếu, Global có thể làm motif yếu hơn về mặt nghiên cứu, vì nó che lỗi Local.
D11 chỉ thành công khi Global giúp Local, không thay Local.
11. Ngoài các vấn đề đã nói, còn vấn đề lớn nào nữa?

Có 5 vấn đề lớn nữa.

Vấn đề A: Slot attention không phải detector giải phẫu thật

D10/D11 dùng slot attention. Slot có thể nhìn mắt/miệng, nhưng cũng có thể nhìn tóc/nền/contrast mạnh. D11 tài liệu cũng nhấn mạnh local branch không được giả định chắc chắn slot tương ứng với mắt/miệng; phải kiểm chứng bằng visualization và causal tests.

Khắc phục:

FACS prior mềm
SupCon trên local_raw
slot diversity nhẹ
Drop-slot causal test
slot consistency under augmentation
Vấn đề B: 1-stage mất lực định hình representation

Bạn đã thấy D11 1-stage chỉ khoảng 0.47 F1, thấp hơn D10 2-stage. Đây không chỉ là vấn đề encoder. Đây là vấn đề objective.

Cross-Entropy chỉ ép logits đúng. Nó không ép embedding local có cấu trúc class. SupCon làm được việc này.

Khắc phục chính:

SupCon trên pooled(local_raw)
warm-up lambda_supcon
theo dõi F1_Raw_Local
Vấn đề C: FER-2013 nhãn nhiễu và class imbalance

Mô hình càng mạnh càng dễ học nhãn nhiễu. Global branch có thể học bias dataset. Class nhỏ như Disgust/Fear dễ bị bỏ.

Khắc phục:

label smoothing
class weights hoặc balanced sampler
per-class F1
confusion delta
early stopping theo macro F1
Vấn đề D: interpretability ảo

Attention map đẹp không chứng minh model dùng vùng đó. Cần causal masking/drop slot.

Khắc phục:

mask vùng attention cao
drop top slot
shuffle global
gate-only probe
background sensitivity
Vấn đề E: schema/data pipeline sai làm kết quả vô nghĩa

Với graph pixel, chỉ cần sai coordinate index, edge_attr scale, batch reshape, hoặc mask là slot center và spatial loss sai hết. D11 code dùng coord_indices=(1,2) để lấy x_norm/y_norm cho FACS center; nếu node feature order sai thì spatial prior sẽ phạt nhầm.

Khắc phục:

unit test visualize image từ x
unit test coords index
unit test edge_index count = 17860
unit test edge_attr khớp edge_index
unit test slot center map đúng orientation
12. Nên sửa encoder ngay không?

Mình không khuyên sửa encoder ngay.

Hiện tại bottleneck lớn hơn có vẻ là:

local motif chưa được định hình mạnh trong 1-stage
SupCon thiếu
slot collapse / spatial prior / diversity chưa cân bằng
global cần chứng minh không shortcut

Encoder hiện tại hợp bài. Nó là local backbone tốt. Việc thay encoder bây giờ sẽ làm loạn biến số.

Thứ tự đúng:

1. Giữ EdgeAware encoder.
2. Cứu local representation bằng SupCon local_raw.
3. Kiểm tra F1_Raw_Local, slot visualization, ablation.
4. Nếu local vẫn yếu, mới nghiên cứu nâng encoder.
5. Nếu nâng encoder, ưu tiên multi-scale hoặc context-aware edge gate, chưa nhảy thẳng Graph Transformer.
13. Kết luận chiến lược

Encoder hiện tại có điểm yếu thật:

local receptive field
gate chưa context-aware
mean aggregation có thể làm mượt
phụ thuộc edge_attr
không có face prior
không tự tạo semantic cao cấp

Nhưng nó vẫn là một backbone hợp lý cho pixel graph.

D11 Global có giúp thật, nhưng giúp ở tầng diễn giải motif, không thay thế nhiệm vụ học motif. Vì vậy:

Nếu Local Motif tốt → Global giúp phân biệt class khó.
Nếu Local Motif yếu → Global dễ thành shortcut hoặc vá lỗi giả.

Hướng đúng nhất hiện tại không phải “bỏ encoder”, mà là:

Giữ encoder edge-aware.
Tăng áp lực học local motif bằng SupCon.
Dùng Global như FiLM context có kiểm soát.
Đánh giá bằng F1_Raw_Local, Zero-Global, Zero-Local, Gate-only, Drop-slot.
Sau đó mới cân nhắc nâng encoder bằng multi-scale/context-aware gate.

Đây là câu hỏi rất quan trọng, vì nó quyết định ta nên vá encoder hiện tại hay thiết kế encoder thế hệ mới. Mình trả lời theo hướng nghiên cứu, có phân biệt rõ phần có cơ sở từ paper/code và phần là đề xuất cần ablation.

1. GAT thuần có xử lý local và global không?

GAT thuần chủ yếu xử lý local, không phải global thật sự.

Paper GAT mô tả nó là các lớp masked self-attention trên graph, trong đó node attend tới các node trong neighborhood của nó. Tức là attention bị giới hạn bởi edge_index, không phải mọi node nhìn mọi node như Transformer toàn cục.

Với pixel graph 8-neighbor:

GAT layer 1:
pixel chỉ attend tới tối đa 8 hàng xóm

GAT layer 2:
thông tin lan xa hơn 2-hop

GAT layer L:
thông tin lan trong L-hop

Vậy GAT thuần vẫn là local message passing. Nó chỉ khác encoder hiện tại ở cách tính trọng số neighbor:

EdgeAware hiện tại:
gate_ij = sigmoid(MLP(edge_attr_ij))

GAT thuần:
alpha_ij = softmax(score(h_i, h_j))

GAT không tự có global context trừ khi ta thêm:

virtual node
long-range edges
global attention layer
graph transformer
pooling/global token

Vậy câu trả lời ngắn gọn:

GAT thuần = local attention theo cạnh.
GAT + virtual/global token/fully-connected attention = mới có global.
2. Nếu nâng cấp encoder thành local-global thì giúp gì?

Nâng encoder local-global có thể giúp đúng vào điểm yếu hiện tại: motif đang được tạo từ node embedding còn quá cục bộ.

Hiện tại D11A thêm Global sau encoder:

EdgeAware local encoder
→ local_raw từ Slot Attention
→ global_context từ Virtual Gather
→ FiLM điều chỉnh local_raw

Như vậy Global chỉ giúp ở readout/fusion, không làm h_pixel giàu quan hệ xa trước khi Slot Attention. Trong code D11, encoded_x = self.encoder(...), sau đó cùng dense_x được đưa vào Slot Attention và VirtualNodeGather.

Nếu nâng encoder thành local-global, luồng sẽ thành:

Pixel graph
→ Local Edge-Aware layers
→ Global Context Injection nhẹ
→ h_pixel đã biết một phần quan hệ xa
→ Slot Attention
→ motif giàu hơn

Lợi ích kỳ vọng:

Thứ nhất, motif đỡ “cục bộ mù”. Slot vùng mắt có thể chứa dấu hiệu “toàn mặt đang há miệng / căng / rũ” ở mức nhẹ, giúp motif mắt không bị diễn giải đơn độc.

Thứ hai, Slot Attention dễ gom part hơn. Nếu pixel embeddings đã mang context rộng hơn, các pixel thuộc cùng vùng biểu cảm có thể trở nên nhất quán hơn.

Thứ ba, giảm gánh nặng cho FiLM cuối. D11A hiện tại để Global sửa ở cuối. Encoder local-global cho phép Global hỗ trợ sớm hơn, nhưng nếu làm quá mạnh sẽ gây nhòe.

Thứ tư, class khó có cơ hội cải thiện. FER-2013 thường gặp nhầm Fear/Surprise, Sad/Neutral, Angry/Sad do biểu cảm cục bộ gần nhau, ảnh 48×48, pose/occlusion/nhãn nhiễu. Các nguồn mô tả FER-2013 là ảnh grayscale 48×48, có biến thiên pose/occlusion, chất lượng thấp, nhãn nhiễu và class imbalance.

Nhưng rủi ro cũng lớn:

global vào quá sớm → h_pixel bị giống nhau
slot mất sắc nét
motif chết
Global shortcut xuất hiện ngay từ encoder

Nên encoder local-global phải là controlled global, không phải broadcast mạnh.

3. Vấn đề 1 + 2 nên nâng cấp chung thế nào?

Bạn nói muốn nâng cấp vấn đề 1 và 2 cùng lúc:

Vấn đề 1: encoder local, thiếu quan hệ xa
Vấn đề 2: gate chỉ dựa edge_attr, không context-aware theo h_src/h_dst

Mình đề xuất hướng hợp lý nhất là:

Context-Aware Edge-Gated Encoder + Read-only Global Token

Nó giữ ưu điểm encoder hiện tại, nhưng nâng lên 2 điểm:

1. Local edge gate không chỉ nhìn edge_attr,
   mà nhìn thêm h_src/h_dst.

2. Có global token đọc toàn ảnh,
   nhưng broadcast rất hạn chế hoặc chỉ late-modulation.

Công thức encoder hiện tại:

g_ij = sigmoid(MLP_e(e_ij))
m_ij = MLP_msg(h_i) ⊙ g_ij
agg_j = mean_i(m_ij)

Bản nâng cấp:

g_ij = sigmoid(MLP_g([e_ij, h_i, h_j, h_i - h_j, h_i ⊙ h_j]))

m_ij = MLP_msg(h_i) ⊙ g_ij

agg_j = weighted_mean_i(m_ij, gate_strength)

global = AttentiveGather(h)

h_j' = LocalUpdate(h_j, agg_j)

h_j'' = h_j' + λ_global * FiLM_or_gate(h_j', global)

Trong đó λ_global phải nhỏ hoặc học được, ví dụ bắt đầu gần 0.

Điểm hay:

local vẫn là chính
edge_attr vẫn được dùng
gate trở nên context-aware
global có mặt trong encoder nhưng bị kiểm soát

Đây là nâng cấp hợp logic nhất từ encoder hiện tại, không phải thay toàn bộ bằng GAT/Graph Transformer.

4. Xử lý vấn đề 3: mean aggregation làm mất cấu trúc

Vấn đề 3 là hiện tại sau gate, layer dùng index_add_ rồi chia theo degree. Cơ chế này có trong EdgeAwarePixelMessageLayer: message được cộng vào node đích, sau đó chia trung bình theo degree.

Điểm yếu: degree mean coi mọi cạnh đã tồn tại như nhau ở bước normalize. Trong khi gate đã nói cạnh nào mở mạnh/yếu.

Cách xử lý hợp FER-2013 nhất là Gate-normalized Aggregation, không phải thay bằng attention phức tạp ngay.

Hiện tại:

agg_j = sum_i m_ij / degree_j

Nên đổi ý tưởng thành:

s_ij = mean_channel(g_ij) hoặc scalar_gate_ij

agg_j = sum_i s_ij * msg_raw_ij / (sum_i s_ij + eps)

Hoặc nếu giữ vector gate:

agg_j = sum_i (msg_raw_ij ⊙ g_ij) / (sum_i g_ij + eps)

Vì sao hợp FER-2013?

FER-2013 ảnh nhỏ 48×48, nhiễu, pose/occlusion; nếu dùng attention quá mạnh dễ overfit. Gate-normalized aggregation là thay đổi nhỏ, giữ inductive bias cục bộ, nhưng làm aggregation nhất quán hơn với gate.

Nó có cơ sở không? Có, về mặt nguyên lý nó gần với attention-weighted aggregation: message có trọng số thì normalization cũng nên theo tổng trọng số, không phải chỉ theo số neighbor. GAT cũng dùng softmax để chuẩn hóa trọng số neighbor, còn Graph Attention Networks có cơ sở rõ ràng là gán trọng số khác nhau cho các node trong neighborhood.

Tuy nhiên, đây vẫn là đề xuất cần ablation trong code của bạn, không thể khẳng định chắc chắn sẽ tăng F1.

5. Các cách giải quyết có cơ sở không, hay chỉ đoán?

Mình phân loại rõ:

Có cơ sở mạnh từ literature

Local attention/gating trên neighbor có cơ sở từ GAT/MPNN. GAT cho thấy masked self-attention trên neighborhood là hướng chuẩn trong graph learning.

Global token/Graph Transformer có cơ sở từ Graphormer/Graph Transformer: Transformer cần structural encoding để làm việc tốt trên graph, và Graphormer cho thấy global attention có thể hiệu quả khi graph structure được mã hóa đúng.

Label smoothing/class balancing/augmentation cho FER-2013 có cơ sở vì FER-2013 được mô tả là low-resolution, noisy labels, class imbalance, pose/occlusion.

Có cơ sở logic mạnh từ code và kiến trúc hiện tại

Context-aware edge gate là mở rộng trực tiếp từ code hiện tại. Vì gate hiện tại chỉ nhìn edge_attr, thêm h_src/h_dst là cách làm gate phụ thuộc ngữ cảnh hơn.

Gate-normalized aggregation là sửa nhất quán với bản chất gate. Hiện tại đã có gate nhưng normalize theo degree; normalize theo tổng gate hợp lý hơn.

Late/controlled global injection là nhất quán với chính tài liệu D11: tránh broadcast sớm vì over-smoothing, dùng Global để điều biến Local chứ không thay Local.

Còn là giả thuyết cần thực nghiệm

Các mức tăng F1, việc Fear/Disgust tăng bao nhiêu, hay encoder mới có vượt D11A hiện tại không là chưa thể khẳng định. Với FER-2013, nhãn nhiễu và ảnh nhỏ khiến nhiều ý tưởng đúng lý thuyết vẫn có thể không tăng metric.

Vì vậy cách nói đúng là:

Các đề xuất có cơ sở về mặt GNN/literature và phù hợp với code hiện tại,
nhưng phải được xác nhận bằng ablation:
EdgeAware baseline
+ gate-normalized
+ context-aware gate
+ local-global injection
6. Vấn đề 7: hướng giải quyết thuần GNN cho “semantic feature nghèo”

Bạn muốn thuần GNN, pixel-to-pixel, chưa dùng CNN. Vậy không nên đưa CNN backbone. Nhưng có thể mượn ý tưởng CNN/Swin theo dạng graph.

Mình đề xuất 4 hướng thuần GNN, từ an toàn đến mạnh.

Hướng 7A: Multi-scale Pixel Graph Encoder

Ý tưởng từ CNN: nhiều receptive field.

Không dùng CNN. Ta vẫn dùng graph, nhưng có nhiều loại edge:

scale 1: 8-neighbor
scale 2: neighbor cách 2 pixel
scale 4: neighbor cách 4 pixel hoặc window-neighbor

Luồng:

h1 = EdgeAwareGNN(scale=1)
h2 = DilatedEdgeGNN(scale=2)
h4 = DilatedEdgeGNN(scale=4)
h = Fuse([h1, h2, h4])

Tác dụng:

scale 1 giữ chi tiết mí mắt/viền môi
scale 2-4 lấy cấu trúc vùng mắt/miệng rộng hơn
slot nhận embedding giàu hơn

Đây là hướng thuần GNN và hợp FER-2013 hơn tăng layer sâu, vì ảnh nhỏ dễ over-smoothing.

Nhược điểm: nếu thêm edge mới thì phải rebuild artifact hoặc sinh edge dilated trong model.

Hướng 7B: Window Graph Attention kiểu Swin nhưng pixel-to-pixel

Bạn nói D8 từng dùng Swin, nhưng muốn pixel-to-pixel. Vậy ta không dùng Swin image backbone, mà dùng windowed graph attention trên pixel nodes.

Luồng:

h_pixel từ EdgeAware local layer
→ chia ảnh thành cửa sổ 6×6 hoặc 8×8
→ trong mỗi window: pixel-to-pixel self-attention
→ shifted window: dịch cửa sổ để nối vùng lân cận
→ trả lại h_pixel

Đây là mượn ý tưởng Swin nhưng vẫn là pixel node attention. Không phải CNN.

Tác dụng:

mỗi pixel nhìn được cả window, không chỉ 8 neighbor
shifted window giúp thông tin qua biên window
rất hợp 48×48 vì có thể chia 8×8 hoặc 6×6

Nhược điểm: compute tăng. Nhưng 48×48 nhỏ, window attention có thể chấp nhận.

Đây là ứng viên rất mạnh cho “encoder local-global nhẹ”.

Hướng 7C: Landmark-free Structural Tokens thuần GNN

Không dùng landmark detector. Ta tạo một số “region tokens” học được:

eye-like token
mouth-like token
brow-like token
center-face token

Nhưng không gán cứng. Chúng attend/gather từ pixel nodes bằng soft assignment, giống slot nhưng nằm trong encoder.

Luồng:

Pixel nodes
→ EdgeAware local encoding
→ Region tokens gather pixels
→ Region tokens exchange information
→ Region tokens modulate pixel nodes nhẹ
→ Slot Attention cuối

Đây gần D11 nhưng global không chỉ một token, mà nhiều structural tokens.

Tác dụng:

không cần landmark
có part-level context trước slot
vẫn thuần graph/attention

Rủi ro: trùng với slot attention, dễ collapse nếu không có diversity/spatial prior.

Hướng 7D: Graph Positional/Structural Encoding mạnh hơn

Hiện node có x/y, edge có dx/dy/dist/delta. Nhưng có thể thêm structural encoding thuần graph:

distance to center
radial distance
angle from center
border distance
multi-scale coordinates
Laplacian positional encoding

Graphormer nhấn mạnh việc encoding structural information là chìa khóa để Transformer hoạt động tốt trên graph.

Với FER-2013, face thường roughly centered theo mô tả dataset, nên center-aware features có cơ sở nhưng phải mềm, vì crop vẫn lệch. Kaggle mô tả ảnh FER là 48×48 grayscale và mặt được registered sao cho tương đối centered.

Đây là hướng ít rủi ro nhất vì không thay kiến trúc quá mạnh.

7. Kiến trúc encoder local-global mình đề xuất nhất

Mình không chọn GAT thuần. Mình chọn:

D12 Encoder: Multi-Scale Context-Aware Edge-Gated Pixel Transformer

Nhưng phiên bản đầu nên đơn giản:

Input x, edge_index, edge_attr
↓
Input projection
↓
Block 1: EdgeAware local message passing hiện tại
↓
Block 2: Context-Aware Edge Gate + Gate-normalized aggregation
↓
Block 3: Window Pixel Self-Attention hoặc Dilated Edge-GNN
↓
Read-only Global/Region Token gather
↓
Light FiLM modulation lên h_pixel hoặc lên local slots
↓
Slot Attention

Bản tối thiểu đáng thử:

SharedPixelEncoder_v2 =
  input_proj
  EdgeAwarePixelMessageLayer
  ContextAwareEdgeMessageLayer
  WindowPixelAttentionBlock

Trong đó:

ContextAwareEdgeMessageLayer xử lý vấn đề 2 + 3
WindowPixelAttentionBlock xử lý vấn đề 1 + 7
Global vẫn kiểm soát, không broadcast mạnh

Vì sao mình chọn window attention thay vì full global attention?

FER-2013 có ảnh nhỏ nhưng nhiễu. Full attention 2304×2304 có thể học bias nền/crop và tốn compute. Window attention cho pixel nhìn xa hơn 8-neighbor nhưng vẫn giữ locality. Shifted window giúp thông tin đi xa dần mà không trộn toàn ảnh quá sớm.

8. Thứ tự ablation hợp lý

Không nên nhảy thẳng vào encoder cực phức tạp. Đi theo bậc:

E0: Encoder hiện tại
E1: Gate-normalized aggregation
E2: Context-aware edge gate
E3: E2 + dilated/multi-scale edges
E4: E2 + window pixel attention
E5: E4 + read-only global token / region tokens

Metric cần xem không chỉ Full F1:

F1_Raw_Local
slot visualization
slot collapse score
over-smoothing cosine
Fear/Sad/Angry/Suirprise F1
Zero-Global / Zero-Local nếu dùng D11

Nếu E1/E2 đã cải thiện Raw Local, đó là tín hiệu rất tốt vì encoder đang làm motif mạnh hơn thật.

9. Kết luận chiến lược

GAT thuần không giải quyết global thật sự. Nó là local masked attention theo edge. Với bài của bạn, thay EdgeAware bằng GAT thuần chưa chắc tốt vì sẽ bỏ mất lợi thế edge_attr.

Nâng cấp encoder local-global có ích nếu mục tiêu là làm motif mạnh từ gốc, không chỉ để Global FiLM vá ở cuối. Hướng hợp lý nhất không phải “global hóa toàn bộ pixel”, mà là:

giữ EdgeAware local bias
+ context-aware edge gate
+ gate-normalized aggregation
+ multi-scale/window pixel-to-pixel attention
+ global/region token kiểm soát nhẹ

Phần có cơ sở mạnh là: GAT/Graph Transformer chứng minh attention trên graph/neighborhood và structural encoding là quan trọng; FER-2013 thật sự có ảnh 48×48, pose/occlusion, label noise, class imbalance; code hiện tại thật sự dùng edge-gated local message passing. Phần cần thực nghiệm là mức độ cải thiện trên D10/D11 của bạn.

Nếu chọn một hướng để phát triển sau SupCon/D11 ổn định, mình sẽ chọn:

Context-Aware Edge-Gated Encoder
+ Gate-normalized Aggregation
+ Windowed Pixel-to-Pixel Attention

Đây là thuần GNN/attention trên pixel, không dùng CNN backbone, nhưng vẫn mượn được ý tưởng multi-scale/local-window rất thành công của CNN/Swin.

Mình nghĩ nên chốt D11 ở mức Exp H như một mốc tốt, không tiếp tục vắt quá sâu D11 bằng tăng loss/slot/global nữa. Kết quả bạn gửi đã cho thấy D11 đã được tối ưu khá sát “vùng vàng”: Exp C tốt nhất trong nhóm SupCon+Div cơ bản, Exp H tốt nhất sau Loss Scheduling; các biến thể tăng SupCon, tăng Local, tăng Diversity, tăng Global Dim, tăng Slot đều giảm. Điều đó là tín hiệu rằng nút thắt không còn nằm chủ yếu ở loss/fusion nữa, mà nằm ở encoder/feature gốc. Trong code hiện tại, D10/D11 đều dùng SharedPixelEncoder, tức input projection rồi stack EdgeAwarePixelMessageLayer làm local edge-gated message passing; D11 chỉ thêm Local/Global/FILM ở phía sau encoder.

1. Có nên tiếp tục D11 hay chuyển D12?

Mình chọn: chuyển sang D12, nhưng không bỏ D11.

Cách đúng không phải “đập đi xây lại”, mà là:

D12 = D11 Exp H
    + encoder thế hệ mới

Tức là giữ các phần đã chứng minh có ích:

Slot Attention FACS
VirtualNodeGather
FiLM Global-Local
Motif Relation Transformer
Aux Local Loss
SupCon + Diversity Margin
CE warmup / Loss Scheduling kiểu Exp H

và thay phần đang nghẽn:

SharedPixelEncoder hiện tại
→ D12 Pixel Encoder mới

Lý do: Exp G No Global tụt, chứng minh global branch có ích; Exp J global_dim 128 sập, chứng minh bottleneck global là cần; Exp K 16 slots sập, chứng minh không nên tăng slot; Exp D/F/E đều sập, chứng minh loss ratio Exp C/H đã gần tối ưu. Vậy tiếp tục tuning D11 kiểu tăng lambda, tăng slots, tăng capacity rất dễ tốn thời gian mà không chạm đúng gốc.

Nên D11 Exp H trở thành baseline chính thức, còn D12 là thay encoder để giải quyết feature bottleneck/local receptive field.

2. Hướng 7A, 7B, 7D có thể tổng hợp không?

Có, và nên tổng hợp. Hướng đáng giá nhất không phải 7A riêng, không phải 7B riêng, mà là:

D12 Encoder =
Context-Aware Edge-Gated Multi-Scale Pixel Graph Encoder
+ Windowed Pixel-to-Pixel Attention
+ Graph Positional / Structural Encoding

Nói ngắn hơn:

D12 = Multi-Scale Graph-Swin Pixel Encoder for D11

Nhưng phải nhấn mạnh: đây không phải Swin image backbone như D8. Đây là pixel-to-pixel attention trên node graph, tức vẫn giữ triết lý GNN/pixel graph.

3. Vì sao 7A đáng làm?

7A là multi-scale pixel graph edges.

Hiện tại encoder chỉ dùng 8-neighbor. Sau 3 layer, node chỉ biết vùng cục bộ vài pixel. Điều này tốt cho viền môi/mí mắt/nếp nhăn, nhưng thiếu vùng rộng như toàn miệng, hai mắt, hoặc quan hệ mắt-miệng.

Multi-scale graph giải quyết bằng cách thêm cạnh dài hơn:

Scale 1: 8-neighbor, khoảng cách 1 pixel
Scale 2: neighbor cách 2 pixel
Scale 4: neighbor cách 4 pixel hoặc region-neighbor

Ưu điểm lớn:

Không cần CNN.
Không phá pixel graph.
Cho node thấy cấu trúc rộng hơn mà không cần tăng quá nhiều layer.
Giảm nguy cơ over-smoothing hơn việc stack 6-8 GNN layers.

Với FER-2013, ảnh 48×48 rất nhỏ, chất lượng thấp, nhiều biến thiên ánh sáng/pose/background; nguồn Kaggle mô tả dữ liệu là ảnh grayscale 48×48, mặt được căn tương đối ở trung tâm, còn các nghiên cứu gần đây cũng nhấn mạnh FER-2013 khó vì ảnh chất lượng thấp, pose/lighting/background, nhãn nhiễu và mất cân bằng lớp. Vì vậy multi-scale rất hợp: nó giúp lấy cấu trúc vùng mặt rộng hơn mà vẫn không đòi semantic feature từ CNN.

Nhược điểm “phải build lại artifact” không còn là vấn đề nếu bạn chấp nhận. Vậy nhược điểm thật sự của 7A là:

edge count tăng;
edge_attr phải thiết kế đúng cho nhiều scale;
nếu scale xa quá, message sẽ trộn qua vùng không liên quan;
cần phân biệt edge_type theo scale.

Vì vậy không nên chỉ nối thêm cạnh rồi dùng chung edge_gate. Nên có scale-aware edge encoding:

edge_attr = [dx, dy, dist, delta_intensity, similarity, scale_id]

hoặc có embedding riêng:

edge_type_embedding(scale=1/2/4)

Nếu không có scale_id, model khó biết cạnh xa và cạnh gần nên xử lý khác nhau.

4. Hướng 7B có hợp không?

7B hợp, nhưng chỉ khi làm đúng kiểu pixel-to-pixel window attention, không phải Swin backbone CNN/image.

Swin Transformer dùng shifted windows để giảm chi phí self-attention và tạo kết nối giữa các window; paper gốc nhấn mạnh shifted-window giúp attention hiệu quả hơn và vẫn cho phép cross-window connection.

Áp dụng vào bài bạn theo cách thuần graph:

h_pixel: [B, 2304, D]
→ reshape [B, 48, 48, D]
→ chia window 6×6 hoặc 8×8
→ trong mỗi window, pixel nodes self-attend với nhau
→ shifted window để thông tin qua biên
→ reshape lại [B, 2304, D]

Cái này vẫn là pixel-to-pixel vì attention diễn ra giữa các pixel nodes trong window.

Nó giúp vấn đề 7 tốt hơn 7A ở chỗ:

7A chỉ thêm cạnh cố định.
7B cho pixel tự học quan hệ trong một vùng 6×6/8×8.

Nhưng 7B có rủi ro:

attention trong window có thể học texture/nền;
nếu dùng quá sớm, làm nhòe local edge;
nếu window quá lớn, dễ overfit;
nếu không có positional/structural bias, attention không biết cấu trúc graph.

Vì vậy 7B nên đặt sau 1-2 lớp EdgeAware local, không đặt ngay đầu. Đầu tiên cần edge-gated GNN giữ biên/nếp; sau đó window attention mới gom vùng rộng hơn.

5. Graph Positional / Structural Encoding có nên thêm không?

Có. Đây là mảnh ghép rất nên thêm nếu dùng attention/Transformer trên graph.

Graphormer cho thấy Transformer trên graph cần structural encodings như centrality encoding, spatial encoding, edge encoding để mô hình hiểu cấu trúc graph; paper kết luận các encoding cấu trúc đơn giản nhưng hiệu quả giúp Transformer làm việc tốt trên graph.

Với pixel graph FER, ta không cần bê nguyên Graphormer. Ta dùng phiên bản phù hợp 48×48:

x_norm, y_norm
distance_to_center
border_distance
radial_distance
angle_from_center
scale_id / edge_type
relative_position_bias trong window
shortest-path distance hoặc Chebyshev distance trong window

Cái này rất hợp với FER-2013 vì mặt thường tương đối centered, nhưng crop vẫn lệch nên prior phải mềm. Không nên hard-code “mắt luôn ở đúng ô này”, mà dùng encoding để model biết vị trí tương đối.

Nói ngắn:

Multi-scale edges cho topology rộng hơn.
Window attention cho quan hệ pixel-to-pixel linh hoạt.
Structural encoding giúp attention hiểu không gian mặt.

Ba phần này bổ sung nhau.

6. Kiến trúc D12 nên như thế nào?

Mình đề xuất D12 có thay encoder, còn các phần sau giữ D11 Exp H.

D12 Encoder v1: đáng làm nhất
Input x, edge_index_multi, edge_attr_multi
        ↓
Input Projection
        ↓
Local Edge-Aware Block
        ↓
Context-Aware Multi-Scale Edge Block
        ↓
Window Pixel-to-Pixel Attention Block
        ↓
Shifted Window Pixel-to-Pixel Attention Block
        ↓
Output h_pixel
        ↓
D11 Head:
Slot Attention FACS
VirtualNodeGather
FiLM
Motif Relation
Classifier
Loss Scheduling Exp H

Cụ thể:

Block 1: Local Edge-Aware hiện tại. Giữ để bảo toàn inductive bias tốt của D10/D11. EdgeAwarePixelMessageLayer hiện tại dùng message từ source node nhân với gate sinh từ edge_attr, rồi mean aggregate và residual update.

Block 2: Context-Aware Multi-Scale Edge Block. Nâng cấp vấn đề 1 và 2 cùng lúc:

gate_ij = sigmoid(MLP([edge_attr_ij, h_i, h_j, h_i - h_j]))

Thay vì gate chỉ nhìn edge_attr, gate nhìn cả trạng thái node. Điều này giúp phân biệt “biên môi” với “biên tóc/nền”.

Block 3-4: Window + shifted-window pixel attention. Mượn ý tưởng Swin nhưng giữ pixel graph. Dùng relative position bias. Window size nên bắt đầu nhỏ:

window_size = 6 hoặc 8
shift_size = window_size / 2

48 chia hết cho 6 và 8. Nếu muốn ít token/window hơn: 6×6 = 36 pixel/window, 64 windows. Nếu muốn vùng rộng hơn: 8×8 = 64 pixel/window, 36 windows. Với FER 48×48, mình ưu tiên 6×6 trước vì an toàn hơn, ít overfit hơn, giữ local hơn.

7. D12 có nên dùng 7A hay 7B làm chính?

Nếu chỉ được chọn một hướng đầu tiên, mình chọn:

D12A = 7A + 7D trước

Tức là:

Multi-scale edge-aware encoder
+ structural/positional encoding

Chưa bật window attention.

Lý do: đây là nâng cấp gần encoder hiện tại nhất, ít phá hệ thống nhất, kiểm chứng rõ ràng nhất. Nếu D12A đã tăng F1_Raw_Local, ta biết encoder mới thật sự giúp motif. Sau đó mới thêm 7B:

D12B = D12A + Window Pixel Attention

Nếu nhảy thẳng D12B, kết quả tăng/giảm sẽ khó biết do multi-scale, structural encoding hay window attention.

Nhưng về mặt kiến trúc cuối cùng đáng giá nhất, mình chọn:

D12B = Multi-scale Graph Edges
     + Context-aware Edge Gate
     + Graph Structural Encoding
     + Window/Shifted-window Pixel Attention
     + D11 Exp H Head

Đây là bản tổng hợp tối ưu hơn.

8. Có cần thay toàn bộ phần sau encoder không?

Không. Không nên thay toàn bộ.

Giữ:

IterativeSlotAttentionFACS
VirtualNodeGather
FiLMFusion
MotifRelationTransformer
Aux classifier
SupCon projection
Loss Scheduling Exp H

Lý do là các phần này đã có bằng chứng thực nghiệm trong pipeline của bạn:

No Global giảm → giữ Global.
Global 128 sập → giữ bottleneck 64/32.
16 Slots giảm → giữ 8 slots.
High SupCon/High Div giảm → giữ ratio Exp C/H.
Soft CE warmup thắng → giữ scheduling.

D12 chỉ thay:

self.encoder = SharedPixelEncoder(...)

bằng:

self.encoder = D12MultiScalePixelEncoder(...)

Đây là cách ít rủi ro nhất và dễ viết báo cáo nhất:

D11 giải quyết global-local readout.
D12 giải quyết feature bottleneck bằng encoder pixel-to-pixel đa tỉ lệ.
9. D11 hiện tại 0.54 accuracy có đáng tiếp tục không?

Có thể chạy thêm 1-2 cấu hình nhẹ trên Exp H, nhưng không nên dành nhiều công sức nghiên cứu nữa.

Những thứ còn đáng làm ở D11:

Exp H long run 100 epochs với early stopping;
seed robustness 2-3 seeds;
ablation Zero-Global/Zero-Local/Gate-only cho bản H;
local raw metric và confusion delta.

Không nên tiếp tục:

tăng SupCon lớn;
tăng Diversity lớn;
tăng Local Aux lớn;
tăng Global Dim;
tăng slots;
thêm loss mới phức tạp.

Vì bạn đã thử và thấy giảm. D11 nên được đóng vai trò:

best stable baseline

còn D12 là hướng giải quyết gốc.

10. D12 có thật sự đáng tốn thời gian không?

Theo mình: đáng, vì nó tấn công đúng giới hạn còn lại.

D11 Exp H đã chứng minh loss/fusion có tác dụng. Nhưng nếu input node embedding vẫn nghèo và local, thì Slot Attention không có đủ vật liệu để tạo motif giàu. Đây là “feature bottleneck” mà tài liệu D11 cũng cảnh báo: D11 chủ yếu giải quyết global context, không giải quyết triệt để node/edge feature nghèo.

D12 đáng vì nó không chỉ “thêm module”, mà giải quyết 3 vấn đề lõi:

1. Receptive field local quá hẹp.
2. Edge gate chưa context-aware.
3. Pixel embedding chưa giàu semantic part.

Và vẫn giữ thuần GNN/pixel-to-pixel, chưa cần CNN.

11. Chốt hướng D12

Mình chốt hướng như sau:

D12A: Multi-scale Context-Aware Edge-Gated Encoder

Mục tiêu: thay SharedPixelEncoder, giữ toàn bộ D11 Exp H head.

Thành phần:

edge scales: 1, 2, maybe 4
edge_type / scale embedding
context-aware gate dùng [edge_attr, h_src, h_dst, h_src - h_dst]
gate-normalized aggregation
graph positional/structural encoding

Nếu D12A tăng F1_Raw_Local và macro F1, tiếp tục:

D12B: D12A + Window/Shifted-window Pixel Attention

Mục tiêu: pixel-to-pixel attention trong window, lấy ý tưởng Swin nhưng không dùng image backbone.

Cấu hình đầu tiên nên bảo thủ:

8 slots giữ nguyên
global_dim 64 giữ nguyên
lambda_supcon 0.1
lambda_div 0.02
diversity_margin 0.3
CE warmup 30 epochs như Exp H
window_size 6
shift_size 3
edge scales 1,2
chưa dùng scale 4 ở bản đầu

Scale 4 nên để ablation sau, vì dễ nối quá xa và học nền.

12. Tiêu chí quyết định D12 thắng thật

D12 không chỉ cần test accuracy tăng. Nó phải cho thấy:

F1_Raw_Local tăng so với D11-H
Full F1 tăng
Fear/Disgust/Sad tăng hoặc không giảm mạnh Angry
slot collapse giảm
Zero-Global không sập
Zero-Local vẫn sập
attention/slot không nhìn nền nhiều hơn
over-smoothing cosine không tăng quá cao

Nếu D12 chỉ tăng Fusion F1 nhưng Raw Local không tăng, thì encoder chưa thật sự giải quyết feature bottleneck; có thể chỉ tạo shortcut mới.

Kết luận cuối cùng: D11 nên chốt ở Exp H làm baseline chính. Qua D12 là đáng. D12 không thay toàn bộ, chỉ thay encoder. Hướng đáng giá nhất là multi-scale context-aware edge-gated encoder trước, rồi thêm windowed pixel-to-pixel attention có structural encoding nếu D12A có tín hiệu tốt.
Đúng, bạn bắt rất chuẩn. Nếu lấy D10 tốt nhất làm mốc, thì dự báo “D12 thực tế 57–59% accuracy” của mình ở lượt trước là quá bảo thủ và chưa quy chiếu đúng vào toàn bộ lịch sử D10.

Sau khi nhìn lại tài liệu D10 bạn vừa gửi, cần tách rõ 3 mốc:

D11 hiện tại: Exp H khoảng 54.28% accuracy, macro F1 0.5078.
D10 Phase 3 mạnh nhất: đã đạt 58.21–58.76% accuracy, macro F1 khoảng 0.559–0.563.
D10 Phase 5 Run 1 Two-stage SupCon: đạt Test F1 0.6130, nhưng có overfit nặng, Train F1 khoảng 0.87 trong khi Val/Test khoảng 0.60–0.613.

Vậy câu trả lời đúng hơn là:

D12 không nên được kỳ vọng chỉ vượt D11.
D12 phải được thiết kế để vượt D10 Phase 3,
và mục tiêu hợp lý là tiệm cận hoặc vượt D10 Phase 5 Run 1.
1. Vì sao D11 “nâng cấp hơn” nhưng lại thua D10?

Vì D11 không phải là “D10 tốt nhất + Global”. Thực tế D11 hiện tại giống một nhánh nghiên cứu mới, đã chứng minh Global + FiLM có ích, nhưng nó chưa kế thừa đầy đủ các thành phần mạnh nhất của D10.

D10 mạnh không chỉ vì Slot Attention. D10 Phase 3 mạnh nhờ một tổ hợp đã được thực nghiệm chứng minh:

Multi-scale GNN
+ Cross-Attention Slot Refinement
+ Residual Slot
+ Iteration = 5
+ hidden dim hợp lý
+ scheduler tốt
+ đôi khi no positional encoding lại tốt hơn

Tài liệu D10 ghi rõ P3 dùng bộ khung Multi-scale GNN kết hợp Cross-Attention Slot Refinement và Residual Slot; trong đó d10_p3_5_iter5 đạt F1 0.5626, Acc 58.21%, còn d10_p3_10_no_pos_enc đạt F1 0.5593, Acc 58.76%.

Trong khi D11 hiện tại tập trung vào:

FACS slots
VirtualNodeGather
FiLM
Local auxiliary loss
SupCon + diversity
CE warmup

Tức là D11 đã giải quyết tốt hơn bài toán global-local fusion, nhưng có thể đã yếu hơn D10 ở phần motif formation pipeline. Nếu encoder/slot-refinement của D11 chưa bằng D10 P3, thì thêm Global cũng không đủ kéo lên.

Nói ngắn:

D11 hơn D10 ở ý tưởng Global-Local.
Nhưng D10 P3/Phase5 hơn D11 ở nền motif encoder/training đã chín hơn.

Vì vậy “nâng cấp hơn” chỉ đúng nếu D12 là:

D12 = D10 best motif engine + D11 global-local fusion + D11/H loss schedule

Không phải:

D12 = D11 hiện tại + thêm vài cạnh
2. Mốc D10 0.6130 có phải mốc cần vượt không?

Có, nhưng phải hiểu đúng.

D10 Phase 5 Run 1 đạt Test F1 0.6130, đây là mốc GNN tốt nhất hiện tại. Nhưng tài liệu cũng ghi rõ nó overfit nặng: Train F1 khoảng 0.87, Val/Test khoảng 0.60–0.613, và Stage 2 với freeze_encoder=false thực chất fine-tune lại GNN/Slot khá nhiều.

Vậy mốc này là:

Một bằng chứng rằng pipeline GNN của bạn có thể lên 0.61 F1.
Nhưng chưa phải một mốc ổn định/hoàn hảo về generalization.

D12 phải nhắm tới hai mục tiêu:

Vượt D10 Phase 3 về accuracy/F1.
Tiệm cận hoặc vượt D10 Phase 5 về F1 nhưng ít overfit hơn.

Nếu D12 chỉ đạt 57–58% accuracy thì không đủ thuyết phục, vì D10 P3 đã có 58.76% accuracy. Nếu D12 là bản “thay đổi đáng giá”, nó cần nhắm:

Accuracy ≥ 59%
Macro F1 ≥ 0.57

làm mốc tối thiểu.

3. Dự báo thực tế nên chỉnh lại như sau

Dựa trên D10 history, dự báo hợp lý hơn là:

Mô hình	Accuracy thực tế	Macro F1 thực tế
D11 Exp H hiện tại	~54.3%	~0.508
D10 P3 best accuracy	~58.8%	~0.559
D10 Phase5 Run1	chưa rõ acc trong tài liệu, nhưng F1 0.613	0.613
D12A nếu chỉ nâng encoder nhẹ	59–60%	0.57–0.59
D12B nếu tích hợp đúng D10 best + D11 fusion	60–62%	0.60–0.63
D12C nếu thêm two-stage/Pseudo-2-stage ổn định	61–64%	0.62–0.65

Vậy mốc mình chỉnh lại:

D12 thực tế nên nhắm 60–62% accuracy.
Best-case hợp lý: 63–64% accuracy nếu D12 kế thừa đúng D10 P3/Phase5.
Macro F1 thực tế nên nhắm 0.60–0.63, best-case 0.64–0.65.

Còn 0.70+ vẫn là rất khó nếu vẫn thuần pixel-GNN không pretrained CNN/ViT, nhưng 0.62–0.65 là mốc đáng nhắm hơn nhiều so với 57–59.

4. Vì sao D12 có khả năng vượt D10?

D12 có lý do để vượt D10 nếu nó không bỏ mất các bài học D10.

D10 đã chứng minh:

Multi-scale GNN có ích, P2.1 đạt accuracy 55.87% và được kết luận nên giữ.
Cross-Attention Slot Refinement + Residual Slot + Iter=5 giúp P3 tăng lên 58% accuracy.
Mean-pooled SupCon Two-stage là hướng đúng; Per-slot SupCon phá encoder và đã bị loại.

D11 lại chứng minh:

Global không vô dụng, vì No Global giảm xuống.
Global quá lớn thì overfit, nên bottleneck là cần.
Soft CE warmup giảm xung đột gradient.
SupCon + diversity margin có ích nhưng phải vừa phải.

Vậy D12 tốt nhất phải là hợp nhất hai hệ kinh nghiệm:

D10: motif engine mạnh.
D11: global-local modulation và loss scheduling tốt.
D12: encoder local-global/multi-scale mới để làm motif giàu hơn từ gốc.

Nếu làm đúng, D12 không chỉ hơn D11 mà còn có cửa vượt D10.

5. D12 không nên thay toàn bộ, mà phải “backport” D10 best

Đây là điểm quan trọng nhất.

D12 nên giữ từ D10:

K = 8 slots
Iterative Slot Attention = 5
Residual slot connection
Cross-Attention Slot Refinement
Motif Relation Transformer
Class-Motif Attention Head
hidden_dim 64/96, tránh 128
CosineWarmup
Mean-pooled SupCon, không per-slot

D12 nên giữ từ D11:

VirtualNodeGather bottleneck 32/64
FiLM modulation, không concat
Aux local loss vừa phải
SupCon + diversity margin
CE warmup / pseudo two-stage kiểu Exp H
Zero-global/zero-local/gate-only test

D12 thay chính ở:

SharedPixelEncoder

thành:

Multi-scale Context-Aware Edge-Gated Pixel Encoder

Tức là D12 không phải “D11 từ đầu”, mà là:

D12 = D10-P3/P5 backbone logic
    + D11 controlled global context
    + new multi-scale/context-aware encoder
6. Vì sao D12 vẫn có thể không vượt D10?

Có 3 lý do.

Thứ nhất, D10 Phase5 đã rất mạnh nhờ training regime. Nếu D12 chỉ đổi encoder nhưng training không chuẩn, nó vẫn có thể thua.

Thứ hai, D10 P3 có một phát hiện ngược trực giác: no positional encoding tăng accuracy lên 58.76%. Nếu D12 thêm structural encoding quá mạnh, nó có thể lặp lại lỗi “tọa độ tuyệt đối gây overfit”. Tài liệu D10 ghi rõ no positional encoding là cú sốc lớn vì GNN tự học vị trí tương đối tốt hơn, còn nhét tọa độ tuyệt đối có thể làm overfit.

Thứ ba, FER2013 có các class khó như Fear, Sad, Disgust; tài liệu dự án ghi rõ ảnh nhỏ/nhiễu, Disgust ít mẫu, Fear/Sad mơ hồ và class tương đồng vùng mắt/miệng. Encoder mạnh hơn có thể cải thiện, nhưng không xoá hết noise/ambiguity.

Vì vậy, D12 có cơ hội vượt D10, nhưng không được kỳ vọng “đương nhiên tốt hơn”. Nó phải kế thừa đúng các bài học đã thắng.

7. Kết luận chốt lại

Bạn nói đúng: nếu D10 đã có 0.61 F1, thì D12 không thể đặt kỳ vọng kiểu 57–59% accuracy như một mục tiêu cuối. Mục tiêu đó chỉ hợp nếu so với D11 hiện tại.

Mục tiêu mới nên là:

D12A tối thiểu: vượt D10 P3
Accuracy ≥ 59%
Macro F1 ≥ 0.57

D12B tốt: tiệm cận D10 Phase5 nhưng ổn định hơn
Accuracy 60–62%
Macro F1 0.60–0.63

D12C rất tốt:
Accuracy 63–64%
Macro F1 0.64–0.65

Còn mốc 0.70+ vẫn nên xem là stretch goal, không phải kỳ vọng chính của single-model thuần pixel-GNN.

Chiến lược đúng nhất bây giờ là:

Không tiếp tục vắt D11 đơn thuần.
Không bỏ D10.
Thiết kế D12 bằng cách hợp nhất D10 best + D11 best.

Cụ thể:

D12 = D10 P3/Phase5 motif pipeline
    + D11 Global FiLM
    + D11 Exp H loss schedule
    + encoder multi-scale/context-aware mới

Nếu D12 không ít nhất vượt D10 P3, nó chưa đáng. Nếu D12 vượt D10 Phase5 Run1 hoặc đạt ngang 0.613 F1 nhưng overfit ít hơn, đó mới là thành công thật sự.

Đúng. Giờ hướng đi tiếp theo không nên là “tiếp tục thêm loss vào D11”, mà phải thiết kế lại thành một mainline mới có kế thừa có chọn lọc.

Mình chốt hướng như sau:

D12 = D10-P3/P5 motif engine mạnh nhất
    + D11 Global-Local FiLM đã chứng minh có ích
    + D11 Exp H loss scheduling
    + encoder mới giải quyết feature bottleneck / local receptive field

Tức là không bỏ D10, không bỏ D11, mà hợp nhất phần thắng của cả hai.

1. Chẩn đoán hiện tại: vì sao cần D12?

Ta có 3 sự thật thực nghiệm.

Thứ nhất, D10 đã mạnh hơn D11 hiện tại. D10 Phase 3 đã đạt khoảng F1 0.5626, Acc 58.21%, còn bản no position encoding đạt F1 0.5593, Acc 58.76%. Điều này cho thấy D10 motif engine hiện tại trưởng thành hơn D11 1-stage thuần.

Thứ hai, D10 Phase 5 Run 1 Mean-pooled SupCon two-stage đã đạt F1 0.6130, nhưng bị overfit nặng. Đây là bằng chứng quan trọng: pipeline GNN/motif của bạn có khả năng chạm vùng 0.61, nhưng training chưa ổn định/generalization chưa sạch.

Thứ ba, D11 đã chứng minh Global có ích, nhưng D11 chưa kế thừa đầy đủ “motif engine” mạnh nhất của D10. D11 tốt ở triết lý Global-Local, FiLM, CE warmup, SupCon + diversity mềm, nhưng phần hình thành motif chưa bằng D10 P3/P5.

Vậy lỗi không phải “D11 sai”. Lỗi là:

D11 hiện tại = Global-Local tốt
nhưng motif engine chưa đủ mạnh như D10 best.

Do đó hướng đúng là:

D12 phải lấy D10 làm xương sống motif,
lấy D11 làm cơ chế global-context,
rồi nâng encoder để giải quyết feature bottleneck.
2. Không nên làm gì tiếp?

Mình đề nghị dừng các hướng sau:

1. Không tiếp tục tăng lambda_supcon mạnh.
2. Không tăng lambda_div mạnh.
3. Không tăng lambda_local lên quá cao.
4. Không tăng global_dim lên 128.
5. Không tăng số slot lên 16.
6. Không thêm edge dropout / node noise.
7. Không dùng per-slot SupCon.
8. Không thử hidden_dim 128 nữa nếu chưa có lý do cực mạnh.

Vì D10/D11 đều đã chứng minh các hướng này dễ gây collapse hoặc giảm điểm. Tài liệu D10 cũng ghi rất rõ: 8 motifs là sweet spot, hidden dim 128 là “tử huyệt”, edge dropout/node noise gây underfitting, per-slot SupCon phá encoder, còn mean-pooled SupCon mới là hướng đúng.

3. Hướng chính: D12 theo 3 tầng

Mình thiết kế D12 theo 3 tầng tiến hóa. Không nhảy thẳng vào bản phức tạp nhất.

D12A — Encoder mới nhưng chưa dùng window attention

Mục tiêu của D12A là kiểm tra:

Nếu chỉ nâng encoder local graph,
Local Motif có mạnh hơn D10/D11 không?

D12A thay SharedPixelEncoder hiện tại bằng:

Multi-Scale Context-Aware Edge-Gated Pixel Encoder

Nhưng giữ toàn bộ head phía sau.

Luồng:

Input pixel graph
    ↓
D12A Encoder
    ↓
Iterative Slot Attention K=8, Iter=5
    ↓
Residual Slot + Cross-Attention Slot Refinement
    ↓
Motif Relation Transformer
    ↓
D11 Global Virtual Gather
    ↓
FiLM Local-Global Modulation
    ↓
Class-Motif Attention Head
    ↓
Emotion logits

Điểm quan trọng: D12A chỉ thay encoder, không đụng quá nhiều head/loss.

D12B — Thêm Window Pixel-to-Pixel Attention

Nếu D12A có tín hiệu tốt, D12B thêm block giống ý tưởng Swin nhưng vẫn thuần pixel graph:

Pixel embeddings
→ chia window 6×6 hoặc 8×8
→ pixel-to-pixel attention trong window
→ shifted window
→ trả về pixel embeddings

Đây không phải Swin CNN/ViT backbone. Nó là:

graph/pixel-token attention trực tiếp trên pixel nodes.

D12B giải quyết vấn đề:

Edge-aware GNN local quá hẹp.
Multi-scale edges vẫn là cạnh cố định.
Window attention cho pixel học quan hệ linh hoạt trong vùng rộng hơn.
D12C — Training regime ổn định kiểu D10 Phase 5 + D11 Exp H

Nếu D12A/B tốt, D12C mới tập trung vào training:

Mean-pooled SupCon
+ CE warmup mềm
+ diversity margin nhẹ
+ local auxiliary vừa phải
+ Stage-like schedule

Mục tiêu của D12C là:

Tiệm cận hoặc vượt D10 Phase 5 Run 1,
nhưng giảm overfitting.
4. Kiến trúc D12 đề xuất chi tiết
4.1. Giữ từ D10

Những thứ đã chứng minh hiệu quả thì giữ:

num_slots = 8
slot_iterations = 5
hidden_dim = 96 hoặc 64
MotifRelationTransformer
ClassMotifAttentionHead
Residual Slot
Cross-Attention Slot Refinement
CosineWarmup
Mean-pooled SupCon

D10 đã chỉ ra Iter=5 là tốt nhất; Iter=3 dễ yếu, Iter=7 giảm; 8 slots là sweet spot; tăng 10/12/16 slots đều gây nhiễu.

4.2. Giữ từ D11

Những thứ đã chứng minh có lý thì giữ:

VirtualNodeGather / Global Context
FiLM modulation, không concat
global_dim = 32 hoặc 64, không 128
lambda_local = 0.3
lambda_supcon = 0.1
lambda_div = 0.02
diversity_margin = 0.3
CE warmup mềm kiểu Exp H

D11 Exp G No Global giảm, nên global có ích. Exp J global 128 sập, nên global phải bị bottleneck. Exp H CE warmup thắng, nên giữ scheduling.

4.3. Thay encoder

Encoder hiện tại:

Input projection
→ EdgeAwarePixelMessageLayer × L

D12A encoder:

Input Projection
    ↓
Local Edge-Aware Block
    ↓
Context-Aware Edge-Gated Block
    ↓
Multi-Scale Edge-Gated Block
    ↓
Gate-Normalized Aggregation
    ↓
Output h_pixel

Công thức nâng cấp:

g_ij = sigmoid(MLP([edge_attr_ij, h_i, h_j, h_i - h_j]))

m_ij = MLP_msg(h_i) ⊙ g_ij

agg_j = sum_i(m_ij) / sum_i(g_ij)

Khác encoder cũ ở 3 điểm:

1. Gate không chỉ nhìn edge_attr, mà nhìn cả h_src/h_dst.
2. Aggregation không chia theo degree cứng, mà chuẩn hóa theo gate.
3. Có multi-scale edges để mở rộng receptive field.
5. Multi-scale graph nên thiết kế thế nào?

Không nên thêm quá nhiều scale ngay.

Bản đầu:

scale = 1: 8-neighbor bình thường
scale = 2: neighbor cách 2 pixel

Chưa dùng scale 4 ở bản đầu, vì scale 4 dễ nối quá xa, trộn qua nền/tóc/crop.

Edge attributes nên mở rộng thành:

dx
dy
dist
delta_intensity
intensity_similarity
scale_id

Hoặc tốt hơn:

edge_attr gốc giữ nguyên
+ edge_type_embedding(scale_id)

Cần phân biệt cạnh gần và cạnh xa. Nếu không, model sẽ xử lý cạnh cách 1 pixel và 2 pixel như nhau, dễ nhiễu.

6. Positional / structural encoding dùng thế nào để không lặp lỗi D10?

D10 P3 phát hiện rất quan trọng: tắt positional encoding lại tăng accuracy. Điều đó không có nghĩa là “không được dùng mọi vị trí”. Nó nghĩa là tọa độ tuyệt đối quá mạnh có thể gây overfit.

Vì vậy D12 không nên dùng absolute x/y quá thô kiểu ép model nhớ mặt ở tọa độ cố định.

Nên dùng structural encoding mềm:

relative dx/dy trên cạnh
scale_id
distance-to-center nhẹ
border-distance nhẹ
relative position bias trong window

Không nên dùng:

hard FACS coordinate quá mạnh
absolute position MLP quá lớn
spatial prior ép slot quá cứng

Với D12B window attention, structural encoding quan trọng nhất là:

relative position bias trong từng window

Nó giúp pixel attention hiểu quan hệ không gian mà không overfit quá nhiều vào tọa độ tuyệt đối.

7. D12A, D12B, D12C chạy theo thứ tự nào?
Giai đoạn 0 — Chốt baseline

Trước khi D12, cần xác định 2 baseline chính:

Baseline 1: D10 P3 best
- d10_p3_5_iter5
- d10_p3_10_no_pos_enc

Baseline 2: D11 Exp H
- D11 Global-Local + SupCon/Div + CE warmup

D12 phải so với cả hai.

Giai đoạn 1 — D12A Encoder-only

Chạy:

D12A-1:
D10 P3 motif engine
+ D12A encoder
+ chưa thêm D11 global

D12A-2:
D10 P3 motif engine
+ D12A encoder
+ D11 global FiLM

Mục tiêu là tách bạch:

Encoder mới có tự giúp Local không?
Global FiLM có tiếp tục giúp trên encoder mới không?

Nếu D12A-1 không hơn D10 P3, encoder chưa đáng.

Nếu D12A-2 hơn D12A-1, Global vẫn có ích.

Giai đoạn 2 — D12B Window Pixel Attention

Chỉ chạy sau khi D12A có tín hiệu.

Cấu hình đầu:

window_size = 6
shift_size = 3
num_heads = 4
hidden_dim = 96
dropout = 0.1 hoặc 0.2

Vì ảnh 48×48, window 6×6 tạo 64 window, mỗi window 36 pixel. Đây là mức an toàn hơn 8×8.

Sau đó thử:

window_size = 8
shift_size = 4

Window 8×8 rộng hơn nhưng rủi ro học nền/crop cao hơn.

Giai đoạn 3 — D12C Training schedule

Khi kiến trúc ổn, dùng schedule:

Epoch 1–30:
CE tăng từ 0 → 1 mềm
SupCon giữ 0.1
Diversity 0.02
Local aux 0.3

Epoch 31–80/100:
CE đầy đủ
SupCon có thể giữ 0.1 hoặc decay nhẹ
Diversity giữ nhẹ

Không dùng hard delay 40 epoch làm mặc định, vì D11 Exp L cho thấy hard delay mạnh nhưng không bằng soft warmup.

8. Config D12 khởi điểm đề xuất
D12A base
model:
  name: d12_global_local_motif
  encoder_name: d12_multiscale_context_edge_encoder

  num_classes: 7
  num_nodes: 2304
  node_dim: 7
  edge_dim: 5

  hidden_dim: 96
  num_slots: 8
  slot_iterations: 5

  use_residual_slot: true
  use_slot_refinement: true
  use_position_encoding: false

  use_global_branch: true
  global_dim: 64
  film_type: residual_tanh

  edge_scales: [1, 2]
  use_context_aware_gate: true
  use_gate_normalized_aggregation: true
  use_edge_type_embedding: true

loss:
  name: d11_global_local_loss
  lambda_local: 0.3
  lambda_supcon: 0.1
  lambda_div: 0.02
  diversity_margin: 0.3
  lambda_spatial: 0.5
  label_smoothing: 0.1
  ce_warmup_epochs: 30
  warmup_epochs: 5

Lưu ý: lambda_spatial nên giảm còn 0.5 hoặc giữ 1.0 tùy D11 FACS prior. Vì D10 no position encoding thắng, không nên để spatial prior quá mạnh.

D12B window attention
model:
  use_window_pixel_attention: true
  window_size: 6
  shift_size: 3
  window_heads: 4
  window_layers: 1
  relative_position_bias: true

Không dùng nhiều hơn 1–2 window layers lúc đầu.

9. Ablation tối thiểu phải chạy

Không cần chạy 20 bản ngay. Chạy 6 bản là đủ quyết định.

Run	Mục tiêu
D10-P3-best	baseline motif mạnh
D11-H	baseline global-local
D12A no-global	encoder mới có giúp local không
D12A full-global	encoder mới + global có cộng hưởng không
D12B window6	window pixel attention có giúp không
D12B window8	kiểm tra window rộng hơn

Nếu có thêm slot Kaggle, thêm:

D12A no position / soft structural only
D12A scale [1,2,4]
D12B no shifted window
D12B no global
10. Tiêu chí thắng / thua

D12 không được xem là thắng chỉ vì accuracy nhích nhẹ.

D12A được xem là có giá trị nếu:
Acc > 58.76%
Macro F1 > 0.5626
F1_Raw_Local tăng so với D11-H
Fear/Disgust/Sad không tụt
slot collapse giảm hoặc không xấu hơn
D12B được xem là có giá trị nếu:
Acc ≥ 60%
Macro F1 ≥ 0.59
F1_Raw_Local tăng
Window attention không nhìn nền quá nhiều
Zero-Global giảm vừa phải
Zero-Local giảm mạnh
D12C thành công lớn nếu:
Macro F1 ≥ 0.613
Acc ≥ 61–62%
Overfit gap nhỏ hơn D10 Phase5 Run1

Điểm quan trọng: Nếu D12 đạt F1 0.61 nhưng train F1 0.87 như D10 Phase5, thì nó chỉ ngang D10. Nếu đạt 0.61 với overfit thấp hơn, đó mới là nâng cấp thật.

11. Kỳ vọng kết quả mới

Sau khi hiệu chỉnh dựa trên D10, mình đặt kỳ vọng như sau:

D12A thực tế:
Acc 59–60%
Macro F1 0.57–0.59

D12B tốt:
Acc 60–62%
Macro F1 0.60–0.63

D12C rất tốt:
Acc 62–64%
Macro F1 0.63–0.65

Mốc 0.70 vẫn là stretch goal, chưa nên xem là kỳ vọng chính nếu vẫn thuần pixel-GNN.

12. Roadmap cuối cùng
Phase 1 — Đóng băng D11

Chốt D11 Exp H làm baseline. Không tiếp tục tuning loss/global/slots nữa.

Việc còn làm với D11:

evaluate zero-global
evaluate zero-local
log F1_Raw_Local
confusion matrix
slot visualization

Mục tiêu là dùng D11 làm baseline khoa học, không phải tiếp tục vắt điểm.

Phase 2 — D12A

Triển khai:

D12MultiScaleContextEdgeEncoder

Gồm:

context-aware edge gate
multi-scale edges [1,2]
edge type embedding
gate-normalized aggregation
soft structural encoding

Giữ:

D10 P3 motif engine
D11 global FiLM
D11 Exp H loss schedule

Đây là phase quan trọng nhất.

Phase 3 — D12B

Thêm:

Window Pixel-to-Pixel Attention
Shifted Window
Relative Position Bias

Chỉ thêm nếu D12A thắng D10 P3.

Phase 4 — D12C

Huấn luyện kiểu:

Mean-pooled SupCon
Soft CE warmup
moderate regularization
longer 100 epochs
multi-seed

Mục tiêu là vượt D10 Phase5 nhưng overfit ít hơn.

13. Câu chốt thiết kế

Hướng đi tiếp theo nên là:

D12 không phải là D11 plus.
D12 là hợp nhất D10 best và D11 best.

Công thức chính xác:

D12 =
    D10 P3/P5 motif pipeline
  + D11 Global-Local FiLM
  + D11 Exp H CE warmup
  + D12 Multi-scale Context-Aware Pixel Encoder

Đây là hướng có xác suất cao nhất để vượt mốc D10, vì nó không bỏ các bài học đã chứng minh bằng thực nghiệm, đồng thời tấn công đúng giới hạn còn lại: pixel encoder hiện tại vẫn quá local và feature còn nghèo.