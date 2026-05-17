# D12 Micro Diagnostics Plan

## Muc tieu

Chan doan lieu D12A co lam mat micro-expression truoc Slot Attention hay khong.

## Vi sao can diagnostics

Rare rescue khong cuu duoc Disgust. Local branch cung co the khong co Disgust signal. Can xac dinh loi nam o:

- runtime
- encoder scale2/context
- slot/motif
- global fusion
- classifier bias

## Diagnostics can doc

- scale1 vs scale2 std
- node cosine similarity delta
- slot entropy / effective slots
- class_part_similarity Angry-Disgust
- local vs full predictions
- slot visualization
- embedding export

## Quyet dinh sau diagnostics

Case A:
scale2 lam cosine similarity tang manh, slot diffuse, Disgust local = 0
=> trien khai D12A-Micro Encoder:

```text
h_fine = scale1 local encoder
h_context = scale2 context encoder
h_pixel = h_fine + alpha * gate * h_context
```

Case B:
local co Disgust, full mat Disgust
=> sua FiLM/global/classifier, khong sua encoder truoc.

Case C:
repeat/aux co binary signal nhung main khong dung
=> can auxiliary integration hoac classifier calibration.

Case D:
runtime AMP/batch gay collapse
=> chot runtime no AMP cho quality truoc khi sua model.

## Khong lam o buoc nay

- Khong D12B window attention.
- Khong node_dim=12.
- Khong SupCon manh.
- Khong rare-loss moi.
- Khong DDP speed runtime.
