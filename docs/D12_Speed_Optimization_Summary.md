# D12 Speed Optimization Summary

## 1. Kết luận ngắn

Đường tốc độ nên dùng cho D12 từ bây giờ là:

```text
DDP 2 GPU + global batch 64 + AMP + chunk-aware sampler
+ fixed-shape train batches + torch.compile before DDP
```

Config runtime dùng lại:

```text
configs/experiments/d12a_ddp_compile_fixedshape_runtime.yaml
```

Kết quả tốt nhất đã đo được:

| Thiết lập | Ước lượng epoch ổn định |
|---|---:|
| DP b64 AMP compile off baseline | `~5.59 min/epoch` |
| DDP b64 chunk-aware eager | `~5.36 min/epoch` |
| DDP b64 compile after DDP | `~2.51 min/epoch` |
| DDP b64 compile before DDP, batch shape còn động | `~2.30 min/epoch` nhưng có spike muộn |
| **DDP b64 compile before DDP + fixed-shape** | **`~2.23 min/epoch` ổn định** |

So với baseline DP cũ, đường cuối cùng giảm thời gian epoch khoảng:

```text
5.59 -> 2.23 phút/epoch
giảm khoảng 60%
```

Với run 100 epoch, riêng phần train ước lượng giảm từ khoảng:

```text
~9.3 giờ -> ~3.7 giờ
```

## 2. Chuỗi tối ưu đã đi qua

### 2.1. Baseline ban đầu

Baseline production ban đầu:

```text
DataParallel
global batch 64
AMP true
torch.compile false
num_workers 2
chunk_cache_size 4
graph_cache_chunks 4
```

Log tham chiếu:

```text
outputs_log/d12_fastio_benchmark_safe_num2_batchsize64_amp.log
```

Kết quả:

| Metric | Giá trị |
|---|---:|
| `estimated_full_epoch_minutes` | `~5.59` |
| `avg_data_time` | `~0.0128s` |
| Bottleneck chính | forward/backward |

Điểm quan trọng: DataLoader của đường DP đã đủ nhanh; vấn đề còn lại chủ yếu nằm ở compute.

### 2.2. DDP Phase 1 thất bại vì mất locality

DDP prototype đầu tiên dùng `DistributedSampler` và tắt chunk-aware batching.

Kết quả:

| Metric | Giá trị |
|---|---:|
| `avg_data_time` | `~2.0-2.3s` |
| `estimated_full_epoch_minutes` | `~7-9` |

Nguyên nhân:

- `DistributedSampler` shuffle toàn dataset.
- Các batch nhảy qua nhiều graph chunks.
- Cache chunk gần như mất tác dụng.
- Pattern batch nhẹ xen kẽ batch rất nặng xuất hiện rõ trong log.

Bài học:

```text
Với graph repo dạng chunk, DDP đúng về orchestration vẫn có thể chậm nếu sampler phá data locality.
```

### 2.3. DDP Phase 1.5 phục hồi locality

Đã thêm:

```text
data/ddp_chunk_aware_sampler.py
```

Ý tưởng:

- Mỗi rank nhận chunk riêng theo round-robin.
- Batch được tạo trong chunk để giữ locality.
- Hai rank được cân bằng số batch bằng truncate về `min_num_batches`.

Kết quả DDP eager b64:

| Metric | Giá trị |
|---|---:|
| `estimated_full_epoch_minutes`, trung bình epoch 2-14 | `5.363` |
| So với DP baseline | nhanh hơn khoảng `3.9%` |

Kết luận lúc đó:

```text
DDP b64 eager chạy đúng nhưng chưa đủ lợi để đáng giữ.
```

### 2.4. Các nhánh bị loại

#### `find_unused_parameters=False`

Config fast mode bị lỗi ngay:

```text
Expected to have finished reduction in the prior iteration
Parameter indices which did not receive grad: 182..187
```

Nguyên nhân thực tế:

- CE-first hiện dùng `lambda_supcon=0.0`.
- `supcon_proj` không tham gia loss nên không nhận gradient.

Kết luận:

```text
Hiện tại phải giữ ddp.find_unused_parameters: true.
```

#### DDP global batch 96

Kết quả benchmark Kaggle:

```text
global batch 96
per-rank batch 48
OOM trên cả hai T4
```

Kết luận:

```text
Không dùng b96 cho D12A hiện tại.
```

### 2.5. Compile trở thành bước ngoặt

Đã thử hai thứ tự:

```text
after_ddp:
  model = DDP(model)
  model = torch.compile(model)

before_ddp:
  model = torch.compile(model)
  model = DDP(model)
```

Kết quả:

| Mode | Trung bình epoch 2-14 | Epoch 15 |
|---|---:|---:|
| `compile_after_ddp` | `2.514 min` | `8.52 min` |
| `compile_before_ddp` | `2.299 min` | `10.18 min` |

Kết luận:

- `compile_before_ddp` nhanh hơn rõ ràng ở steady state.
- Đây cũng là thứ tự phù hợp hơn với hướng dẫn của PyTorch cho DDP.
- Tuy nhiên hai bản đầu đều có compile spike muộn.

Các warning đã thấy:

| Warning | Ý nghĩa |
|---|---|
| `Graph break from Tensor.item()` | D12 có nhánh kiểm tra runtime bằng `int(src.max())`, khiến Dynamo tách graph. Chưa sửa ở bước tối ưu tốc độ này. |
| `Profiler function record_function will be ignored` | Xuất hiện ở `compile_after_ddp`; không phải lỗi correctness. |
| `Not enough SMs to use max_autotune_gemm mode` | T4 không đủ SM cho mode autotune đó; warning vô hại cho correctness. |

### 2.6. DDP Phase 1.6: fixed-shape sampler

Nguyên nhân spike muộn được khoanh lại là batch shape động từ leftover batch cuối chunk.

Đã sửa sampler để hỗ trợ:

```text
fixed_batch_size: true
drop_incomplete_batches: true
carry_over_leftovers: true
```

Logic mới:

- Gom leftover qua ranh giới chunk trong cùng rank.
- Chỉ yield batch đầy đủ `per_rank_batch_size`.
- Cuối epoch drop phần buffer còn thiếu.
- Sau đó vẫn cân bằng số batch giữa các rank để tránh DDP hang.

Log xác nhận:

```text
unique_batch_sizes_per_rank=[[32], [32]]
```

Không có:

```text
[DDP FixedShape Violation]
```

Kết quả cuối:

| Metric | Giá trị |
|---|---:|
| Epoch 2-14 average | `2.232 min` |
| Epoch 2-15 average | `2.238 min` |
| Epoch 15 | `2.32 min` |
| `avg_data_time` ổn định | khoảng `0.019s` |
| `avg_forward_time` ổn định | khoảng `0.116s` |
| `avg_backward_time` ổn định | khoảng `0.164s` |
| `avg_batch_time` ổn định | khoảng `0.302s` |
| `cuda_rankmax` | khoảng `8.415 GB` |

Kết luận:

```text
Fixed-shape đã loại bỏ compile spike muộn và trở thành đường production hợp lý cho D12.
```

## 3. Những file đã thêm hoặc thay đổi

### File mới

- `scripts/train_d5a_ddp.py`
- `data/ddp_chunk_aware_sampler.py`
- `configs/experiments/d12a_stable_ce_first_ddp_b64_amp.yaml`
- `configs/experiments/d12a_stable_ce_first_ddp_b64_amp_chunkaware.yaml`
- `configs/experiments/d12a_stable_ce_first_ddp_b64_amp_chunkaware_fast.yaml`
- `configs/experiments/d12a_stable_ce_first_ddp_b64_amp_chunkaware_compile_fixedshape.yaml`
- `configs/experiments/d12a_ddp_compile_fixedshape_runtime.yaml`

### File đã cập nhật

- `notebooks/kaggle-d12-fastio-benchmark.ipynb`
- `configs/experiments/d12a_stable_ce_first_no_global.yaml`
- `configs/experiments/d12a_stable_ce_first_no_scale2.yaml`

## 4. Cấu hình tốc độ cuối cùng

Runtime overlay hiện dùng:

```yaml
data:
  batch_size: 64
  num_workers: 2
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  chunk_cache_size: 4
  graph_cache_chunks: 4
  chunk_aware_shuffle: false
  shuffle_chunks: true
  shuffle_within_chunk: true
  ddp_chunk_aware: true
  fixed_batch_size: true
  drop_incomplete_batches: true
  carry_over_leftovers: true
  ddp_drop_last_batches: true

training:
  batch_size: 64
  num_workers: 2
  amp: true
  use_compile: true
  torch_compile: true
  compile_order: before_ddp

ddp:
  enabled: true
  backend: nccl
  compile: true
  compile_order: before_ddp
  find_unused_parameters: true
  rank0_full_validation: true
```

Các điểm không nên đổi tùy tiện:

- Không đổi `find_unused_parameters` sang `false` khi `lambda_supcon=0.0`.
- Không đổi sang global batch 96 ở D12A hiện tại vì đã OOM.
- Không tắt `fixed_batch_size` nếu vẫn dùng compile.
- Không thay `compile_order` sang `after_ddp` nếu mục tiêu là đường nhanh nhất hiện tại.

## 5. Cách dùng cho các config D12 mới

### 5.1. Pattern khuyến nghị

Với config D12 mới, chỉ cần inherit config base của thí nghiệm và runtime overlay tốc độ:

```yaml
inherits:
  - d12a_stable_ce_first.yaml
  - d12a_ddp_compile_fixedshape_runtime.yaml

run:
  config_name: d12a_my_new_variant

model:
  # Chỉ override phần kiến trúc hoặc ablation cần test.
```

Ví dụ hiện tại:

```yaml
inherits:
  - d12a_stable_ce_first.yaml
  - d12a_ddp_compile_fixedshape_runtime.yaml

run:
  config_name: d12a_stable_ce_first_no_global

model:
  use_global_branch: false
```

Ý nghĩa:

- `d12a_stable_ce_first.yaml` giữ nội dung thí nghiệm.
- `d12a_ddp_compile_fixedshape_runtime.yaml` giữ toàn bộ runtime speed path.
- Config con chỉ nên chứa phần khác biệt nghiên cứu thật sự.

### 5.2. Lệnh chạy production

Ví dụ chạy config baseline CE-first đã dùng runtime mới:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_stable_ce_first_ddp_b64_amp_chunkaware_compile_fixedshape.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --use_compile \
  --compile_order before_ddp \
  --no_wandb
```

Ví dụ chạy một ablation D12 đã thừa kế overlay:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_stable_ce_first_no_global.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --use_compile \
  --compile_order before_ddp \
  --no_wandb
```

### 5.3. Lệnh benchmark ngắn để kiểm tra một config mới

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_my_new_variant.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --global_batch_size 64 \
  --max_train_batches 30 \
  --max_val_batches 10 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --use_compile \
  --compile_order before_ddp \
  --no_wandb
```

Sau benchmark mới, nên kiểm tra tối thiểu:

- `unique_batch_sizes_per_rank=[[32], [32]]`
- Không có `[DDP FixedShape Violation]`
- `avg_data_time` vẫn thấp, thường quanh vài phần trăm giây.
- `estimated_full_epoch_minutes` không lệch xa khỏi vùng `~2.2-2.4`.
- Không có OOM.

## 6. Những điều rút ra được

1. Với repo này, sampler quan trọng gần ngang model:
   mất chunk locality là đủ để phá hoàn toàn lợi ích DDP.

2. DDP eager không đáng chỉ vì "dùng 2 GPU":
   ở b64 nó chỉ hơn DP khoảng 4%, quá ít để đổi đường production.

3. `torch.compile` có ích lớn với DDP ở D12:
   không giống benchmark DP trước đó, compile trong DDP giảm thời gian rất mạnh.

4. Thứ tự compile có ý nghĩa:
   `before_ddp` nhanh hơn `after_ddp` rõ ràng.

5. Batch shape động làm compile khó ổn định:
   chỉ khi ép fixed shape hoàn toàn thì compile mới hết spike muộn.

6. B96 không phải lối thoát cho model này trên 2xT4:
   local batch 48 đã OOM.

7. Từ giờ nên tách:
   - logic nghiên cứu trong config con
   - logic runtime tốc độ trong overlay dùng lại

## 7. Quyết định production hiện tại

Giữ đường:

```text
DDP b64 + AMP + chunk-aware + fixed-shape + compile_before_ddp
```

Dùng overlay:

```text
d12a_ddp_compile_fixedshape_runtime.yaml
```

Đây là cấu hình nên mặc định dùng cho các biến thể D12 tiếp theo, trừ khi một ablation mới thay đổi loss hoặc output theo cách làm `find_unused_parameters` không còn cần thiết, hoặc thay đổi footprint bộ nhớ đủ lớn để phải benchmark lại.
