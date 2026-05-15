# D12A 6-Run Experiment Plan

Ngay 2026-05-15. Muc tieu cua 6 run nay la giu nguyen D12A full
(`use_global_branch=true`, `use_scale2=true`) va runtime speed overlay da chot,
sau do chi thay doi objective/motif depth nho de cuu Angry/Disgust va tang
macro F1.

## Runtime Contract

Tat ca config moi deu inherit theo thu tu:

```yaml
inherits:
  - d12a_stable_ce_first.yaml
  - d12a_ddp_compile_fixedshape_runtime.yaml
```

Nhung diem co dinh:

| Item | Value |
|---|---|
| Global batch | 64 |
| DDP | enabled |
| `ddp.find_unused_parameters` | true |
| AMP | true |
| Compile | true |
| Compile order | before_ddp |
| Fixed batch | true |
| Epochs | 30 |
| Early stopping patience | 15 |

## Six Runs

| # | Config | Research question | Expected outcome |
|---:|---|---|---|
| 1 | `d12a_ce_balance_w05.yaml` | Class weight 0.5 co du de mo khoa Angry/Disgust khong? | Mainline candidate, should improve rare-class recall with lowest objective risk. |
| 2 | `d12a_ce_balance_w075.yaml` | Weight 0.75 co cuu rare classes hon hay lam mat Happy/Neutral? | Higher rare-class pressure; watch macro F1 vs acc tradeoff. |
| 3 | `d12a_logit_adjust_tau05.yaml` | Logit adjustment tau 0.5 co tao prior bias tot hon CE weights khong? | More direct bias to minority classes; watch over-prediction of rare classes. |
| 4 | `d12a_focal_gamma1_w05.yaml` | Focal gamma 1.0 co tap trung vao mau kho ma khong gay collapse moi khong? | May help hard examples; compare pred_count spread and rare-class F1. |
| 5 | `d12a_iter5_w05.yaml` | Slot iterations 5 co lay lai motif power tu D10-style refinement khong? | Better motif refinement; higher cost but same speed overlay path. |
| 6 | `d12a_supcon_light_w05.yaml` | SupCon nhe sau warmup CE co tang class separation khong? | Potential macro F1 lift; monitor stability and `effective_lambda_supcon`. |

## Kaggle Commands

Session 1:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_ce_balance_w05.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --use_compile \
  --compile_order before_ddp \
  --no_wandb
```

Session 2:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_ce_balance_w075.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --use_compile \
  --compile_order before_ddp \
  --no_wandb
```

Session 3:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_logit_adjust_tau05.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --use_compile \
  --compile_order before_ddp \
  --no_wandb
```

Session 4:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_focal_gamma1_w05.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --use_compile \
  --compile_order before_ddp \
  --no_wandb
```

Session 5:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_iter5_w05.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --use_compile \
  --compile_order before_ddp \
  --no_wandb
```

Session 6:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_supcon_light_w05.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --use_compile \
  --compile_order before_ddp \
  --no_wandb
```

## Reading Results

Ket luan nen dua tren `val_macro_f1`, `val_acc`, local metrics, `pred_count`,
per-class report neu co, va loss diagnostics:

- `loss_ce`, `loss_local`, `loss_supcon`, `loss_div`
- `effective_ce_weight`, `effective_lambda_supcon`
- `logit_adjust_tau`, `focal_gamma` neu run dung
- `h_pixel_mean`, `h_pixel_std`, `encoder_gate_mean`, `encoder_gate_std`
- `slot_area_entropy`, `logits_mean`, `logits_std`

Khong ket luan chi tu accuracy neu Angry/Disgust van zero-predict.
