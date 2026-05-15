# D12A Rare-Class Rescue 6-Run Plan

Ngay 2026-05-15. Vong nay giu D12A full (`use_global_branch=true`,
`use_scale2=true`) va runtime overlay da chot, chi test data exposure,
binary rare auxiliary supervision, va hard-negative margin cho Angry/Disgust.

## Runtime Contract

Tat ca config moi inherit:

```yaml
inherits:
  - d12a_stable_ce_first.yaml
  - d12a_ddp_compile_fixedshape_runtime.yaml
```

Bat buoc giu:

| Item | Value |
|---|---|
| Global batch | 64 |
| AMP | true |
| DDP | enabled |
| Fixed batch | true |
| Compile | true |
| Compile order | before_ddp |
| `ddp.find_unused_parameters` | true |
| Epochs | 30 |
| Early stopping patience | 15 |

## Six Runs

| # | Config | Hypothesis | Readout |
|---:|---|---|---|
| 1 | `d12a_speed_control_ce_first.yaml` | Speed-runtime control should reproduce CE-first behavior under the current DDP compile path. | Compare against previous D12A stable CE-first before judging new tricks. |
| 2 | `d12a_target_repeat_disgust.yaml` | Repeating class 1 samples inside chunk-local train pools increases Disgust exposure without changing graph repo. | Disgust pred_count and Disgust recall/F1 should become non-zero. |
| 3 | `d12a_rare_aux_bce.yaml` | Binary auxiliary heads on pooled motifs give Angry/Disgust a direct supervision path. | Check `loss_rare_aux`, Angry/Disgust pred_count, and rare-class F1. |
| 4 | `d12a_rare_aux_logit_tau05.yaml` | Rare aux plus logit adjustment combines the previous Angry signal with rare binary supervision. | Angry should stay non-zero; Disgust should improve beyond zero if useful. |
| 5 | `d12a_repeat_disgust_rare_aux.yaml` | Disgust repeat plus rare aux targets the most collapsed class from both data and loss sides. | Primary run for Disgust recovery; watch over-prediction or Happy collapse. |
| 6 | `d12a_rare_hardneg_margin.yaml` | Margin against common confusers pushes Angry/Disgust boundaries without changing sampling. | Check rare-class recall and whether Fear/Sad/Happy trade-offs are acceptable. |

## Kaggle Commands

Session 1:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_speed_control_ce_first.yaml \
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
  --config configs/experiments/d12a_target_repeat_disgust.yaml \
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
  --config configs/experiments/d12a_rare_aux_bce.yaml \
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
  --config configs/experiments/d12a_rare_aux_logit_tau05.yaml \
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
  --config configs/experiments/d12a_repeat_disgust_rare_aux.yaml \
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
  --config configs/experiments/d12a_rare_hardneg_margin.yaml \
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

Read these artifacts before concluding:

- `resolved_config.yaml`: confirm the intended config, repeat factors, rare aux, or margin was active.
- `training_history.json`: best epoch by `val_macro_f1`, `val_pred_count_*`, `loss_rare_aux`, `loss_rare_margin`, `lambda_rare_aux`, `lambda_rare_margin`.
- `evaluation/metrics.json`: test `accuracy`, `macro_f1`, `weighted_f1`, and `pred_count`.
- `evaluation/classification_report.json`: Angry/Disgust precision, recall, F1.
- `evaluation/predictions.csv`: verify pred_count and inspect whether rare classes are actually predicted.

Primary success criterion: Angry and Disgust are no longer both zero-predict while macro F1 does not collapse below the CE-first control.
