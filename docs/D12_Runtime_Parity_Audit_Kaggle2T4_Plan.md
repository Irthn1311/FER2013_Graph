# D12 Runtime Parity Audit Plan - Kaggle 2xT4

## 1. Why This Audit Exists

All D12A experiments in question ran on Kaggle 2xT4. The problem is therefore not a hardware mismatch; it is a runtime/training-path mismatch.

Reference stable CE-first run:

- eval acc about 0.3876
- eval macro F1 about 0.2715
- eval weighted F1 about 0.3350
- pred_count about `[0, 0, 355, 1369, 525, 623, 717]`

Speed-control run under the production speed path:

- DDP 2 GPU
- global batch 64
- AMP true
- fixed-shape batches
- torch.compile before DDP
- chunk-aware sampler
- eval macro F1 about 0.0562
- pred_count `[0, 0, 0, 3589, 0, 0, 0]`

The audit isolates runtime factors while keeping the same D12A stable CE-first research logic.

No rare aux, no logit adjustment, no focal, no margin, no target repeat, no D12B, no node_dim=12, no batch96.

## 2. Five Runtime Parity Configs

| Order | Config | Runtime Path | Purpose |
|---:|---|---|---|
| 1 | `d12a_parity_legacy_dp_ce_first.yaml` | legacy DP, batch 32, no AMP, no compile | Closest path to old stable baseline |
| 2 | `d12a_parity_legacy_dp_b64_amp_ce_first.yaml` | legacy DP, batch 64, AMP, no compile | Test DP batch64 + AMP without DDP |
| 3 | `d12a_parity_ddp_eager_noamp_ce_first.yaml` | DDP, chunk-aware fixed-shape, no AMP, no compile | Isolate DDP sampler/fixed-shape |
| 4 | `d12a_parity_ddp_amp_no_compile_ce_first.yaml` | DDP, AMP, no compile | Add AMP to DDP |
| 5 | `d12a_parity_ddp_amp_compile_fixedshape_ce_first.yaml` | DDP, AMP, compile before DDP, fixed-shape | Re-run production speed path with diagnostics |

All configs keep:

```yaml
model:
  use_global_branch: true
  encoder:
    use_scale2: true
  slot_iterations: 3
  residual_slot_connection: false

loss:
  class_weight_power: 0.25
  label_smoothing: 0.05
  lambda_supcon: 0.0
  lambda_div: 0.0
  lambda_spatial: 0.0
  ce_warmup_epochs: 0

training:
  epochs: 30
  early_stopping_patience: 15
```

## 3. Hypotheses

| Question | Readout |
|---|---|
| Does Config 1 reproduce macro F1 near 0.27? | If yes, current model/code can still reproduce the stable path. |
| Does Config 2 collapse while Config 1 is healthy? | Suspect batch64/AMP/DataParallel interaction. |
| Does Config 3 collapse while Config 2 is healthy? | Suspect DDP sampler or fixed-shape distribution. |
| Does Config 4 collapse while Config 3 is healthy? | Suspect AMP in DDP. |
| Does Config 5 collapse while Config 4 is healthy? | Suspect compile/fixed-shape production path. |
| Do all configs collapse? | Stable baseline is no longer reproducible; audit code/config changes. |
| Do all configs behave well? | Runtime is not the main issue; rare-rescue failure points back to objective/representation. |

## 4. Commands

### Config 1: Legacy DP, Batch 32, No AMP, No Compile

```bash
python scripts/train_d5a.py \
  --config configs/experiments/d12a_parity_legacy_dp_ce_first.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --output_root /kaggle/working/outputs/d12_experiments/d12a_parity_legacy_dp_ce_first \
  --chunk_cache_size 8 \
  --no_wandb
```

### Config 2: Legacy DP, Batch 64, AMP, No Compile

```bash
python scripts/train_d5a.py \
  --config configs/experiments/d12a_parity_legacy_dp_b64_amp_ce_first.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --output_root /kaggle/working/outputs/d12_experiments/d12a_parity_legacy_dp_b64_amp_ce_first \
  --chunk_cache_size 4 \
  --no_wandb
```

### Config 3: DDP Eager, No AMP, No Compile

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_parity_ddp_eager_noamp_ce_first.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --no_wandb
```

### Config 4: DDP AMP, No Compile

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_parity_ddp_amp_no_compile_ce_first.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --no_wandb
```

### Config 5: DDP AMP Compile Fixed-Shape

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_parity_ddp_amp_compile_fixedshape_ce_first.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/graph_repo \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --use_compile \
  --compile_order before_ddp \
  --no_wandb
```

## 5. Runtime Diagnostics

Each run writes sampler/runtime diagnostics in the output folder:

- legacy DP: `sampler_diagnostics.json`
- DDP: `sampler_diagnostics_rank0.json`, `sampler_diagnostics_rank1.json`, plus rank-0 `sampler_diagnostics.json`

Fields include:

- runtime mode
- world size and rank
- global/per-rank batch size
- AMP/compile/DDP flags
- fixed-shape/drop/carry-over flags
- number of train batches
- unique batch sizes
- dataset length before sampler
- number of yielded samples
- per-rank label histogram
- global label histogram on rank 0 when distributed gather is available
- class 0 and class 1 exposure counts
- DDP chunk-aware sampler summary when available

## 6. How To Read Outcomes

### Case A

Config 1 is close to macro F1 0.27 and Config 5 collapses:

Runtime speed path is the main suspect.

### Case B

Config 1 also collapses:

Current code/config no longer reproduces the old stable baseline. Audit model/loss/config changes before more experiments.

### Case C

Config 1 is healthy, Config 2 collapses:

Batch64/AMP/DP interaction is suspect.

### Case D

Config 2 is healthy, Config 3 collapses:

DDP sampler/fixed-shape distribution is suspect.

### Case E

Config 3 is healthy, Config 4 collapses:

AMP in DDP is suspect.

### Case F

Config 4 is healthy, Config 5 collapses:

torch.compile or production fixed-shape compile path is suspect.

### Case G

All five configs are healthy:

Runtime is not the main cause. Rare-rescue failure should be treated as objective/representation failure, and the next step is D12A-Micro diagnostics.

## 7. Required Pre-Flight Checks

Run locally before Kaggle training:

```bash
python scripts/check_d12_runtime_parity_configs.py
python scripts/smoke_d12_model.py
```

These do not train. They only validate config resolution and D12 smoke behavior.

