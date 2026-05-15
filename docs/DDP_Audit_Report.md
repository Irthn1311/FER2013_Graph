# DDP Audit Report for FER Graph D12A

## 1. Executive Summary

Verdict: DDP đáng làm dưới dạng prototype riêng, nhưng chưa nên nhúng thẳng vào `scripts/train_d5a.py`.

Repo hiện tại đã có baseline DataParallel chạy ổn trên Kaggle 2xT4 cho D12A với batch global 64, AMP bật, compile tắt. Benchmark log `outputs_log/d12_fastio_benchmark_safe_num2_batchsize64_amp.log` cho thấy DataLoader không còn là bottleneck chính: average batch khoảng 0.731s, forward khoảng 0.416s, backward khoảng 0.302s, estimated full epoch khoảng 5.59-5.60 phút/epoch ở các epoch cuối. Vì phần nặng nằm ở forward/backward, DDP có cơ hội giảm overhead của `torch.nn.DataParallel`, nhất là scatter/gather và single-process Python overhead.

Kết luận thực dụng:

- DDP global batch 64 có khả năng nhanh hơn DP khoảng 7-16%, best-case khoảng 18-23%, worst-case có thể ngang hoặc chậm hơn nếu sampler/eval/barrier/logging xử lý chưa gọn.
- DDP global batch 96 đáng test sau khi DDP b64 ổn vì local batch mỗi GPU là 48. Có khả năng fit hơn DP batch 96 do bỏ DP root-gather overhead, nhưng vẫn có rủi ro OOM trên T4 16GB.
- DDP global batch 128 không nên test sớm. Local batch mỗi GPU là 64, trong khi DP b64 hiện đã chạm khoảng 9.7-9.8GB allocated; scale activation lên local64 rất dễ OOM.
- torch.compile không nên bật trong DDP prototype đầu tiên. Code D12 có dict output, `index_add_`, runtime scale-2 edge attr, attention/transformer blocks và trước đó đã có Dynamo/recompile warning trong DP.

Recommendation: Do, nhưng làm sau audit bằng `scripts/train_d5a_ddp.py` riêng, không sửa model D10/D11/D12, không đổi dataset/artifact format, không refactor lớn trainer.

## 2. Current DP Baseline

### Files inspected

- `scripts/train_d5a.py`
- `scripts/common.py`
- `training/trainer.py`
- `training/optimizer.py`
- `training/losses.py`
- `data/full_graph_dataset.py`
- `data/graph_repository.py`
- `data/graph_resolver.py`
- `evaluation/evaluator.py`
- `models/registry.py`
- `models/d12_global_local_motif_model.py`
- `configs/experiments/d12a_stable_ce_first.yaml`
- `configs/experiments/d12a_stable_ce_first_fastio_safe.yaml`
- `configs/experiments/d12a_no_global.yaml`
- `configs/experiments/d12a_stable_ce_first_fastio.yaml`

Note: `configs/experiments/d12a_stable_ce_first_amp_b64.yaml` is not present in this checkout.

### Batch schema

`FullGraphDataset.__getitem__` returns one resolved graph sample. `collate_fn_full_graph` stacks samples into dense batch tensors.

Current effective schema:

| Key | Shape | Source |
|---|---:|---|
| `batch["x"]` | `[B, 2304, 7]` | stacked node features |
| `batch["node_features"]` | `[B, 2304, 7]` | alias of `x` |
| `batch["edge_index"]` | `[B, 2, 17860]` | `batch[0]["edge_index"].unsqueeze(0).expand(B, -1, -1).contiguous()` |
| `batch["edge_attr"]` | `[B, 17860, 5]` | stacked resolved static+dynamic edge attrs |
| `batch["node_mask"]` | `[B, 2304]` | all true |
| `batch["y"]` | `[B]` | long labels |
| `batch["graph_id"]` | `[B]` | graph ids |
| `batch["sample_idx"]` | `[B]` | dataset indices |

Important confirmations:

- `edge_index` is currently `[B, 2, E]`, not `[2, E]`, after collate.
- D12 explicitly accepts `[B, 2, E]` and immediately converts it to `[2, E]` with `edge_index = edge_index[0]` in both `D12PixelEncoder.forward` and `D12GlobalLocalMotifModel._parse_inputs`.
- DataParallel currently splits dict tensors along dimension 0. Therefore each GPU receives a shard with `x [local_B, 2304, 7]`, `edge_attr [local_B, 17860, 5]`, `edge_index [local_B, 2, 17860]`, then D12 uses the first topology inside each shard.
- This is safe only because the graph topology is shared for all samples. If future graph artifacts made per-sample topology differ, `edge_index[0]` would become invalid.

### Training flow

Current flow from `scripts/train_d5a.py`:

1. `run_train(config)` builds train and val loaders using `build_dataloader(config, split="train"/"val")`.
2. `create_trainer(config)` builds model, loss, optimizer, scheduler, output root, config snapshot, and `D5Trainer`.
3. Optional resume/init checkpoint is loaded before `trainer.fit(...)`.
4. `trainer.fit(...)` handles epoch loop, train, val, scheduler, checkpoint, early stopping, history, and W&B finish.
5. `log_experiment(trainer.output_root)` runs after training.

Where core objects are built:

| Object | Location |
|---|---|
| model | `scripts/common.py::prepare_training_objects`, via `models.registry.build_model` |
| criterion | `scripts/common.py::prepare_training_objects`, via `training.losses.build_loss` |
| optimizer | `scripts/common.py::prepare_training_objects`, via `training.optimizer.build_optimizer` |
| scheduler | `scripts/common.py::prepare_training_objects`, via `training.optimizer.build_scheduler` |
| trainer | `scripts/common.py::create_trainer` |

Training mechanics:

| Concern | Current location |
|---|---|
| AMP autocast | `training/trainer.py::D5Trainer._autocast` |
| GradScaler | `D5Trainer.__init__`, one scaler per trainer process |
| backward | `D5Trainer.train_one_epoch`, `self.scaler.scale(loss).backward()` |
| `grad_clip_norm` | after `scaler.unscale_(optimizer)` in `train_one_epoch` |
| non-finite grad skip under AMP | `train_one_epoch`, local skip and scaler backoff |
| torch.compile | `D5Trainer.__init__`, after DataParallel wrapping |
| DataParallel wrap | `D5Trainer.__init__`, when `training.multi_gpu=true`, CUDA, and device count > 1 |
| checkpoint save | `D5Trainer.save_checkpoint`, `output_root/checkpoints/best.pth` and `last.pth` |
| checkpoint load | `D5Trainer.load_checkpoint`, loads into `self._raw_model` |
| W&B | `D5Trainer.__init__` and `_log_metrics` |
| training history | `D5Trainer._save_history`, `output_root/training_history.json` |
| validation | `D5Trainer.fit` calls `self.validate(val_loader, ...)` |
| test evaluation | separate `scripts/evaluate_d5a.py` and `evaluation/evaluator.py` |

Checkpoint behavior is already close to DDP-friendly because `save_checkpoint` saves the raw model when DataParallel is active and unwraps `_orig_mod` if compiled. DDP would still need a generic unwrap helper for `DistributedDataParallel.module`.

### DataLoader flow

`build_dataloader` lives in `scripts/common.py`.

Current data config behavior:

- `batch_size` is read from `data.batch_size`, falling back to `training.batch_size`.
- `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor` are read from `data` first, then `training`.
- `chunk_cache_size` is read from `data.chunk_cache_size`, with `graph_cache_chunks` as compatibility alias.
- `ChunkAwareBatchSampler` is used only when `split == "train"`, `shuffle=True`, and `chunk_aware_shuffle=true`.
- When `ChunkAwareBatchSampler` is active, DataLoader receives `batch_sampler`, not `batch_size` and `shuffle`.
- Val/test use ordinary sequential batching.

Current D12A fast config:

- `data.batch_size=64`
- `training.batch_size=64`
- `num_workers=2`
- `persistent_workers=true`
- `prefetch_factor=2` in fastio variants
- `chunk_cache_size=4`
- `graph_cache_chunks=4`
- `chunk_aware_shuffle=true`
- `training.amp=true`
- `training.multi_gpu=true`
- `training.use_compile=false`

Batch size meaning today: global batch in the single process DataParallel pipeline. DP receives `B=64`, then PyTorch scatters it across GPUs.

## 3. Required Code Changes for DDP

### A. Script entrypoint

Recommendation: create `scripts/train_d5a_ddp.py` riêng.

Reason:

- Existing `scripts/train_d5a.py` is stable and already benchmarked.
- DDP needs process setup, rank gating, sampler changes, validation synchronization, checkpoint/logging guards, and different batch-size semantics.
- Adding all of that behind flags inside the current script raises risk of breaking DP and notebook flows.
- A separate script can reuse `common.prepare_training_objects`, config loading, loss/model builders, and most trainer logic without changing model/dataset/artifact format.

Minimal-risk design:

- Keep `scripts/train_d5a.py` untouched for DP.
- Add `scripts/train_d5a_ddp.py` for torchrun only.
- Add small reusable helpers only if necessary, for example `unwrap_model()` or DDP-safe dataloader builder, without changing D12.

### B. Process setup

DDP prototype needs:

- launch with `torchrun --standalone --nproc_per_node=2`
- read `LOCAL_RANK`, `RANK`, `WORLD_SIZE`
- `torch.cuda.set_device(local_rank)`
- `torch.distributed.init_process_group(backend="nccl")`
- set `device = cuda:{local_rank}` in config/runtime
- rank predicates: `is_main_process = rank == 0`
- `dist.barrier()` around rank-0-only validation/checkpoint/logging
- cleanup with `dist.destroy_process_group()` in `finally`
- ideally set a generous timeout for Kaggle if validation on rank 0 is long

### C. Model wrapping

Current DataParallel wrap is in `D5Trainer.__init__`.

DDP should wrap after model is moved to the rank-local CUDA device and before training:

```python
model = model.to(device)
model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
```

Do not also enable current `multi_gpu=True` DataParallel. DDP mode must force `training.multi_gpu=false` or bypass the DP branch.

Checkpoint save should unwrap:

```python
raw = model.module if isinstance(model, DistributedDataParallel) else model
```

`find_unused_parameters`: start with `False`.

Why `False` is likely OK for D12A mainline:

- D12 forward always computes `logits`, `logits_local`, `motif_supcon`, `motif_embeddings`, `part_masks`, relation transformer output, and local/global heads.
- Current stable CE-first config has `lambda_supcon=0`, `lambda_div=0`, `lambda_spatial=0`, but CE main and CE local use class head, local head, encoder, slot attention, relation transformer, global branch and FiLM.
- `motif_supcon` projection is computed but unused when `lambda_supcon=0`; its parameters may be unused by loss in the stable config.
- `part_masks`/centers diagnostics are computed but diagnostics do not backprop.

This creates one DDP hazard: `supcon_proj` parameters may be unused when `lambda_supcon=0`, and possibly some diagnostic-only paths do not receive gradients. With `find_unused_parameters=False`, DDP can error if parameters participate in forward but not in loss. There are two safe options:

1. Phase 1 set `find_unused_parameters=True` for the first smoke only, then inspect overhead/warnings.
2. Preferably, keep `False` but ensure the DDP prototype config disables or routes unused trainable heads cleanly. This would be a model/loss behavior change, so it should not be part of this audit.

Recommendation for prototype: start with `find_unused_parameters=True` only for smoke if DDP errors, but measure final speed with `False` if stable. Record this explicitly in the benchmark table because `True` can slow DDP.

### D. Branches and unused params

D12 has `use_global_branch`. In `use_global_branch=false`, the model still constructs `virtual_node_gather` and `film_fusion` modules, but forward skips them and returns zero tensors for global context, virtual attention, gamma/beta. Under DDP, those module parameters will be unused.

Therefore:

- For Phase 1 DDP, use `d12a_stable_ce_first_fastio_safe.yaml` with `use_global_branch=true`.
- Do not use `d12a_no_global.yaml` as the first DDP test unless `find_unused_parameters=True`.
- If no-global becomes important later, either use `find_unused_parameters=True` or instantiate no-global variants without unused modules, but that would be model work and is outside this audit.

## 4. Batch Size Semantics

Current DP:

- `training.batch_size=64` and `data.batch_size=64` mean global batch 64.
- One Python process creates batch 64.
- DataParallel scatters it across two GPUs, roughly local batch 32 each.

DDP:

- Each rank owns its own DataLoader.
- If each rank uses `batch_size=64`, effective global batch becomes 128.
- That would silently change optimization, memory, step count, LR behavior, and comparability.

Recommendation:

- Keep `training.batch_size` and `data.batch_size` as global batch for config continuity.
- In DDP script, compute `per_rank_batch_size = global_batch_size // world_size`.
- Require divisibility by world size for prototype.
- Write resolved config fields into output:
  - `training.global_batch_size: 64`
  - `training.per_rank_batch_size: 32`
  - `training.world_size: 2`
  - keep original `training.batch_size: 64` for user-facing continuity
- Override `data.batch_size` to per-rank batch inside the DDP runtime config before building DataLoader.

For Phase 1:

| Config global batch | world size | per-rank batch | Expected optimizer batch |
|---:|---:|---:|---:|
| 64 | 2 | 32 | same as DP b64 |
| 96 | 2 | 48 | larger, fewer steps |
| 128 | 2 | 64 | high OOM risk |

## 5. Validation Strategy

Two options:

### Option 1: Rank 0 validates full val, other ranks wait

Flow:

1. All ranks train one epoch.
2. `dist.barrier()`.
3. Rank 0 runs validation on full val loader using unwrapped model or DDP model on rank 0 device.
4. Rank 0 computes metrics, checkpoint decisions, scheduler monitor, early stopping.
5. Rank 0 broadcasts a small control object/tensor: monitor values, should_stop, best_metric, best_epoch, scheduler state decision if needed.
6. `dist.barrier()`.
7. All ranks continue or stop together.

Pros:

- Minimal metric mismatch risk.
- Reuses current `D5Trainer.validate` almost unchanged.
- No all-gather of logits/preds needed.
- Matches existing DP val metrics exactly, aside from nondeterminism.

Cons:

- Other GPU idles during validation.
- Need barrier and broadcast to avoid deadlock.
- Long rank-0 validation can hit distributed timeout on Kaggle if timeout is too short.

### Option 2: Distributed validation with all_gather y_true/y_pred

Flow:

1. Val DataLoader uses distributed sampler or rank-strided split.
2. Each rank predicts its shard.
3. Gather `y_true`, `y_pred`, optional losses and diagnostics.
4. Rank 0 computes metrics/checkpoint.

Pros:

- Faster validation.
- More symmetric DDP utilization.

Cons:

- More implementation surface.
- Need careful handling for uneven final batches, duplicated samples from `DistributedSampler`, deterministic ordering, diagnostics averaging, and classification report consistency.
- Greater risk of metric mismatch against existing DP baseline.

Recommendation for first DDP prototype: Option 1, rank-0 full validation. It is slower but much less risky and enough to benchmark training speed.

## 6. Logging/Checkpoint Strategy

Rank gating is mandatory.

Only rank 0 should:

- create final output directory as canonical run directory
- write `resolved_config.yaml`
- write `checkpoints/best.pth`
- write `checkpoints/last.pth`
- write `training_history.json`
- call `wandb.init`, `wandb.log`, `wandb.finish`
- print full epoch summaries
- call `log_experiment`
- run test evaluation or write evaluation artifacts

All ranks may print very short rank-tagged startup lines, but benchmark logs are easier to read if non-rank0 output is minimal.

Best checkpoint:

- rank 0 computes `val_macro_f1`.
- rank 0 decides best epoch.
- rank 0 saves checkpoint from unwrapped model.
- other ranks do not write.

Early stopping:

- rank 0 computes `stale_epochs` and `should_stop`.
- broadcast `should_stop` to all ranks after validation/checkpoint.
- all ranks break together.

Scheduler:

- For current `cosine_warmup`, scheduler steps by epoch and does not need val monitor.
- If using `ReduceLROnPlateau`, rank 0 monitor value must be broadcast or all ranks must receive the same monitor value before calling `scheduler.step(monitor_value)`.

Checkpoint compatibility:

- Save raw model state without `module.` prefixes.
- Current non-DDP `load_checkpoint_model` should remain able to load DDP-produced checkpoints if state dict is unwrapped.
- DDP resume must load same raw state into each rank before wrapping or into `model.module` after wrapping.

## 7. AMP/Compile Strategy

### AMP with DDP

GradScaler per rank is normal and acceptable.

Current AMP flow is mostly compatible:

1. autocast forward/loss
2. `scaler.scale(loss).backward()`
3. `scaler.unscale_(optimizer)`
4. gradient clip
5. skip or `scaler.step(optimizer)`
6. `scaler.update()`

DDP hazard:

- Current non-finite grad skip is local. In DDP, if one rank skips and another rank steps, ranks diverge.

DDP-safe fix needed:

- after local non-finite detection, all-reduce a skip flag across ranks
- if any rank reports skip, all ranks skip optimizer step, zero grads, and update/backoff scaler consistently
- log AMP skipped step only on rank 0, with optional count of ranks that skipped

### torch.compile with DDP

Do not enable compile in Phase 1.

Reasons:

- DP compile already did not help and produced Dynamo/recompile warnings.
- D12 returns a large dict.
- D12 uses `index_add_`, runtime `_scale2_edge_attr(x)`, scale-2 edge construction buffers, transformer encoder, and several shape-dependent diagnostics.
- DDP adds process-level complexity; mixing compile early makes failures harder to attribute.

If testing later:

- First get DDP eager stable.
- Then test compile in a separate benchmark config.
- Preferred order to try first: compile raw model before DDP wrapping, then wrap compiled model in DDP.
- Keep a separate run name such as `ddp_b64_amp_compile`.
- Treat compile speedup as optional. If compile increases recompile warnings or memory, keep it off.

## 8. Risk Assessment Table

| Area | Rating | Notes |
|---|---|---|
| Difficulty | Medium | Process setup and rank control are standard; validation/checkpoint synchronization is the real work. |
| Risk to existing DP pipeline | Low if separate script, Medium/High if modifying `train_d5a.py` directly | Keep DP entrypoint untouched. |
| Risk of metric mismatch | Medium | Rank-0 full validation lowers this. Distributed validation raises it. |
| Risk of checkpoint/logging bugs | Medium | Current trainer writes from one process; DDP must gate all writes to rank 0. |
| Expected speedup | Medium | Forward/backward dominate; DDP can remove DP overhead but all-reduce adds cost. |
| Recommendation | Do | Do as isolated prototype, keep only if speedup >= 10-15% and metrics/checkpoints match. |

Specific risks:

| Risk | Severity | Mitigation |
|---|---|---|
| DDP deadlock | High | Barrier placement, broadcast early-stop/control state, generous NCCL timeout. |
| Multiple ranks writing files | High | Rank-0-only output writes. |
| Validation mismatch | Medium | Phase 1 rank-0 full validation. |
| Batch size misunderstood | High | Treat config batch as global; compute per-rank batch explicitly. |
| DistributedSampler changes shuffle | Medium | Accept for DDP, set `sampler.set_epoch(epoch)`, record seed/world size. |
| ChunkAwareBatchSampler incompatible with DistributedSampler | Medium/High | First DDP prototype should disable chunk-aware sampler or implement a rank-aware chunk sampler. |
| Worker/RAM doubled | Medium | `num_workers=2` per rank means 4 workers total; chunk cache 4 per process doubles cache RAM. |
| `find_unused_parameters` | Medium | `supcon_proj` and no-global modules can be unused depending on config/loss. |
| Large output dict | Low/Medium | DDP does not sync outputs, but compile and gather can suffer. |
| Checkpoint state dict prefixes | Medium | Always unwrap DDP before save. |
| Kaggle torchrun/notebook compatibility | Medium | Use notebook shell cell with `torchrun`; avoid multiprocessing launch from inside Python. |
| AMP skip divergence | High | All-reduce skip flag before optimizer step. |

## 9. Expected Speedup Table

Baseline used for estimates:

- DP + AMP + compile off + global batch64
- estimated full epoch: about 5.59-5.60 min/epoch
- first batch schema in benchmark: `x [64, 2304, 7]`, `edge_attr [64, 17860, 5]`
- CUDA max allocated: about 9.72GB
- average profile around epoch end: data 0.0128s, forward 0.4157s, backward 0.3021s, batch 0.7311s

### DDP global batch64, AMP on, compile off

| Estimate | Minutes/epoch | Speedup vs 5.59 |
|---|---:|---:|
| Best-case | 4.30-4.60 | 18-23% |
| Expected | 4.70-5.20 | 7-16% |
| Worst-case | 5.60-6.00 | 0% to -7% |

Interpretation: Worth prototyping because forward/backward dominate and DP overhead is real. But if Phase 1 only improves about 5%, the complexity is probably not worth keeping for routine training.

### DDP global batch96, AMP on, compile off

| Item | Estimate |
|---|---|
| per-rank batch | 48 |
| fit chance | plausible but not guaranteed |
| OOM risk | medium/high |
| expected minutes/epoch if fit | 3.70-4.40 |
| best-case | 3.40-3.70 |
| worst-case | OOM or 4.80-5.30 if memory pressure hurts |

Interpretation: This is the most interesting DDP variant after b64 is stable. DP batch96 OOM does not prove DDP b96 OOM, because DDP avoids DP gather/replica overhead in one process. But local batch48 is still 1.5x local b32, so T4 memory may be tight.

### DDP global batch128, AMP on, compile off

| Item | Estimate |
|---|---|
| per-rank batch | 64 |
| fit chance | low |
| OOM risk | high |
| expected minutes/epoch if fit | 3.00-3.80 |
| recommendation | do not test until b96 fits with comfortable memory |

Interpretation: Not a Phase 2 default. Current DP b64 already allocates about 9.7GB. Local b64 DDP can exceed T4 memory once activations scale.

### DDP + torch.compile

| Scenario | Estimate |
|---|---|
| Potential speedup if stable | 0-10%, rarely 15% |
| Potential slowdown | yes, especially with recompiles |
| Memory risk | medium |
| Recommendation | only test after eager DDP is stable |

### 100-epoch impact

Using 5.59 min/epoch baseline:

| Scenario | Epoch min | 100 epoch time | Saved vs DP |
|---|---:|---:|---:|
| DP baseline | 5.59 | 9.32 h | 0 h |
| DDP b64 weak +5% | 5.31 | 8.85 h | 0.47 h |
| DDP b64 expected +12% | 4.92 | 8.20 h | 1.12 h |
| DDP b64 good +18% | 4.58 | 7.63 h | 1.69 h |
| DDP b96 expected | 4.00 | 6.67 h | 2.65 h |
| DDP b96 best | 3.50 | 5.83 h | 3.49 h |

Decision threshold:

- If DDP gives only 5% speedup, do not keep it unless needed for larger batch experiments.
- If DDP gives 10-15% speedup with identical metric/checkpoint behavior, keep it as optional Kaggle path.
- If DDP b96 fits and gives 20%+ speedup, it is worth using for 100-epoch D12A runs.

## 10. Recommendation: Is DDP Worth Implementing?

Yes, DDP is worth implementing as an isolated prototype.

It is not guaranteed to be a dramatic win at global batch64, but the current profile is favorable for a DDP trial: DataLoader time is small, forward/backward dominate, and DP has known single-process scatter/gather overhead. The main practical value may come from DDP b96 if local batch48 fits.

Do not replace the DP path until DDP passes:

- b64 smoke train with no deadlock
- checkpoint load/save compatibility
- rank-0 validation metric sanity
- no duplicate output writes
- benchmark shows at least 10-15% speedup or enables stable b96 with meaningful epoch-time reduction

## 11. Proposed DDP Implementation Plan

### Phase 0: Audit-only / no code changes

Status: this report.

No DDP code, no model changes, no dataset format changes.

### Phase 1: Safe DDP prototype

Target:

- `scripts/train_d5a_ddp.py` riêng
- config: `configs/experiments/d12a_stable_ce_first_fastio_safe.yaml`
- global batch64
- per-rank batch32
- AMP true
- compile false
- rank0 full validation
- rank0 checkpoint/logging/history
- W&B disabled or rank0 only
- benchmark cap: `max_train_batches=30`, `max_val_batches=10`
- `num_workers=2` per rank at first; if RAM/CPU pressure appears, try 1 per rank
- disable `chunk_aware_shuffle` in first DDP smoke unless a rank-aware chunk sampler is implemented

Implementation notes:

- Add DDP script, not a broad trainer refactor.
- Reuse `prepare_training_objects`.
- Build train loader with `DistributedSampler` for train.
- For first version, use ordinary per-rank DataLoader batching instead of `ChunkAwareBatchSampler`.
- Call `train_sampler.set_epoch(epoch)` before each train epoch.
- Implement rank-0 validation outside distributed val.
- Add all-rank AMP skip synchronization.
- Add rank-0-only output writes.

### Phase 2: DDP benchmark variants

Run variants:

- `ddp_b64_amp`
- `ddp_b96_amp`
- `ddp_b64_amp_compile`
- `ddp_b96_amp_compile`
- `ddp_b128_amp` only if b96 fits comfortably and max allocated memory leaves real headroom

Record for each:

- `estimated_full_epoch_minutes`
- average `data/to_device/forward/backward/optimizer/batch`
- CUDA max allocated per rank
- first-batch shapes
- AMP skipped steps
- validation macro F1 sanity
- whether `find_unused_parameters` was needed

### Phase 3: DDP production

Only after Phase 2 passes:

1. Run full 15-epoch stable CE-first config.
2. Compare with DP baseline:
   - same config global batch
   - same seed as much as DDP allows
   - same monitor `val_macro_f1`
   - checkpoint compatibility verified by `evaluate_d5a.py`
3. If model learns normally, run 100-epoch candidate.
4. Keep DP path as fallback.

## 12. Exact Benchmark Commands if Implemented

These commands assume a future `scripts/train_d5a_ddp.py` supports:

- `--config`
- `--environment`
- `--global_batch_size`
- `--max_train_batches`
- `--max_val_batches`
- `--num_workers`
- `--chunk_cache_size`
- `--no_chunk_aware_shuffle`
- `--amp`
- `--no_wandb`
- optional `--use_compile` / `--no_compile`

### Phase 1 smoke: DDP b64 AMP eager

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_stable_ce_first_fastio_safe.yaml \
  --environment kaggle \
  --global_batch_size 64 \
  --max_train_batches 30 \
  --max_val_batches 10 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --no_chunk_aware_shuffle \
  --amp \
  --no_compile \
  --no_wandb
```

### Phase 2: DDP b64 AMP eager benchmark

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_stable_ce_first_fastio_safe.yaml \
  --environment kaggle \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --no_chunk_aware_shuffle \
  --amp \
  --no_compile \
  --no_wandb
```

### Phase 2: DDP b96 AMP eager benchmark

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_stable_ce_first_fastio_safe.yaml \
  --environment kaggle \
  --global_batch_size 96 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --no_chunk_aware_shuffle \
  --amp \
  --no_compile \
  --no_wandb
```

### Phase 2: DDP b64 AMP compile benchmark

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_stable_ce_first_fastio_safe.yaml \
  --environment kaggle \
  --global_batch_size 64 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --no_chunk_aware_shuffle \
  --amp \
  --use_compile \
  --no_wandb
```

### Phase 2: DDP b96 AMP compile benchmark

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_stable_ce_first_fastio_safe.yaml \
  --environment kaggle \
  --global_batch_size 96 \
  --num_workers 2 \
  --chunk_cache_size 4 \
  --no_chunk_aware_shuffle \
  --amp \
  --use_compile \
  --no_wandb
```

### Phase 2 optional: DDP b128 AMP eager

Only run if b96 fits with comfortable memory.

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_d5a_ddp.py \
  --config configs/experiments/d12a_stable_ce_first_fastio_safe.yaml \
  --environment kaggle \
  --global_batch_size 128 \
  --max_train_batches 30 \
  --max_val_batches 10 \
  --num_workers 1 \
  --chunk_cache_size 2 \
  --no_chunk_aware_shuffle \
  --amp \
  --no_compile \
  --no_wandb
```

## Final Short Answers

### DDP có đáng làm không?

Có, đáng làm dưới dạng prototype riêng. Lý do chính là forward/backward đang chiếm phần lớn epoch time, DP batch64 đã ổn nhưng 100 epoch vẫn tốn khoảng 9.3 giờ, và DDP có thể tiết kiệm khoảng 1-1.7 giờ ở b64 hoặc hơn 2.5 giờ nếu b96 fit tốt.

### Độ khó ra sao?

Medium. DDP setup không quá khó, nhưng rủi ro nằm ở batch-size semantics, rank-0-only checkpoint/logging, validation synchronization, `DistributedSampler`, AMP skipped-step synchronization, và unused parameters của D12 khi một số loss/head bị tắt.

### Nên bắt đầu bằng config nào?

Bắt đầu bằng `configs/experiments/d12a_stable_ce_first_fastio_safe.yaml`, global batch64, per-rank batch32, AMP on, compile off, rank0 full validation, no W&B hoặc rank0 only, `max_train_batches=30`, `max_val_batches=10`.

Không bắt đầu bằng `d12a_no_global.yaml` vì `use_global_branch=false` có thể tạo unused parameters trong DDP.

### Ngưỡng speedup bao nhiêu thì giữ DDP?

- Dưới 5%: không đáng giữ làm đường train chính.
- 10-15%: giữ làm optional Kaggle DDP path nếu checkpoint/metrics ổn.
- 15-20% hoặc b96 fit ổn: đáng dùng cho run 100 epoch.
- B128 chỉ giữ nếu b96 đã fit tốt và còn dư VRAM rõ ràng.
