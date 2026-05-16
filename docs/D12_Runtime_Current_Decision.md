# D12 Runtime Decision

## Current known results

Quality-safe b32 no AMP:

- DP 2 GPU
- batch32
- AMP off
- compile off
- best_val_macro_f1 = 0.2084 @ epoch 8
- active classes = 5
- PASS screen

Fastio_safe:

- DP 2 GPU
- batch64
- AMP on
- compile off
- best_val_macro_f1 = 0.1263
- collapse all-Happy after epoch 8
- 21 AMP skipped optimizer steps
- FAIL parity

## Why test b32 AMP no compile?

This test isolates AMP at batch32:

- If PASS: AMP at batch32 is usable; batch64 or batch64+AMP likely caused fastio failure.
- If FAIL: AMP is unsafe for D12A quality runs.

## Pass criteria

- best_val_macro_f1 >= 0.18-0.20
- active predicted classes >= 4
- no all-Happy collapse after epoch 8
- no repeated AMP skipped optimizer steps
- no NaN/Inf loss

## Fail criteria

- best_val_macro_f1 < 0.15
- active predicted classes <= 3
- all-Happy or near all-Happy prediction
- repeated AMP skipped optimizer steps
- NaN/Inf loss

## Run command

```bash
python scripts/train_d5a.py \
  --config configs/experiments/d12a_quality_b32_amp_no_compile_screen12.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --output_root /kaggle/working/outputs/d12_experiments/d12a_quality_b32_amp_no_compile_screen12 \
  --chunk_cache_size 8 \
  --no_wandb
```

## Decision after run

If PASS:

- Use `DP b32 AMP no compile` as the next quality runtime.

If FAIL:

- Use `DP b32 no AMP no compile` as the quality runtime.
- Do not use AMP/batch64 for D12A quality until further stabilization.
