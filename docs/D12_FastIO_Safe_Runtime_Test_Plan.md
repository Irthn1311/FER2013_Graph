# D12 FastIO Safe Runtime Test Plan

## 1. Muc tieu

Kiem tra runtime `DP + batch64 + AMP + no compile` co giu duoc quality cho D12A CE-first hay khong.

Runtime can test la `fastio_safe`: DataParallel, batch size 64, AMP bat, torch compile tat, chunk cache nho hon, va chunk-aware shuffle bat. Baseline doi chieu la quality-safe DP batch32 no AMP.

## 2. Vi sao screen 8 epoch la du

- Cac collapse D12A truoc day lo rat som, thuong tu epoch 2 den epoch 5.
- Neu den epoch 8 van all-Happy hoac active predicted classes <= 2 thi dung runtime do.
- Neu epoch 8 con song, chay tiep validation 15 epoch de xem quality co gan quality-safe hay khong.

## 3. Tieu chi PASS screen8

- Khong all-Happy sau epoch 2-3.
- Active predicted classes >= 4.
- `val_macro_f1 >= 0.16-0.18` o epoch 8 hoac best epoch.
- `val_pred_count` co Fear/Sad/Surprise/Neutral hoac phan bo tuong tu, khong chi tap trung vao mot class.
- Khong co AMP scaler skip, NaN, hoac Inf lien tuc.

## 4. Tieu chi FAIL

- `pred_count` all-Happy hoac active classes <= 2.
- `best_val_macro_f1 < 0.12` sau epoch 8.
- AMP skipped optimizer step lien tuc.
- Loss NaN hoac Inf.

## 5. Tieu chi PASS val15

- `best_val_macro_f1 >= 0.22`.
- `pred_count` nhieu class.
- Khong thap hon quality-safe b32 qua nhieu.

## 6. Lenh chay

Screen fastio:

```bash
python scripts/train_d5a.py \
  --config configs/experiments/d12a_fastio_safe_screen8.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --output_root /kaggle/working/outputs/d12_experiments/d12a_fastio_safe_screen8 \
  --chunk_cache_size 4 \
  --no_wandb
```

Validation 15:

```bash
python scripts/train_d5a.py \
  --config configs/experiments/d12a_fastio_safe_val15.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --output_root /kaggle/working/outputs/d12_experiments/d12a_fastio_safe_val15 \
  --chunk_cache_size 4 \
  --no_wandb
```

Quality-safe comparison:

```bash
python scripts/train_d5a.py \
  --config configs/experiments/d12a_quality_safe_b32_noamp_screen8.yaml \
  --environment kaggle \
  --graph_repo_path /kaggle/working/artifacts/graph_repo \
  --output_root /kaggle/working/outputs/d12_experiments/d12a_quality_safe_b32_noamp_screen8 \
  --chunk_cache_size 8 \
  --no_wandb
```

## 7. Cach doc ket qua

Doc `training_history.json` trong run output. Cac field quan trong:

- `val_macro_f1`, `val_accuracy`
- `train_macro_f1`, `train_accuracy`
- `val_pred_count_0..6`, va neu co local branch: `val_pred_count_local_0..6`
- `val_macro_f1_local` neu co
- `train_loss_ce`, `val_loss_ce`, `train_loss_local`, `val_loss_local`
- `train_diag_logits_std`, `val_diag_logits_std`
- `train_diag_slot_area_entropy`, `val_diag_slot_area_entropy`
- `train_diag_encoder_gate_mean/std`, `val_diag_encoder_gate_mean/std`
- `skipped_nonfinite_grad_batches` neu AMP gap grad non-finite

Doc nhanh best epoch bang Python:

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("/kaggle/working/outputs/d12_experiments/d12a_fastio_safe_screen8")
history_files = sorted(run.glob("*/training_history.json"))
history = json.loads(history_files[-1].read_text())
best = max(history, key=lambda e: e.get("val_macro_f1", -1))
counts = [int(best.get(f"val_pred_count_{i}", 0)) for i in range(7)]
active = sum(c > 0 for c in counts)
print("best_epoch", int(best["epoch"]))
print("best_val_macro_f1", best.get("val_macro_f1"))
print("val_accuracy", best.get("val_accuracy"))
print("active_classes", active)
print("val_pred_count", counts)
PY
```

Quyet dinh:

- Neu fastio screen8 pass: chay `d12a_fastio_safe_val15`.
- Neu fastio val15 gan quality-safe: dung `fastio_safe` lam runtime chinh.
- Neu fastio sap nhung quality-safe song: batch64/AMP van gay lech, dung b32 no AMP.
- Neu ca hai sap: audit code/config.

## 8. Smoke khong train

```bash
python scripts/check_d12_fastio_safe_configs.py
python scripts/smoke_d12_model.py
```

Hai lenh nay chi kiem config/model smoke. Khong train that.
