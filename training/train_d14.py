"""Train D14 performance-first runs.

D14 reuses the D13C diagnostic model/training stack but treats the run as a
classification performance sweep. It does not add prototype, motif-level
SupCon, or evidence claims. Optional augmentation is applied only to train
batches and only to the existing graph feature tensors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, load_config
from training import train_d13c as d13c
from training.trainer import move_to_device


ACTIVE_AUGMENTATION: Dict[str, Any] = {}


def _chance(p: float) -> bool:
    return bool(p > 0.0 and torch.rand(()) < float(p))


def _feature_grid(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or x.shape[1] != 48 * 48:
        return x
    return x.reshape(x.shape[0], 48, 48, x.shape[2])


def _recompute_dynamic_edges(batch: Dict[str, torch.Tensor]) -> None:
    if "edge_attr" not in batch or "edge_index" not in batch or "x" not in batch:
        return
    edge_attr = batch["edge_attr"]
    if edge_attr.ndim != 3 or edge_attr.shape[-1] < 5:
        return
    edge_index = batch["edge_index"]
    base = edge_index[0] if edge_index.ndim == 3 else edge_index
    if base.ndim != 2 or base.shape[0] != 2:
        return
    src = base[0].long()
    dst = base[1].long()
    intensity = batch["x"][..., 0].clamp(0.0, 1.0)
    delta = (intensity.index_select(1, src) - intensity.index_select(1, dst)).abs()
    edge_attr[..., 3] = delta
    edge_attr[..., 4] = torch.exp(-delta)


def _apply_graph_augmentation(batch: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return batch
    if "x" not in batch:
        return batch
    out = dict(batch)
    x = out["x"].clone()
    grid = _feature_grid(x)
    if grid is x:
        return batch

    dyn_cols = [idx for idx in (0, 3, 4, 5, 6) if idx < grid.shape[-1]]

    if _chance(float(cfg.get("horizontal_flip_p", 0.0))):
        grid[..., dyn_cols] = torch.flip(grid[..., dyn_cols], dims=[2])
        if grid.shape[-1] > 3:
            grid[..., 3] = -grid[..., 3]

    translate_p = float(cfg.get("translation_p", 0.0))
    max_shift = int(cfg.get("max_translate_pixels", 0) or 0)
    if max_shift > 0 and _chance(translate_p):
        shift_y = int(torch.randint(-max_shift, max_shift + 1, ()).item())
        shift_x = int(torch.randint(-max_shift, max_shift + 1, ()).item())
        shifted = torch.zeros_like(grid[..., dyn_cols])
        src_y0 = max(0, -shift_y)
        src_y1 = min(48, 48 - shift_y)
        src_x0 = max(0, -shift_x)
        src_x1 = min(48, 48 - shift_x)
        dst_y0 = max(0, shift_y)
        dst_y1 = min(48, 48 + shift_y)
        dst_x0 = max(0, shift_x)
        dst_x1 = min(48, 48 + shift_x)
        shifted[:, dst_y0:dst_y1, dst_x0:dst_x1, :] = grid[:, src_y0:src_y1, src_x0:src_x1, dyn_cols]
        grid[..., dyn_cols] = shifted

    if _chance(float(cfg.get("brightness_contrast_p", 0.0))):
        brightness = float(cfg.get("brightness", 0.08))
        contrast = float(cfg.get("contrast", 0.10))
        b = (torch.rand((grid.shape[0], 1, 1), device=grid.device, dtype=grid.dtype) * 2 - 1) * brightness
        c = 1.0 + (torch.rand((grid.shape[0], 1, 1), device=grid.device, dtype=grid.dtype) * 2 - 1) * contrast
        grid[..., 0] = ((grid[..., 0] - 0.5) * c + 0.5 + b).clamp(0.0, 1.0)
        for idx in (3, 4, 5, 6):
            if idx < grid.shape[-1]:
                grid[..., idx] = (grid[..., idx] * c).clamp(-1.0, 1.0)

    if _chance(float(cfg.get("gaussian_noise_p", 0.0))):
        std = float(cfg.get("gaussian_noise_std", 0.025))
        grid[..., 0] = (grid[..., 0] + torch.randn_like(grid[..., 0]) * std).clamp(0.0, 1.0)

    if _chance(float(cfg.get("random_erasing_p", 0.0))):
        max_frac = float(cfg.get("random_erasing_max_frac", 0.18))
        erase_h = max(2, int(48 * max_frac))
        erase_w = max(2, int(48 * max_frac))
        for bidx in range(grid.shape[0]):
            h = int(torch.randint(2, erase_h + 1, ()).item())
            w = int(torch.randint(2, erase_w + 1, ()).item())
            y0 = int(torch.randint(0, 49 - h, ()).item())
            x0 = int(torch.randint(0, 49 - w, ()).item())
            fill = grid[bidx, :, :, 0].mean()
            grid[bidx, y0 : y0 + h, x0 : x0 + w, 0] = fill
            for idx in (3, 4, 5, 6):
                if idx < grid.shape[-1]:
                    grid[bidx, y0 : y0 + h, x0 : x0 + w, idx] = 0.0

    x = grid.reshape_as(x)
    out["x"] = x
    out["node_features"] = x
    if "edge_attr" in out:
        out["edge_attr"] = out["edge_attr"].clone()
        _recompute_dynamic_edges(out)
    return out


def _run_epoch_d14(
    model,
    criterion,
    loader,
    optimizer,
    device,
    epoch: int,
    split: str,
    max_batches: int | None,
    grad_clip: float,
    amp: bool,
):
    is_train = optimizer is not None
    model.train(is_train)
    d13c._set_model_epoch(model, epoch)
    amp_enabled = d13c._amp_is_enabled(amp, device)
    scaler = d13c._make_grad_scaler(amp_enabled)
    totals: Dict[str, float] = {}
    slot_totals: Dict[str, float] = {}
    pool_totals: Dict[str, float] = {}
    supcon_totals: Dict[str, float] = {}
    y_true, y_pred = [], []
    count = 0
    import time

    start = time.perf_counter()
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        if is_train:
            batch = _apply_graph_augmentation(batch, ACTIVE_AUGMENTATION)
        batch = move_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with d13c._autocast(amp_enabled):
                out = model(batch)
                loss_dict = criterion(out, batch["y"], batch)
                loss_dict = {key: d13c._reduce_loss_value(value) for key, value in loss_dict.items()}
                loss = loss_dict["loss"]
        d13c._finite_check(out, loss, split, epoch, batch_idx)
        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip and float(grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and float(grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
        pred = out["logits"].detach().argmax(dim=1)
        y_true.extend(batch["y"].detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + d13c._float(value)
        for key, value in d13c._slot_stats(out.get("aux", {})).items():
            slot_totals[key] = slot_totals.get(key, 0.0) + float(value)
        for key, value in d13c.compute_assignment_stats(out.get("aux", {})).items():
            pool_totals[key] = pool_totals.get(key, 0.0) + float(value)
        for key, value in d13c._supcon_stats(loss_dict).items():
            supcon_totals[key] = supcon_totals.get(key, 0.0) + float(value)
        count += 1
    if count == 0:
        raise RuntimeError(f"No batches processed for split={split}")
    metrics = {key: value / count for key, value in totals.items()}
    metrics.update(d13c._metrics(y_true, y_pred))
    metrics["seconds"] = float(time.perf_counter() - start)
    metrics["num_batches"] = int(count)
    metrics["num_samples"] = int(len(y_true))
    return (
        metrics,
        {k: v / count for k, v in slot_totals.items()},
        {k: v / count for k, v in pool_totals.items()},
        {k: v / count for k, v in supcon_totals.items()},
    )


def run_train(config: Dict[str, Any], output_dir: str | Path | None = None, device_arg: str | None = None):
    global ACTIVE_AUGMENTATION
    ACTIVE_AUGMENTATION = dict(config.get("augmentation", {}) or {})
    old_run_epoch = d13c._run_epoch
    d13c._run_epoch = _run_epoch_d14
    try:
        result = d13c.run_train(config, output_dir=output_dir, device_arg=device_arg)
    finally:
        d13c._run_epoch = old_run_epoch
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true", default=False)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    if args.output_dir:
        config.setdefault("paths", {})["resolved_output_root"] = args.output_dir
    run_train(config, output_dir=args.output_dir, device_arg=args.device)


if __name__ == "__main__":
    main()
