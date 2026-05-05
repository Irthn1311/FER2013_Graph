"""Clean full-graph trainer for D5A with profiling and AMP support."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from data.labels import EMOTION_NAMES
from evaluation.metrics import classification_report_dict
from evaluation.metrics import compute_metrics
from training.optimizer import step_scheduler

HARD_CLASS_NAMES = ("Angry", "Fear", "Sad", "Surprise")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(v, device) for v in value)
    return value


def _to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _sync(device: torch.device) -> None:
    """Synchronize CUDA device if applicable for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_mem_stats(device: torch.device) -> Dict[str, float]:
    """Return current CUDA memory stats in GiB (zeros on CPU)."""
    if device.type != "cuda":
        return {"cuda_allocated_gb": 0.0, "cuda_reserved_gb": 0.0, "cuda_max_allocated_gb": 0.0}
    gib = 1024 ** 3
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated(device) / gib,
        "cuda_reserved_gb": torch.cuda.memory_reserved(device) / gib,
        "cuda_max_allocated_gb": torch.cuda.max_memory_allocated(device) / gib,
    }


def _optimizer_lr_metrics(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
    if not lrs:
        return {}
    metrics = {
        "lr": lrs[0],
        "train_lr": lrs[0],
        "lr_min": min(lrs),
        "lr_max": max(lrs),
    }
    for idx, lr in enumerate(lrs):
        metrics[f"lr_group_{idx}"] = lr
    return metrics


def _normalize_metric_mode(mode: str) -> str:
    mode = str(mode or "max").strip().lower()
    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported metric mode: {mode!r}. Expected 'max' or 'min'.")
    return mode


def _initial_metric_value(mode: str) -> float:
    return float("inf") if mode == "min" else -float("inf")


def _is_improved(value: float, best_value: float, mode: str) -> bool:
    return value < best_value if mode == "min" else value > best_value


def _get_metric_value(metrics: Dict[str, float], name: str) -> float:
    if name in metrics:
        return float(metrics[name])
    available = ", ".join(sorted(metrics.keys()))
    raise KeyError(f"Metric {name!r} not found. Available metrics: {available}")


def _scheduler_requires_monitor(scheduler) -> bool:
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def _scheduler_name(scheduler) -> str:
    return "none" if scheduler is None else scheduler.__class__.__name__


def _cosine_mean_similarity(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        return vectors.new_tensor(0.0)
    normed = torch.nn.functional.normalize(vectors.float(), dim=1, eps=1e-8)
    sim = normed @ normed.transpose(0, 1)
    k = sim.shape[0]
    off_diag = sim.masked_select(~torch.eye(k, dtype=torch.bool, device=sim.device))
    return off_diag.mean() if off_diag.numel() > 0 else vectors.new_tensor(0.0)


def _pair_cosine(vectors: torch.Tensor, left: int, right: int) -> Optional[torch.Tensor]:
    if vectors.ndim != 2 or vectors.shape[0] <= max(left, right):
        return None
    a = vectors[left].float()
    b = vectors[right].float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-8)


class D5Trainer:
    """Trainer with one intentional route: full graph D5 retrieval."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: torch.device | str = "cpu",
        output_root: str | Path = "outputs",
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        grad_clip_norm: Optional[float] = 5.0,
        amp: bool = False,
        amp_init_scale: float = 65536.0,
        profile_batches: int = 0,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.output_root = Path(output_root)
        self.checkpoint_dir = self.output_root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.use_wandb = bool(use_wandb)
        self.grad_clip_norm = grad_clip_norm
        self.best_metric = -float("inf")
        self.best_epoch = -1
        self._logged_train_device = False
        self.wandb = None
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            run_name = wandb_run_name or f"{self.config.get('run', {}).get('config_name', 'd5a')}_{self.output_root.name}"
            self.wandb.init(
                project=wandb_project or "FER-GRAPH",
                entity=wandb_entity or None,
                name=run_name,
                config=self.config,
                dir=str(self.output_root),
            )
            print(
                f"[W&B] enabled project={wandb_project or 'FER-GRAPH'} "
                f"entity={wandb_entity or 'default'} run={run_name}"
            )

        # AMP setup
        self.amp_enabled = bool(amp) and self.device.type == "cuda"
        if bool(amp) and self.device.type != "cuda":
            print("[AMP] WARNING: AMP requested but device is not CUDA – AMP disabled.")
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled,
            init_scale=float(amp_init_scale),
        )
        print(f"[AMP] amp_enabled={self.amp_enabled} init_scale={float(amp_init_scale):.1f}")

        # Profiling
        self.profile_batches = int(profile_batches)
        # configured_bs for bs_mismatch validation (set by fit() before training)
        self._configured_bs: Optional[int] = None
        self._skipped_nonfinite_grad_batches = 0

        first_param = next(self.model.parameters())
        print(f"trainer device: {self.device}")
        print(f"model first parameter device: {first_param.device}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _autocast(self):
        """Return the appropriate autocast context manager."""
        try:
            # torch >= 1.10 supports torch.amp.autocast
            return torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled)
        except AttributeError:
            return torch.cuda.amp.autocast(enabled=self.amp_enabled)

    def _print_profile(self, batch_idx: int, times: Dict[str, float], mem: Dict[str, float]) -> None:
        print(
            f"[PROFILE batch={batch_idx}]\n"
            f"  data_time      ={times.get('data', 0.0):.4f}s\n"
            f"  to_device_time ={times.get('to_device', 0.0):.4f}s\n"
            f"  forward_time   ={times.get('forward', 0.0):.4f}s\n"
            f"  loss_time      ={times.get('loss', 0.0):.4f}s\n"
            f"  backward_time  ={times.get('backward', 0.0):.4f}s\n"
            f"  optimizer_time ={times.get('optimizer', 0.0):.4f}s\n"
            f"  batch_time     ={times.get('batch', 0.0):.4f}s\n"
            f"  cuda_allocated_gb    ={mem.get('cuda_allocated_gb', 0.0):.3f}\n"
            f"  cuda_reserved_gb     ={mem.get('cuda_reserved_gb', 0.0):.3f}\n"
            f"  cuda_max_allocated_gb={mem.get('cuda_max_allocated_gb', 0.0):.3f}"
        )

    def _print_profile_avg(
        self,
        requested: int,
        recorded: int,
        acc: Dict[str, float],
        total_train_batches: Optional[int] = None,
    ) -> None:
        """Print average profile over *recorded* batches.

        requested = profile_batches config value
        recorded  = actual number of batches we accumulated (may be < requested
                    if max_train_batches cut training short).
        """
        avg = {k: v / max(recorded, 1) for k, v in acc.items()}
        est_epoch_min: Optional[float] = None
        if total_train_batches is not None and avg.get("batch", 0.0) > 0:
            est_epoch_min = avg["batch"] * total_train_batches / 60.0
        est_str = f"{est_epoch_min:.2f}" if est_epoch_min is not None else "unknown"
        print(
            f"\n[PROFILE average first {requested} batches (recorded={recorded})]\n"
            f"  avg_data_time      ={avg.get('data', 0.0):.4f}s\n"
            f"  avg_to_device_time ={avg.get('to_device', 0.0):.4f}s\n"
            f"  avg_forward_time   ={avg.get('forward', 0.0):.4f}s\n"
            f"  avg_loss_time      ={avg.get('loss', 0.0):.4f}s\n"
            f"  avg_backward_time  ={avg.get('backward', 0.0):.4f}s\n"
            f"  avg_optimizer_time ={avg.get('optimizer', 0.0):.4f}s\n"
            f"  avg_batch_time     ={avg.get('batch', 0.0):.4f}s\n"
            f"  estimated_full_epoch_minutes={est_str}"
        )

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_one_epoch(
        self,
        loader,
        epoch: int,
        max_batches: Optional[int] = None,
        total_train_batches: Optional[int] = None,
        full_epoch_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        count = 0
        pred_count = torch.zeros(7, dtype=torch.long)
        y_true = []
        y_pred = []
        epoch_start = time.perf_counter()
        progress = tqdm(loader, desc=f"train {epoch}", leave=False)

        # Profiling accumulators
        profile_n = self.profile_batches          # requested
        profile_recorded = 0                       # actually accumulated
        profile_acc: Dict[str, float] = {k: 0.0 for k in (
            "data", "to_device", "forward", "loss", "backward", "optimizer", "batch"
        )}
        profile_avg_printed = False

        # first-batch shape logging (only on epoch 1, batch_idx 0)
        _first_batch_logged = epoch > 1  # skip for epochs after first

        # Data timer starts right before the first batch is fetched
        t_data_start = time.perf_counter()

        for batch_idx, batch in enumerate(progress):
            is_last = (max_batches is not None and batch_idx + 1 >= int(max_batches))

            if max_batches is not None and batch_idx >= int(max_batches):
                break

            # ---- first-batch shape + bs_mismatch (from actual train loop, epoch 1 only) ----
            if not _first_batch_logged:
                x_shape = list(batch["x"].shape)
                ea_shape = list(batch["edge_attr"].shape)
                n_samples = x_shape[0]
                print(f"[SPEED_BENCH] first_batch_x_shape={x_shape}")
                print(f"[SPEED_BENCH] first_batch_edge_attr_shape={ea_shape}")
                configured_bs = self._configured_bs
                if configured_bs is not None:
                    # Compute expected first-batch size:
                    # If the dataset is smaller than configured_bs, the last/only
                    # batch will have fewer samples — that is NOT a mismatch.
                    dataset_size = (
                        len(loader.dataset)
                        if hasattr(loader, "dataset")
                        else None
                    )
                    if dataset_size is not None:
                        expected_bs = min(configured_bs, dataset_size)
                    else:
                        expected_bs = configured_bs
                    if n_samples != expected_bs or ea_shape[0] != expected_bs:
                        print(
                            f"[SPEED_BENCH] bs_mismatch=True  "
                            f"(expected {expected_bs} = min({configured_bs}, "
                            f"dataset={dataset_size}), "
                            f"got x.shape[0]={n_samples}, "
                            f"edge_attr.shape[0]={ea_shape[0]})"
                        )
                    else:
                        print(
                            f"[SPEED_BENCH] bs_mismatch=False  "
                            f"(x.shape[0]={n_samples}, expected={expected_bs} \u2713)"
                        )
                _first_batch_logged = True

            do_profile = profile_n > 0 and batch_idx < profile_n

            # ---- data_time ----
            if do_profile:
                _sync(self.device)
                t_data_end = time.perf_counter()
                t_batch_start = t_data_end
                data_time = t_data_end - t_data_start

            # ---- to_device_time ----
            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            batch = move_to_device(batch, self.device)
            if do_profile:
                _sync(self.device)
                to_device_time = time.perf_counter() - t0

            self.optimizer.zero_grad(set_to_none=True)

            # ---- forward_time ----
            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            with self._autocast():
                out = self.model(batch)
            if do_profile:
                _sync(self.device)
                forward_time = time.perf_counter() - t0

            # ---- loss_time ----
            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            with self._autocast():
                loss_dict = self.criterion(out, batch["y"], batch)
            loss = loss_dict["loss"]
            if do_profile:
                _sync(self.device)
                loss_time = time.perf_counter() - t0

            # Device log (first batch only)
            if not self._logged_train_device:
                print(
                    "train tensor devices: "
                    f"x={batch['x'].device} edge_index={batch['edge_index'].device} "
                    f"edge_attr={batch['edge_attr'].device} y={batch['y'].device} "
                    f"logits={out['logits'].device} loss={loss.device}"
                )
                self._logged_train_device = True

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at batch {batch_idx}")

            # ---- backward_time ----
            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            self.scaler.scale(loss).backward()
            if do_profile:
                _sync(self.device)
                backward_time = time.perf_counter() - t0

            # ---- optimizer_time ----
            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.grad_clip_norm)
                )
            else:
                grad_norm = self._compute_grad_norm()

            grad_norm_finite = bool(
                torch.isfinite(torch.as_tensor(grad_norm)).detach().cpu().item()
            )
            if not grad_norm_finite:
                if not self.amp_enabled:
                    raise FloatingPointError(f"Non-finite grad norm at batch {batch_idx}")
                self._skipped_nonfinite_grad_batches += 1
                totals["skipped_nonfinite_grad_batches"] = (
                    totals.get("skipped_nonfinite_grad_batches", 0.0) + 1.0
                )
                old_scale = self.scaler.get_scale()
                new_scale = old_scale * self.scaler.get_backoff_factor()
                self.scaler.update(new_scale=new_scale)
                new_scale = self.scaler.get_scale()
                self.optimizer.zero_grad(set_to_none=True)
                grad_norm = torch.zeros((), device=self.device)
                print(
                    f"[AMP] skipped optimizer step at epoch={epoch} batch={batch_idx} "
                    f"due to non-finite grad norm; scale {old_scale:.1f}->{new_scale:.1f}"
                )
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if do_profile:
                _sync(self.device)
                optimizer_time = time.perf_counter() - t0

            # ---- profiling record ----
            if do_profile:
                _sync(self.device)
                batch_time = time.perf_counter() - t_batch_start
                mem = _cuda_mem_stats(self.device)
                times = {
                    "data": data_time,
                    "to_device": to_device_time,
                    "forward": forward_time,
                    "loss": loss_time,
                    "backward": backward_time,
                    "optimizer": optimizer_time,
                    "batch": batch_time,
                }
                self._print_profile(batch_idx, times, mem)
                for k, v in times.items():
                    profile_acc[k] += v
                profile_recorded += 1

                # Print avg when requested window is full, OR on last batch
                # (in case max_batches cuts the run before profile_n).
                window_done = (batch_idx == profile_n - 1)
                last_profile_batch = is_last and not profile_avg_printed
                if (window_done or last_profile_batch) and not profile_avg_printed:
                    # Use full_epoch_batches for accurate epoch estimate;
                    # fall back to total_train_batches if not provided.
                    epoch_batches_for_estimate = full_epoch_batches or total_train_batches
                    self._print_profile_avg(
                        requested=profile_n,
                        recorded=profile_recorded,
                        acc=profile_acc,
                        total_train_batches=epoch_batches_for_estimate,
                    )
                    profile_avg_printed = True

            pred = out["logits"].detach().argmax(dim=1).cpu()
            pred_count += torch.bincount(pred, minlength=7)
            y_pred.extend(pred.tolist())
            y_true.extend(batch["y"].detach().cpu().tolist())
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + _to_float(value)
            totals["grad_norm"] = totals.get("grad_norm", 0.0) + _to_float(grad_norm)
            self._add_diagnostics(totals, out.get("diagnostics", {}))
            self._add_output_diagnostics(totals, out)
            count += 1
            progress.set_postfix(loss=f"{_to_float(loss):.4f}")

            # Reset data timer for next batch
            t_data_start = time.perf_counter()

        # If profiling window was never printed (e.g. profile_n=0 or nothing recorded),
        # print a final avg if we did accumulate something.
        if profile_recorded > 0 and not profile_avg_printed:
            epoch_batches_for_estimate = full_epoch_batches or total_train_batches
            self._print_profile_avg(
                requested=profile_n,
                recorded=profile_recorded,
                acc=profile_acc,
                total_train_batches=epoch_batches_for_estimate,
            )

        metrics = {f"train_{k}": v / max(count, 1) for k, v in totals.items()}
        if y_true:
            metrics.update({f"train_{k}": float(v) for k, v in compute_metrics(y_true, y_pred).items()})
        metrics["train_batches"] = float(count)
        elapsed = time.perf_counter() - epoch_start
        metrics["train_seconds"] = float(elapsed)
        metrics["train_sec_per_batch"] = float(elapsed / max(count, 1))
        for i, c in enumerate(pred_count.tolist()):
            metrics[f"train_pred_count_{i}"] = float(c)
        return metrics

    @torch.no_grad()
    def validate(self, loader, max_batches: Optional[int] = None, prefix: str = "val") -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        y_true = []
        y_pred = []
        count = 0
        pred_count = torch.zeros(7, dtype=torch.long)
        start_time = time.perf_counter()
        for batch_idx, batch in enumerate(tqdm(loader, desc=prefix, leave=False)):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_to_device(batch, self.device)
            with self._autocast():
                out = self.model(batch)
                loss_dict = self.criterion(out, batch["y"], batch)
            logits = out["logits"]
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite logits during {prefix} at batch {batch_idx}")
            pred = logits.argmax(dim=1)
            y_true.extend(batch["y"].detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())
            pred_count += torch.bincount(pred.detach().cpu(), minlength=7)
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + _to_float(value)
            self._add_diagnostics(totals, out.get("diagnostics", {}))
            self._add_output_diagnostics(totals, out)
            count += 1

        metrics = {f"{prefix}_{k}": v / max(count, 1) for k, v in totals.items()}
        elapsed = time.perf_counter() - start_time
        metrics[f"{prefix}_seconds"] = float(elapsed)
        metrics[f"{prefix}_sec_per_batch"] = float(elapsed / max(count, 1))
        cls_metrics = compute_metrics(y_true, y_pred) if y_true else {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }
        metrics.update({f"{prefix}_{k}": float(v) for k, v in cls_metrics.items()})
        self._add_selected_class_metrics(metrics, prefix, y_true, y_pred, pred_count)
        metrics[f"{prefix}_batches"] = float(count)
        for i, c in enumerate(pred_count.tolist()):
            metrics[f"{prefix}_pred_count_{i}"] = float(c)
        return metrics

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        monitor: Optional[str] = None,
        scheduler_mode: str = "max",
        checkpoint_monitor: Optional[str] = None,
        checkpoint_mode: str = "max",
        early_stopping_monitor: Optional[str] = None,
        early_stopping_mode: Optional[str] = None,
        early_stopping_patience: int = 20,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        import math
        dataset_size = getattr(train_loader.dataset, "__len__", lambda: None)()
        batch_size = getattr(train_loader, "batch_size", None)
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if batch_size is None and batch_sampler is not None:
            batch_size = getattr(batch_sampler, "batch_size", None)
        # Set configured_bs for bs_mismatch validation in train_one_epoch
        self._configured_bs = int(batch_size) if batch_size is not None else None
        # full_epoch_batches: used for estimated epoch time (NOT capped by max_train_batches).
        # total_train_batches: passed to train_one_epoch as the effective batches this run.
        if dataset_size is not None and batch_size is not None and batch_size > 0:
            try:
                full_epoch_batches = int(len(train_loader))
            except TypeError:
                full_epoch_batches = math.ceil(dataset_size / batch_size)
            total_train_batches = (
                min(full_epoch_batches, int(max_train_batches))
                if max_train_batches is not None
                else full_epoch_batches
            )
        else:
            full_epoch_batches = None
            total_train_batches = None

        checkpoint_monitor = str(checkpoint_monitor or "val_macro_f1")
        checkpoint_mode = _normalize_metric_mode(checkpoint_mode)
        early_stopping_monitor = str(early_stopping_monitor or checkpoint_monitor)
        early_stopping_mode = _normalize_metric_mode(early_stopping_mode or checkpoint_mode)
        scheduler_uses_monitor = _scheduler_requires_monitor(self.scheduler)
        if scheduler_uses_monitor:
            monitor = str(monitor or "val_loss")
            scheduler_mode = _normalize_metric_mode(scheduler_mode or "min")
        else:
            if self.scheduler is not None and monitor is not None:
                print(
                    f"[Scheduler] WARNING: {_scheduler_name(self.scheduler)} steps by epoch; "
                    f"ignoring scheduler monitor={monitor!r}."
                )
            monitor = None
        self.best_metric_name = checkpoint_monitor
        self.best_metric_mode = checkpoint_mode
        restored_metric_name = getattr(self, "restored_best_metric_name", None)
        restored_metric_mode = getattr(self, "restored_best_metric_mode", None)
        if self.best_epoch >= 0 and restored_metric_name and restored_metric_name != checkpoint_monitor:
            print(
                f"[Resume] WARNING: checkpoint best metric {restored_metric_name!r} "
                f"does not match current checkpoint monitor {checkpoint_monitor!r}; "
                "resetting best metric for the current policy."
            )
            self.best_metric = _initial_metric_value(checkpoint_mode)
            self.best_epoch = -1
        elif self.best_epoch >= 0 and restored_metric_mode and _normalize_metric_mode(restored_metric_mode) != checkpoint_mode:
            print(
                f"[Resume] WARNING: checkpoint best mode {restored_metric_mode!r} "
                f"does not match current checkpoint mode {checkpoint_mode!r}; "
                "resetting best metric for the current policy."
            )
            self.best_metric = _initial_metric_value(checkpoint_mode)
            self.best_epoch = -1
        if self.best_epoch < 0:
            self.best_metric = _initial_metric_value(checkpoint_mode)
        best_early_stopping_metric = _initial_metric_value(early_stopping_mode)
        stale_epochs = 0
        history = []
        start_epoch = int(getattr(self, "start_epoch", 1))
        print(
            f"[Scheduler] type={_scheduler_name(self.scheduler)} "
            f"monitor={monitor if scheduler_uses_monitor else 'epoch'} "
            f"mode={scheduler_mode if scheduler_uses_monitor else 'n/a'}"
        )
        print(f"[Checkpoint] save_best_metric={checkpoint_monitor} mode={checkpoint_mode}")
        print(f"[EarlyStopping] monitor={early_stopping_monitor} mode={early_stopping_mode}")
        if start_epoch > int(epochs):
            print(
                f"[Resume] start_epoch={start_epoch} is greater than epochs={int(epochs)}; "
                "no training epochs will run."
            )
        for epoch in range(start_epoch, int(epochs) + 1):
            if hasattr(self.criterion, "set_epoch"):
                self.criterion.set_epoch(epoch)
            lr_metrics_before = _optimizer_lr_metrics(self.optimizer)
            train_metrics = self.train_one_epoch(
                train_loader,
                epoch,
                max_batches=max_train_batches,
                total_train_batches=total_train_batches,
                full_epoch_batches=full_epoch_batches,
            )
            val_metrics = self.validate(val_loader, max_val_batches, prefix="val")
            metrics = {"epoch": float(epoch), **train_metrics, **val_metrics}
            metrics.update(lr_metrics_before)
            monitor_value = _get_metric_value(metrics, monitor) if scheduler_uses_monitor else None
            if scheduler_uses_monitor:
                metrics["scheduler_monitor_value"] = float(monitor_value)
                metrics[f"scheduler_monitor_{monitor}"] = float(monitor_value)
            checkpoint_value = _get_metric_value(metrics, checkpoint_monitor)
            early_stopping_value = _get_metric_value(metrics, early_stopping_monitor)
            metrics["checkpoint_monitor_value"] = checkpoint_value
            metrics[f"checkpoint_monitor_{checkpoint_monitor}"] = checkpoint_value
            metrics["early_stopping_monitor_value"] = early_stopping_value
            metrics[f"early_stopping_monitor_{early_stopping_monitor}"] = early_stopping_value
            lr_before_step = _optimizer_lr_metrics(self.optimizer)
            step_scheduler(self.scheduler, monitor_value)
            lr_after_step = _optimizer_lr_metrics(self.optimizer)
            if lr_after_step:
                metrics["lr_after_scheduler"] = lr_after_step.get("lr", 0.0)
                metrics["lr_min_after_scheduler"] = lr_after_step.get("lr_min", 0.0)
                metrics["lr_max_after_scheduler"] = lr_after_step.get("lr_max", 0.0)
                metrics["scheduler_lr_reduced"] = float(
                    lr_after_step.get("lr_min", 0.0) < lr_before_step.get("lr_min", 0.0)
                )
            checkpoint_improved = _is_improved(checkpoint_value, self.best_metric, checkpoint_mode)
            early_stopping_improved = _is_improved(
                early_stopping_value,
                best_early_stopping_metric,
                early_stopping_mode,
            )
            if checkpoint_improved:
                self.best_metric = checkpoint_value
                self.best_epoch = epoch
                self.save_checkpoint("best.pth", epoch, metrics)
            if early_stopping_improved:
                best_early_stopping_metric = early_stopping_value
                stale_epochs = 0
            else:
                stale_epochs += 1
            self.save_checkpoint("last.pth", epoch, metrics)
            history.append(metrics)
            self._log_metrics(metrics)
            print(
                f"Epoch {epoch:03d}/{int(epochs):03d} | "
                f"loss={metrics.get('train_loss', 0.0):.4f} "
                f"acc={metrics.get('train_accuracy', 0.0):.4f} "
                f"macro_f1={metrics.get('train_macro_f1', 0.0):.4f} | "
                f"val_loss={metrics.get('val_loss', 0.0):.4f} "
                f"val_acc={metrics.get('val_accuracy', 0.0):.4f} "
                f"val_macro_f1={metrics.get('val_macro_f1', 0.0):.4f} | "
                f"ckpt_monitor={checkpoint_monitor}:{checkpoint_value:.4f} "
                f"best_{checkpoint_monitor}={self.best_metric:.4f} "
                f"best_epoch={self.best_epoch} "
                f"early_{early_stopping_monitor}={best_early_stopping_metric:.4f} "
                f"sec/batch={metrics.get('train_sec_per_batch', 0.0):.3f}s"
            )
            print(
                "          val pred_count: "
                f"{[int(metrics.get(f'val_pred_count_{i}', 0.0)) for i in range(7)]}"
            )
            if checkpoint_improved:
                print(f"          checkpoint improvement: best_epoch={self.best_epoch}")
            if early_stopping_improved:
                print("          early stopping monitor improved")
            else:
                print(f"          no improvement: {stale_epochs}/{int(early_stopping_patience)}")
            if stale_epochs >= int(early_stopping_patience):
                print(f"Early stopping after {stale_epochs} stale epochs")
                break
        self._save_history(history)
        if self.wandb is not None:
            self.wandb.finish()
        return {"best_metric": self.best_metric, "best_epoch": self.best_epoch, "history": history}

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]) -> Path:
        path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_metric": float(self.best_metric),
            "best_epoch": int(self.best_epoch),
            "best_metric_name": getattr(self, "best_metric_name", None),
            "best_metric_mode": getattr(self, "best_metric_mode", None),
            "config": self.config,
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_epoch: bool = True,
        strict: bool = True,
        best_metric: Optional[float] = None,
        best_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        path = Path(checkpoint_path)
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state, strict=bool(strict))

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._move_optimizer_state_to_device()
        elif load_optimizer:
            print(f"[Resume] optimizer_state_dict not found in {path}; optimizer not restored.")

        if load_scheduler and self.scheduler is not None:
            scheduler_state = checkpoint.get("scheduler_state_dict")
            if scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
            else:
                print(f"[Resume] scheduler_state_dict not found in {path}; scheduler not restored.")

        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None and self.scaler is not None:
            self.scaler.load_state_dict(scaler_state)

        ckpt_epoch = int(checkpoint.get("epoch", 0) or 0)
        if load_epoch:
            self.start_epoch = ckpt_epoch + 1
        if best_metric is not None:
            self.best_metric = float(best_metric)
        elif "best_metric" in checkpoint:
            self.best_metric = float(checkpoint["best_metric"])
        else:
            metrics = checkpoint.get("metrics", {}) or {}
            monitor = self.config.get("training", {}).get("monitor", "val_macro_f1")
            if monitor in metrics:
                self.best_metric = float(metrics[monitor])
            elif f"scheduler_monitor_{monitor}" in metrics:
                self.best_metric = float(metrics[f"scheduler_monitor_{monitor}"])
            elif "scheduler_monitor_value" in metrics:
                self.best_metric = float(metrics["scheduler_monitor_value"])

        if best_epoch is not None:
            self.best_epoch = int(best_epoch)
        elif "best_epoch" in checkpoint:
            self.best_epoch = int(checkpoint["best_epoch"])
        elif ckpt_epoch > 0:
            self.best_epoch = ckpt_epoch

        self.restored_best_metric_name = checkpoint.get("best_metric_name")
        self.restored_best_metric_mode = checkpoint.get("best_metric_mode")

        print(
            f"[Resume] loaded {path} epoch={ckpt_epoch} "
            f"start_epoch={getattr(self, 'start_epoch', 1)} "
            f"best_epoch={self.best_epoch} best_metric={self.best_metric:.6f} "
            f"best_metric_name={self.restored_best_metric_name} "
            f"best_metric_mode={self.restored_best_metric_mode}"
        )
        return checkpoint

    def init_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        strict: bool = True,
    ) -> Dict[str, Any]:
        checkpoint = self.load_checkpoint(
            checkpoint_path,
            load_optimizer=False,
            load_scheduler=False,
            load_epoch=False,
            strict=strict,
        )
        self.start_epoch = 1
        self.best_metric = -float("inf")
        self.best_epoch = -1
        print(f"[Init] loaded model weights from {checkpoint_path}")
        return checkpoint

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _save_history(self, history) -> None:
        path = self.output_root / "training_history.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        if self.wandb is not None:
            self.wandb.log(metrics)

    @staticmethod
    def _add_diagnostics(totals: Dict[str, float], diagnostics: Dict[str, Any]) -> None:
        for key, value in diagnostics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                totals[f"diag_{key}"] = totals.get(f"diag_{key}", 0.0) + _to_float(value)

    @staticmethod
    def _add_output_diagnostics(totals: Dict[str, float], out: Dict[str, Any]) -> None:
        class_part_attn = out.get("class_part_attn")
        if torch.is_tensor(class_part_attn):
            attn = class_part_attn.detach().float()
            entropy = -(attn * attn.clamp_min(1e-8).log()).sum(dim=2)
            max_prob = attn.max(dim=2).values
            avg_by_class = attn.mean(dim=0)
            values: Dict[str, Optional[torch.Tensor]] = {
                "diag_class_part_entropy_mean": entropy.mean(),
                "diag_class_part_max_mean": max_prob.mean(),
                "diag_class_part_top1_mean": max_prob.mean(),
                "diag_class_part_similarity_mean": _cosine_mean_similarity(avg_by_class),
                "diag_class_part_similarity_sad_neutral": _pair_cosine(avg_by_class, 4, 6),
                "diag_class_part_similarity_fear_neutral": _pair_cosine(avg_by_class, 2, 6),
                "diag_class_part_similarity_fear_surprise": _pair_cosine(avg_by_class, 2, 5),
                "diag_class_part_similarity_disgust_angry": _pair_cosine(avg_by_class, 1, 0),
            }
            for key, value in values.items():
                if torch.is_tensor(value):
                    totals[key] = totals.get(key, 0.0) + _to_float(value)

        slot_area = out.get("slot_area")
        if torch.is_tensor(slot_area):
            area = slot_area.detach().float()
            area_norm = area / area.sum(dim=1, keepdim=True).clamp_min(1e-8)
            entropy = -(area_norm * area_norm.clamp_min(1e-8).log()).sum(dim=1)
            totals["diag_effective_slots"] = totals.get("diag_effective_slots", 0.0) + _to_float(entropy.exp().mean())

        border_mass = out.get("border_mass_per_slot", out.get("border_mass"))
        if torch.is_tensor(border_mass):
            border = border_mass.detach().float()
            totals["diag_border_mass_max"] = totals.get("diag_border_mass_max", 0.0) + _to_float(border.amax(dim=1).mean())

    @staticmethod
    def _add_selected_class_metrics(
        metrics: Dict[str, float],
        prefix: str,
        y_true,
        y_pred,
        pred_count: torch.Tensor,
    ) -> None:
        if not y_true:
            return
        report = classification_report_dict(y_true, y_pred)
        class_f1: Dict[str, float] = {}
        for label in EMOTION_NAMES:
            values = report.get(label, {})
            if not isinstance(values, dict):
                continue
            short_name = label.lower()
            precision = float(values.get("precision", 0.0))
            recall = float(values.get("recall", 0.0))
            f1 = float(values.get("f1-score", 0.0))
            class_f1[label] = f1
            metrics[f"{prefix}_{short_name}_precision"] = precision
            metrics[f"{prefix}_{short_name}_recall"] = recall
            metrics[f"{prefix}_{short_name}_f1"] = f1
            metrics[f"{prefix}_f1_{label}"] = f1

        hard_values = [class_f1[name] for name in HARD_CLASS_NAMES if name in class_f1]
        if hard_values:
            hard_avg = float(sum(hard_values) / len(hard_values))
            macro_f1 = float(metrics.get(f"{prefix}_macro_f1", 0.0))
            fear_f1 = float(class_f1.get("Fear", 0.0))
            metrics[f"{prefix}_hard_avg_f1"] = hard_avg
            metrics[f"{prefix}_macro_plus_025_fear"] = macro_f1 + 0.25 * fear_f1
            metrics[f"{prefix}_macro_plus_05_hard"] = macro_f1 + 0.5 * hard_avg

        label_to_idx = {name.lower(): idx for idx, name in enumerate(EMOTION_NAMES)}
        for short_name in label_to_idx:
            idx = label_to_idx.get(short_name)
            if idx is not None:
                metrics[f"{prefix}_pred_count_{short_name}"] = float(pred_count[idx].item())

    def _compute_grad_norm(self) -> torch.Tensor:
        norms = [
            p.grad.detach().norm(2)
            for p in self.model.parameters()
            if p.grad is not None
        ]
        if not norms:
            return torch.zeros((), device=self.device)
        return torch.norm(torch.stack(norms), p=2)
