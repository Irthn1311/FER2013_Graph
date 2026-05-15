"""DDP Phase 1 trainer for D5/D12 full-graph FER experiments.

This entrypoint is intentionally separate from ``scripts/train_d5a.py`` so the
existing DataParallel path stays untouched.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    apply_cli_overrides,
    load_config,
    make_run_output_root,
    prepare_training_objects,
    resolve_path,
    save_config,
)
from data.ddp_chunk_aware_sampler import DDPChunkAwareBatchSampler  # noqa: E402
from data.full_graph_dataset import FullGraphDataset, collate_fn_full_graph  # noqa: E402
from evaluation.metrics import compute_metrics  # noqa: E402
from scripts.log_experiment import log_experiment  # noqa: E402
from training.optimizer import step_scheduler  # noqa: E402
from training.trainer import (  # noqa: E402
    D5Trainer,
    _cuda_mem_stats,
    _get_metric_value,
    _initial_metric_value,
    _is_improved,
    _normalize_metric_mode,
    _optimizer_lr_metrics,
    _scheduler_name,
    _scheduler_requires_monitor,
    _sync,
    _to_float,
    move_to_device,
)


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_rank0() -> bool:
    return _rank() == 0


def _rank_print(*args, **kwargs) -> None:
    if _is_rank0():
        print(*args, **kwargs)


def _quiet_non_rank0():
    if _is_rank0():
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def setup_ddp() -> tuple[int, int, int, torch.device]:
    rank = _rank()
    local_rank = _local_rank()
    world_size = _world_size()
    if not torch.cuda.is_available():
        raise RuntimeError("DDP Phase 1 requires CUDA. Kaggle accelerator must be GPU.")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=90),
        )
    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, world_size, device


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _broadcast_string(value: Optional[str], src: int = 0) -> str:
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    if payload[0] is None:
        raise RuntimeError("Broadcasted string payload is None")
    return str(payload[0])


def _all_reduce_bool(value: bool, device: torch.device) -> bool:
    flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _all_reduce_float(value: float, device: torch.device, op=dist.ReduceOp.SUM) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        if isinstance(model, DistributedDataParallel):
            model = model.module
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        break
    return model


def _autocast(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=enabled)


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected 'true' or 'false', got {value!r}")


def _compile_inner_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "compile"):
        model.compile()
        return model
    if hasattr(torch, "compile"):
        return torch.compile(model)
    return model


def _set_nested(config: Dict[str, Any], section: str, key: str, value: Any) -> None:
    config.setdefault(section, {})[key] = value


def resolve_global_batch_size(config: Dict[str, Any], cli_global_batch_size: Optional[int]) -> int:
    if cli_global_batch_size is not None:
        return int(cli_global_batch_size)
    data_cfg = config.get("data", {}) or {}
    training_cfg = config.get("training", {}) or {}
    return int(data_cfg.get("batch_size", training_cfg.get("batch_size", 16)) or 16)


def apply_ddp_runtime_overrides(
    config: Dict[str, Any],
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    global_batch_size: int,
    per_rank_batch_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    cfg = dict(config)
    data_cfg = dict(cfg.get("data", {}) or {})
    training_cfg = dict(cfg.get("training", {}) or {})
    logging_cfg = dict(cfg.get("logging", {}) or {})
    ddp_cfg = dict(cfg.get("ddp", {}) or {})

    data_cfg["batch_size"] = int(per_rank_batch_size)
    data_cfg["chunk_aware_shuffle"] = False
    if args.ddp_chunk_aware is not None:
        data_cfg["ddp_chunk_aware"] = bool(args.ddp_chunk_aware)
    else:
        data_cfg.setdefault("ddp_chunk_aware", True)
    data_cfg.setdefault("ddp_drop_last_batches", True)
    data_cfg.setdefault("fixed_batch_size", False)
    data_cfg.setdefault("drop_incomplete_batches", False)
    data_cfg.setdefault("carry_over_leftovers", False)
    training_cfg["batch_size"] = int(global_batch_size)
    training_cfg["global_batch_size"] = int(global_batch_size)
    training_cfg["per_rank_batch_size"] = int(per_rank_batch_size)
    training_cfg["world_size"] = int(world_size)
    training_cfg["rank"] = int(rank)
    training_cfg["local_rank"] = int(local_rank)
    training_cfg["device"] = str(device)
    training_cfg["multi_gpu"] = False
    if args.use_compile:
        training_cfg["use_compile"] = True
    elif args.no_compile:
        training_cfg["use_compile"] = False
    else:
        training_cfg["use_compile"] = _cfg_bool(training_cfg.get("use_compile", False), False)
    if args.compile_order is not None:
        training_cfg["compile_order"] = args.compile_order
    else:
        training_cfg.setdefault("compile_order", "before_ddp")
    data_cfg["num_workers"] = int(args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 2))
    training_cfg["num_workers"] = int(data_cfg["num_workers"])
    if args.chunk_cache_size is not None:
        data_cfg["chunk_cache_size"] = int(args.chunk_cache_size)
    data_cfg.setdefault("chunk_cache_size", 4)
    data_cfg.setdefault("graph_cache_chunks", data_cfg.get("chunk_cache_size", 4))
    if args.no_wandb:
        logging_cfg["use_wandb"] = False
    if getattr(args, "wandb", False):
        logging_cfg["use_wandb"] = True
    if getattr(args, "wandb_project", None) is not None:
        logging_cfg["project"] = str(args.wandb_project)
    if getattr(args, "wandb_entity", None) is not None:
        logging_cfg["entity"] = str(args.wandb_entity)

    cfg["data"] = data_cfg
    cfg["training"] = training_cfg
    cfg["logging"] = logging_cfg
    ddp_cfg.update(
        {
            "enabled": True,
            "backend": "nccl",
            "rank0_full_validation": True,
            "chunk_aware_shuffle_forced_off": True,
        }
    )
    find_unused_override = _parse_optional_bool(args.find_unused_parameters)
    if find_unused_override is not None:
        ddp_cfg["find_unused_parameters"] = find_unused_override
    else:
        ddp_cfg.setdefault("find_unused_parameters", True)
    cfg["ddp"] = ddp_cfg
    return cfg


def build_ddp_dataloader(
    config: Dict[str, Any],
    *,
    split: str,
    batch_size: int,
    rank: int,
    world_size: int,
    distributed_train: bool,
) -> tuple[DataLoader, Optional[Any]]:
    paths = config.get("paths", {}) or {}
    data_cfg = config.get("data", {}) or {}
    training_cfg = config.get("training", {}) or {}
    repo = resolve_path(paths.get("graph_repo_path", "artifacts/graph_repo"))

    chunk_cache_size = int(data_cfg.get("chunk_cache_size", data_cfg.get("graph_cache_chunks", 0)) or 0)
    graph_cache_chunks = data_cfg.get("graph_cache_chunks")
    if graph_cache_chunks is not None and chunk_cache_size <= 0:
        chunk_cache_size = int(graph_cache_chunks or 0)

    with _quiet_non_rank0():
        dataset = FullGraphDataset(
            repo_root=repo,
            split=split,
            chunk_cache_size=chunk_cache_size,
            graph_cache_chunks=int(graph_cache_chunks) if graph_cache_chunks is not None else None,
        )
    num_workers = int(data_cfg.get("num_workers", training_cfg.get("num_workers", 2)) or 0)
    pin_memory = _cfg_bool(data_cfg.get("pin_memory", training_cfg.get("pin_memory", True)), True)
    persistent_workers_cfg = _cfg_bool(
        data_cfg.get("persistent_workers", training_cfg.get("persistent_workers", True)),
        True,
    )
    persistent_workers = persistent_workers_cfg and num_workers > 0
    prefetch_factor_cfg = data_cfg.get("prefetch_factor", training_cfg.get("prefetch_factor"))
    prefetch_factor = int(prefetch_factor_cfg) if (prefetch_factor_cfg is not None and num_workers > 0) else None

    sampler: Optional[Any] = None
    batch_sampler: Optional[DDPChunkAwareBatchSampler] = None
    shuffle = False
    if distributed_train:
        if _cfg_bool(data_cfg.get("ddp_chunk_aware", True), True):
            batch_sampler = DDPChunkAwareBatchSampler(
                dataset,
                batch_size=int(batch_size),
                num_replicas=world_size,
                rank=rank,
                shuffle_chunks=_cfg_bool(data_cfg.get("shuffle_chunks", True), True),
                shuffle_within_chunk=_cfg_bool(data_cfg.get("shuffle_within_chunk", True), True),
                drop_last=False,
                seed=int(training_cfg.get("seed", config.get("run", {}).get("seed", 42))),
                ddp_drop_last_batches=_cfg_bool(data_cfg.get("ddp_drop_last_batches", True), True),
                fixed_batch_size=_cfg_bool(data_cfg.get("fixed_batch_size", False), False),
                drop_incomplete_batches=_cfg_bool(
                    data_cfg.get("drop_incomplete_batches", False),
                    False,
                ),
                carry_over_leftovers=_cfg_bool(data_cfg.get("carry_over_leftovers", False), False),
            )
            sampler = batch_sampler
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=int(training_cfg.get("seed", config.get("run", {}).get("seed", 42))),
                drop_last=False,
            )
    elif split == "train":
        shuffle = True

    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": collate_fn_full_graph,
    }
    if batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs["batch_size"] = int(batch_size)
        kwargs["shuffle"] = shuffle
        kwargs["sampler"] = sampler
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **kwargs)
    if _is_rank0():
        if batch_sampler is not None:
            summary = batch_sampler.summary(epoch=0)
            print(
                "[DDP ChunkAware] "
                "enabled=True "
                f"num_chunks={summary['num_chunks']} "
                f"chunk_sizes_min_mean_max="
                f"{summary['chunk_size_min']}/{summary['chunk_size_mean']:.2f}/{summary['chunk_size_max']} "
                f"total_samples={summary['total_samples']} "
                f"rank0_chunk_count={summary['rank_chunk_counts'][0]} "
                f"rank1_chunk_count={summary['rank_chunk_counts'][1] if world_size > 1 else 0} "
                f"per_rank_batch_size={batch_size} "
                f"batches_per_rank_before={summary['batches_before_balance']} "
                f"batches_per_rank_after={summary['batches_after_balance']} "
                f"truncated_batches={summary['truncated_batches']} "
                f"ddp_drop_last_batches={batch_sampler.ddp_drop_last_batches} "
                f"shuffle_chunks={batch_sampler.shuffle_chunks} "
                f"shuffle_within_chunk={batch_sampler.shuffle_within_chunk}"
            )
            print(
                "[DDP ChunkAware FixedShape] "
                f"fixed_batch_size={batch_sampler.fixed_batch_size} "
                f"drop_incomplete_batches={batch_sampler.drop_incomplete_batches} "
                f"carry_over_leftovers={batch_sampler.carry_over_leftovers} "
                f"per_rank_batch_size={batch_size} "
                f"dropped_samples_per_rank={summary['dropped_samples_per_rank']} "
                f"unique_batch_sizes_per_rank={summary['unique_batch_sizes_per_rank']} "
                f"batches_per_rank_before_balance={summary['batches_before_balance']} "
                f"batches_per_rank_after_balance={summary['batches_after_balance']}"
            )
        else:
            print("[DDP ChunkAware] enabled=False")
        print(
            f"[DDP DataLoader split={split}] "
            f"num_samples={len(dataset)} batch_size={batch_size} "
            f"num_workers={num_workers} pin_memory={pin_memory} "
            f"persistent_workers={persistent_workers} prefetch_factor={prefetch_factor} "
            f"chunk_cache_size={chunk_cache_size} distributed_sampler={isinstance(sampler, DistributedSampler)} "
            f"chunk_aware_sampler={batch_sampler is not None} batches_per_epoch={len(loader)}"
        )
    return loader, sampler


class DDPPhase1Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        config: Dict[str, Any],
        output_root: Path,
    ) -> None:
        self.model = model
        self.raw_model = _unwrap_model(model)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.output_root = Path(output_root)
        self.checkpoint_dir = self.output_root / "checkpoints"
        self.rank = _rank()
        self.local_rank = _local_rank()
        self.world_size = _world_size()
        self.is_rank0 = self.rank == 0
        self.amp_enabled = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled,
            init_scale=float(config.get("training", {}).get("amp_init_scale", 65536.0)),
        )
        self.grad_clip_norm = config.get("training", {}).get("grad_clip_norm", 5.0)
        self.profile_batches = int(config.get("training", {}).get("profile_batches", 20) or 0)
        self.best_metric = -float("inf")
        self.best_epoch = -1
        self._logged_train_device = False
        self.wandb = None
        logging_cfg = config.get("logging", {}) or {}
        if self.is_rank0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if bool(logging_cfg.get("use_wandb", False)):
                import wandb

                self.wandb = wandb
                run_name = logging_cfg.get("run_name") or (
                    f"{config.get('run', {}).get('config_name', 'd5a_ddp')}_{self.output_root.name}"
                )
                self.wandb.init(
                    project=logging_cfg.get("project") or "FER-GRAPH",
                    entity=logging_cfg.get("entity") or None,
                    name=run_name,
                    config=config,
                    dir=str(self.output_root),
                )
        if self.is_rank0:
            print(
                f"[DDP AMP] amp_enabled={self.amp_enabled} "
                f"init_scale={float(config.get('training', {}).get('amp_init_scale', 65536.0)):.1f}"
            )

    def train_one_epoch(
        self,
        loader: DataLoader,
        *,
        epoch: int,
        max_batches: Optional[int],
        full_epoch_batches: Optional[int],
    ) -> Dict[str, float]:
        self.model.train()
        if hasattr(self.criterion, "set_epoch"):
            self.criterion.set_epoch(epoch)

        totals: Dict[str, float] = {}
        count = 0
        y_true = []
        y_pred = []
        pred_count = torch.zeros(7, dtype=torch.long)
        skipped_batches = 0
        epoch_start = time.perf_counter()
        profile_n = self.profile_batches
        profile_recorded = 0
        profile_acc: Dict[str, float] = {k: 0.0 for k in (
            "data", "to_device", "forward", "loss", "backward", "optimizer", "batch"
        )}
        profile_avg_printed = False
        t_data_start = time.perf_counter()
        progress = tqdm(
            loader,
            desc=f"train {epoch} rank{self.rank}",
            leave=False,
            disable=not self.is_rank0,
        )

        for batch_idx, batch in enumerate(progress):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            is_last = max_batches is not None and batch_idx + 1 >= int(max_batches)
            do_profile = self.is_rank0 and profile_n > 0 and batch_idx < profile_n
            actual_batch_size = int(batch["x"].shape[0])
            expected_batch_size = int(self.config["training"]["per_rank_batch_size"])
            local_fixed_shape_violation = actual_batch_size != expected_batch_size
            if local_fixed_shape_violation:
                print(
                    "[DDP FixedShape Violation] "
                    f"rank={self.rank} epoch={epoch} batch={batch_idx} "
                    f"actual_batch_size={actual_batch_size} expected_batch_size={expected_batch_size}"
                )

            if self.is_rank0 and epoch == 1 and batch_idx == 0:
                print(f"[SPEED_BENCH] first_batch_x_shape={list(batch['x'].shape)}")
                print(f"[SPEED_BENCH] first_batch_edge_attr_shape={list(batch['edge_attr'].shape)}")
                expected_bs = min(
                    int(self.config["training"]["per_rank_batch_size"]),
                    len(loader.dataset),
                )
                got_bs = int(batch["x"].shape[0])
                print(
                    f"[SPEED_BENCH] ddp_per_rank_bs_mismatch={got_bs != expected_bs} "
                    f"(rank0 batch={got_bs}, expected={expected_bs})"
                )

            if do_profile:
                _sync(self.device)
                t_data_end = time.perf_counter()
                t_batch_start = t_data_end
                data_time = t_data_end - t_data_start
                _sync(self.device)
                t0 = time.perf_counter()

            batch = move_to_device(batch, self.device)

            if do_profile:
                _sync(self.device)
                to_device_time = time.perf_counter() - t0

            self.optimizer.zero_grad(set_to_none=True)

            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            with _autocast(self.device, self.amp_enabled):
                out = self.model(batch)
            if do_profile:
                _sync(self.device)
                forward_time = time.perf_counter() - t0

            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            with _autocast(self.device, self.amp_enabled):
                loss_dict = self.criterion(out, batch["y"], batch)
            loss = loss_dict["loss"]
            if do_profile:
                _sync(self.device)
                loss_time = time.perf_counter() - t0

            if self.is_rank0 and not self._logged_train_device:
                print(
                    "train tensor devices: "
                    f"x={batch['x'].device} edge_index={batch['edge_index'].device} "
                    f"edge_attr={batch['edge_attr'].device} y={batch['y'].device} "
                    f"logits={out['logits'].device} loss={loss.device}"
                )
                self._logged_train_device = True

            loss_bad = _all_reduce_bool(
                not bool(torch.isfinite(loss).detach().cpu().item()),
                self.device,
            )
            if loss_bad:
                raise FloatingPointError(
                    f"Non-finite training loss on at least one rank at epoch={epoch} batch={batch_idx}"
                )

            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            self.scaler.scale(loss).backward()
            if do_profile:
                _sync(self.device)
                backward_time = time.perf_counter() - t0

            if do_profile:
                _sync(self.device)
                t0 = time.perf_counter()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(self.grad_clip_norm),
                )
            else:
                grad_norm = self._compute_grad_norm()
            local_grad_finite = bool(torch.isfinite(torch.as_tensor(grad_norm)).detach().cpu().item())
            skip_step = _all_reduce_bool(not local_grad_finite, self.device)

            if skip_step:
                skipped_batches += 1
                old_scale = self.scaler.get_scale()
                new_scale = old_scale * self.scaler.get_backoff_factor()
                self.scaler.update(new_scale=new_scale)
                self.optimizer.zero_grad(set_to_none=True)
                grad_norm = torch.zeros((), device=self.device)
                if self.is_rank0:
                    print(
                        f"[AMP][DDP] skipped optimizer step at epoch={epoch} batch={batch_idx} "
                        f"because at least one rank had non-finite grad; scale "
                        f"{old_scale:.1f}->{self.scaler.get_scale():.1f}"
                    )
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()

            if do_profile:
                _sync(self.device)
                optimizer_time = time.perf_counter() - t0
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
                print(
                    f"[PROFILE rank=0 batch={batch_idx}]\n"
                    f"  actual_batch_size={actual_batch_size}\n"
                    f"  data_time      ={times['data']:.4f}s\n"
                    f"  to_device_time ={times['to_device']:.4f}s\n"
                    f"  forward_time   ={times['forward']:.4f}s\n"
                    f"  loss_time      ={times['loss']:.4f}s\n"
                    f"  backward_time  ={times['backward']:.4f}s\n"
                    f"  optimizer_time ={times['optimizer']:.4f}s\n"
                    f"  batch_time     ={times['batch']:.4f}s\n"
                    f"  cuda_allocated_gb    ={mem['cuda_allocated_gb']:.3f}\n"
                    f"  cuda_reserved_gb     ={mem['cuda_reserved_gb']:.3f}\n"
                    f"  cuda_max_allocated_gb={mem['cuda_max_allocated_gb']:.3f}"
                )
                for key, value in times.items():
                    profile_acc[key] += value
                profile_recorded += 1
                if not profile_avg_printed and (batch_idx == profile_n - 1 or is_last):
                    self._print_profile_average(profile_n, profile_recorded, profile_acc, full_epoch_batches)
                    profile_avg_printed = True

            pred = out["logits"].detach().argmax(dim=1).cpu()
            y_pred.extend(pred.tolist())
            y_true.extend(batch["y"].detach().cpu().tolist())
            pred_count += torch.bincount(pred, minlength=7)
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + _to_float(value)
            totals["grad_norm"] = totals.get("grad_norm", 0.0) + _to_float(grad_norm)
            D5Trainer._add_diagnostics(totals, out.get("diagnostics", {}))
            D5Trainer._add_output_diagnostics(totals, out)
            count += 1
            if self.is_rank0:
                progress.set_postfix(loss=f"{_to_float(loss):.4f}")
            t_data_start = time.perf_counter()

        if self.is_rank0 and profile_recorded > 0 and not profile_avg_printed:
            self._print_profile_average(profile_n, profile_recorded, profile_acc, full_epoch_batches)

        gathered_true: list[list[int]] = [None for _ in range(self.world_size)]  # type: ignore[list-item]
        gathered_pred: list[list[int]] = [None for _ in range(self.world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered_true, y_true)
        dist.all_gather_object(gathered_pred, y_pred)

        total_count = _all_reduce_float(float(count), self.device, op=dist.ReduceOp.SUM)
        metrics: Dict[str, float] = {}
        for key, value in totals.items():
            metrics[f"train_{key}"] = _all_reduce_float(value, self.device, op=dist.ReduceOp.SUM) / max(total_count, 1.0)
        skipped_total = _all_reduce_float(float(skipped_batches), self.device, op=dist.ReduceOp.SUM)
        metrics["train_skipped_nonfinite_grad_batches"] = float(skipped_total)

        flat_true = [item for part in gathered_true if part for item in part]
        flat_pred = [item for part in gathered_pred if part for item in part]
        if flat_true:
            metrics.update({f"train_{k}": float(v) for k, v in compute_metrics(flat_true, flat_pred).items()})
        metrics["train_batches"] = float(total_count)
        elapsed = time.perf_counter() - epoch_start
        elapsed_max = _all_reduce_float(elapsed, self.device, op=dist.ReduceOp.MAX)
        metrics["train_seconds"] = float(elapsed_max)
        metrics["train_sec_per_batch"] = float(elapsed_max / max(count, 1))
        cuda_max = _cuda_mem_stats(self.device)["cuda_max_allocated_gb"]
        metrics["ddp_cuda_max_allocated_gb_rankmax"] = _all_reduce_float(cuda_max, self.device, op=dist.ReduceOp.MAX)
        metrics["ddp_cuda_max_allocated_gb_rank0"] = float(cuda_max) if self.is_rank0 else 0.0
        pred_count_all = pred_count.to(self.device, dtype=torch.float64)
        dist.all_reduce(pred_count_all, op=dist.ReduceOp.SUM)
        for idx, value in enumerate(pred_count_all.detach().cpu().tolist()):
            metrics[f"train_pred_count_{idx}"] = float(value)
        return metrics

    def _print_profile_average(
        self,
        requested: int,
        recorded: int,
        acc: Dict[str, float],
        full_epoch_batches: Optional[int],
    ) -> None:
        avg = {key: value / max(recorded, 1) for key, value in acc.items()}
        est_epoch_min = None
        if full_epoch_batches is not None and avg.get("batch", 0.0) > 0:
            est_epoch_min = avg["batch"] * full_epoch_batches / 60.0
        est_str = f"{est_epoch_min:.2f}" if est_epoch_min is not None else "unknown"
        print(
            f"\n[PROFILE average rank=0 first {requested} batches (recorded={recorded})]\n"
            f"  avg_data_time      ={avg.get('data', 0.0):.4f}s\n"
            f"  avg_to_device_time ={avg.get('to_device', 0.0):.4f}s\n"
            f"  avg_forward_time   ={avg.get('forward', 0.0):.4f}s\n"
            f"  avg_loss_time      ={avg.get('loss', 0.0):.4f}s\n"
            f"  avg_backward_time  ={avg.get('backward', 0.0):.4f}s\n"
            f"  avg_optimizer_time ={avg.get('optimizer', 0.0):.4f}s\n"
            f"  avg_batch_time     ={avg.get('batch', 0.0):.4f}s\n"
            f"  estimated_full_epoch_minutes={est_str}"
        )

    @torch.no_grad()
    def validate_rank0(self, loader: DataLoader, max_batches: Optional[int], prefix: str = "val") -> Dict[str, float]:
        if not self.is_rank0:
            return {}
        model = self.raw_model
        model.eval()
        totals: Dict[str, float] = {}
        y_true = []
        y_pred = []
        y_pred_local = []
        count = 0
        pred_count = torch.zeros(7, dtype=torch.long)
        pred_count_local = torch.zeros(7, dtype=torch.long)
        start_time = time.perf_counter()
        for batch_idx, batch in enumerate(tqdm(loader, desc=prefix, leave=False)):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = move_to_device(batch, self.device)
            with _autocast(self.device, self.amp_enabled):
                out = model(batch)
                loss_dict = self.criterion(out, batch["y"], batch)
            logits = out["logits"]
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite logits during {prefix} at batch {batch_idx}")
            pred = logits.argmax(dim=1)
            y_true.extend(batch["y"].detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())
            pred_count += torch.bincount(pred.detach().cpu(), minlength=7)
            logits_local = out.get("logits_local")
            if torch.is_tensor(logits_local):
                if not torch.isfinite(logits_local).all():
                    raise FloatingPointError(f"Non-finite local logits during {prefix} at batch {batch_idx}")
                pred_local = logits_local.argmax(dim=1)
                y_pred_local.extend(pred_local.detach().cpu().tolist())
                pred_count_local += torch.bincount(pred_local.detach().cpu(), minlength=7)
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + _to_float(value)
            D5Trainer._add_diagnostics(totals, out.get("diagnostics", {}))
            D5Trainer._add_output_diagnostics(totals, out)
            count += 1

        metrics = {f"{prefix}_{key}": value / max(count, 1) for key, value in totals.items()}
        elapsed = time.perf_counter() - start_time
        metrics[f"{prefix}_seconds"] = float(elapsed)
        metrics[f"{prefix}_sec_per_batch"] = float(elapsed / max(count, 1))
        cls_metrics = compute_metrics(y_true, y_pred) if y_true else {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }
        metrics.update({f"{prefix}_{key}": float(value) for key, value in cls_metrics.items()})
        if y_pred_local and len(y_pred_local) == len(y_true):
            local_metrics = compute_metrics(y_true, y_pred_local)
            metrics[f"{prefix}_acc_local"] = float(local_metrics["accuracy"])
            metrics[f"{prefix}_accuracy_local"] = float(local_metrics["accuracy"])
            metrics[f"{prefix}_macro_f1_local"] = float(local_metrics["macro_f1"])
            metrics[f"{prefix}_weighted_f1_local"] = float(local_metrics["weighted_f1"])
        D5Trainer._add_selected_class_metrics(metrics, prefix, y_true, y_pred, pred_count)
        metrics[f"{prefix}_batches"] = float(count)
        for idx, value in enumerate(pred_count.tolist()):
            metrics[f"{prefix}_pred_count_{idx}"] = float(value)
        if y_pred_local and len(y_pred_local) == len(y_true):
            for idx, value in enumerate(pred_count_local.tolist()):
                metrics[f"{prefix}_pred_count_local_{idx}"] = float(value)
        return metrics

    def fit(
        self,
        *,
        train_loader: DataLoader,
        train_sampler: Any,
        val_loader: Optional[DataLoader],
        epochs: int,
        max_train_batches: Optional[int],
        max_val_batches: Optional[int],
    ) -> Dict[str, Any]:
        training_cfg = self.config.get("training", {}) or {}
        checkpoint_cfg = self.config.get("checkpoint", {}) or {}
        early_cfg = self.config.get("early_stopping", {}) or {}
        scheduler_cfg = self.config.get("scheduler", {}) or {}
        classification_monitor = training_cfg.get("monitor", "val_macro_f1")
        checkpoint_monitor = str(checkpoint_cfg.get("save_best_metric", classification_monitor))
        checkpoint_mode = _normalize_metric_mode(checkpoint_cfg.get("save_best_mode", checkpoint_cfg.get("mode", "max")))
        early_monitor = str(early_cfg.get("monitor", classification_monitor))
        early_mode = _normalize_metric_mode(early_cfg.get("mode", checkpoint_mode))
        early_patience = int(early_cfg.get("patience", training_cfg.get("early_stopping_patience", 20)))
        scheduler_uses_monitor = _scheduler_requires_monitor(self.scheduler)
        scheduler_monitor = None
        scheduler_mode = "max"
        if scheduler_uses_monitor:
            scheduler_monitor = str(scheduler_cfg.get("monitor", "val_loss"))
            scheduler_mode = _normalize_metric_mode(scheduler_cfg.get("mode", "min"))

        self.best_metric = _initial_metric_value(checkpoint_mode)
        best_early = _initial_metric_value(early_mode)
        stale_epochs = 0
        history = []
        if self.is_rank0:
            print(
                f"[Scheduler] type={_scheduler_name(self.scheduler)} "
                f"monitor={scheduler_monitor if scheduler_uses_monitor else 'epoch'} "
                f"mode={scheduler_mode if scheduler_uses_monitor else 'n/a'}"
            )
            print(f"[Checkpoint] save_best_metric={checkpoint_monitor} mode={checkpoint_mode}")
            print(f"[EarlyStopping] monitor={early_monitor} mode={early_mode}")

        for epoch in range(1, int(epochs) + 1):
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            full_epoch_batches = len(train_loader)
            lr_before = _optimizer_lr_metrics(self.optimizer)
            train_metrics = self.train_one_epoch(
                train_loader,
                epoch=epoch,
                max_batches=max_train_batches,
                full_epoch_batches=full_epoch_batches,
            )

            dist.barrier()
            val_metrics = self.validate_rank0(val_loader, max_val_batches, prefix="val") if self.is_rank0 else {}
            control = {
                "should_stop": False,
                "monitor_value": None,
                "checkpoint_value": None,
                "early_value": None,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
            }
            if self.is_rank0:
                metrics = {"epoch": float(epoch), **train_metrics, **val_metrics}
                metrics.update(lr_before)
                monitor_value = _get_metric_value(metrics, scheduler_monitor) if scheduler_uses_monitor else None
                if scheduler_uses_monitor:
                    metrics["scheduler_monitor_value"] = float(monitor_value)
                    metrics[f"scheduler_monitor_{scheduler_monitor}"] = float(monitor_value)
                checkpoint_value = _get_metric_value(metrics, checkpoint_monitor)
                early_value = _get_metric_value(metrics, early_monitor)
                metrics["checkpoint_monitor_value"] = checkpoint_value
                metrics[f"checkpoint_monitor_{checkpoint_monitor}"] = checkpoint_value
                metrics["early_stopping_monitor_value"] = early_value
                metrics[f"early_stopping_monitor_{early_monitor}"] = early_value
            else:
                metrics = {}
                monitor_value = None
                checkpoint_value = None
                early_value = None

            control = [control]
            if self.is_rank0:
                control[0].update(
                    {
                        "monitor_value": monitor_value,
                        "checkpoint_value": checkpoint_value,
                        "early_value": early_value,
                    }
                )
            dist.broadcast_object_list(control, src=0)
            monitor_value = control[0]["monitor_value"]
            step_scheduler(self.scheduler, monitor_value)
            lr_after = _optimizer_lr_metrics(self.optimizer)

            if self.is_rank0:
                if lr_after:
                    metrics["lr_after_scheduler"] = lr_after.get("lr", 0.0)
                    metrics["lr_min_after_scheduler"] = lr_after.get("lr_min", 0.0)
                    metrics["lr_max_after_scheduler"] = lr_after.get("lr_max", 0.0)
                checkpoint_improved = _is_improved(float(checkpoint_value), self.best_metric, checkpoint_mode)
                early_improved = _is_improved(float(early_value), best_early, early_mode)
                if checkpoint_improved:
                    self.best_metric = float(checkpoint_value)
                    self.best_epoch = epoch
                    self.save_checkpoint("best.pth", epoch, metrics)
                if early_improved:
                    best_early = float(early_value)
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
                    f"ckpt_monitor={checkpoint_monitor}:{float(checkpoint_value):.4f} "
                    f"best_{checkpoint_monitor}={self.best_metric:.4f} "
                    f"best_epoch={self.best_epoch} "
                    f"early_{early_monitor}={best_early:.4f} "
                    f"sec/batch={metrics.get('train_sec_per_batch', 0.0):.3f}s "
                    f"cuda_rankmax={metrics.get('ddp_cuda_max_allocated_gb_rankmax', 0.0):.3f}GB"
                )
                print(
                    "          val pred_count: "
                    f"{[int(metrics.get(f'val_pred_count_{i}', 0.0)) for i in range(7)]}"
                )
                if checkpoint_improved:
                    print(f"          checkpoint improvement: best_epoch={self.best_epoch}")
                if early_improved:
                    print("          early stopping monitor improved")
                else:
                    print(f"          no improvement: {stale_epochs}/{early_patience}")
                should_stop = stale_epochs >= early_patience
                if should_stop:
                    print(f"Early stopping after {stale_epochs} stale epochs")
                control[0].update(
                    {
                        "should_stop": should_stop,
                        "best_metric": self.best_metric,
                        "best_epoch": self.best_epoch,
                    }
                )

            dist.broadcast_object_list(control, src=0)
            self.best_metric = float(control[0]["best_metric"])
            self.best_epoch = int(control[0]["best_epoch"])
            dist.barrier()
            if bool(control[0]["should_stop"]):
                break

        if self.is_rank0:
            self._save_history(history)
            if self.wandb is not None:
                self.wandb.finish()
        return {"best_metric": self.best_metric, "best_epoch": self.best_epoch, "history": history}

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]) -> Path:
        path = self.checkpoint_dir / filename
        model_to_save = _unwrap_model(self.model)
        checkpoint_monitor = self.config.get("checkpoint", {}).get(
            "save_best_metric",
            self.config.get("training", {}).get("monitor", "val_macro_f1"),
        )
        checkpoint_mode = self.config.get("checkpoint", {}).get("save_best_mode", "max")
        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_metric": float(self.best_metric),
            "best_epoch": int(self.best_epoch),
            "best_metric_name": checkpoint_monitor,
            "best_metric_mode": checkpoint_mode,
            "config": self.config,
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(checkpoint, path)
        return path

    def _save_history(self, history) -> None:
        path = self.output_root / "training_history.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        if self.wandb is not None:
            self.wandb.log(metrics)

    def _compute_grad_norm(self) -> torch.Tensor:
        norms = [
            p.grad.detach().norm(2)
            for p in self.model.parameters()
            if p.grad is not None
        ]
        if not norms:
            return torch.zeros((), device=self.device)
        return torch.norm(torch.stack(norms), p=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/d12a_stable_ce_first_ddp_b64_amp.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--global_batch_size", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--ddp_chunk_aware", dest="ddp_chunk_aware", action="store_true")
    parser.add_argument("--no_ddp_chunk_aware", dest="ddp_chunk_aware", action="store_false")
    parser.set_defaults(ddp_chunk_aware=None)
    parser.add_argument("--find_unused_parameters", choices=["true", "false"], default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true", default=False)
    parser.add_argument("--use_compile", action="store_true", default=False)
    parser.add_argument("--compile_order", choices=["before_ddp", "after_ddp"], default=None)
    parser.add_argument("--no_compile", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup_ddp()
    try:
        config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
        if args.graph_repo_path is not None:
            _set_nested(config, "paths", "graph_repo_path", args.graph_repo_path)
        if args.max_train_batches is not None:
            _set_nested(config, "training", "max_train_batches", int(args.max_train_batches))
        if args.max_val_batches is not None:
            _set_nested(config, "training", "max_val_batches", int(args.max_val_batches))
        if args.amp:
            _set_nested(config, "training", "amp", True)
        if args.no_amp:
            _set_nested(config, "training", "amp", False)
        if args.use_compile:
            _set_nested(config, "training", "use_compile", True)
        if args.compile_order is not None:
            _set_nested(config, "training", "compile_order", args.compile_order)
        if args.no_compile:
            _set_nested(config, "training", "use_compile", False)

        global_batch_size = resolve_global_batch_size(config, args.global_batch_size)
        if global_batch_size % world_size != 0:
            raise ValueError(
                f"DDP requires global_batch_size divisible by world_size: "
                f"global_batch_size={global_batch_size}, world_size={world_size}"
            )
        per_rank_batch_size = global_batch_size // world_size
        config = apply_ddp_runtime_overrides(
            config,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            global_batch_size=global_batch_size,
            per_rank_batch_size=per_rank_batch_size,
            device=device,
            args=args,
        )

        if rank == 0:
            output_root = make_run_output_root(config)
            config.setdefault("paths", {})["resolved_output_root"] = str(output_root)
            save_config(config, output_root)
            print(f"[Output] run_dir={output_root}")
        else:
            output_root = None
        output_root = Path(_broadcast_string(str(output_root) if output_root is not None else None, src=0))
        config.setdefault("paths", {})["resolved_output_root"] = str(output_root)

        if rank == 0:
            find_unused_parameters = _cfg_bool(
                config.get("ddp", {}).get("find_unused_parameters", True),
                True,
            )
            print(
                "[DDP Setup] "
                f"rank={rank} local_rank={local_rank} world_size={world_size} "
                f"device={device} backend=nccl find_unused_parameters={find_unused_parameters}"
            )
            print(
                "[DDP Batch] "
                f"global_batch_size={global_batch_size} "
                f"per_rank_batch_size={per_rank_batch_size} "
                f"world_size={world_size}"
            )
            fixed_batch_size = _cfg_bool(config.get("data", {}).get("fixed_batch_size", False), False)
            print(
                f"[DDP Phase{'1.6' if fixed_batch_size else '1.5'}] "
                f"ddp_chunk_aware={_cfg_bool(config.get('data', {}).get('ddp_chunk_aware', True), True)} "
                f"fixed_batch_size={fixed_batch_size} "
                f"torch.compile={_cfg_bool(config.get('training', {}).get('use_compile', False), False)}"
            )

        seed = int(config.get("training", {}).get("seed", config.get("run", {}).get("seed", 42)))
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)

        train_loader, train_sampler = build_ddp_dataloader(
            config,
            split="train",
            batch_size=per_rank_batch_size,
            rank=rank,
            world_size=world_size,
            distributed_train=True,
        )
        val_loader = None
        if rank == 0:
            val_loader, _ = build_ddp_dataloader(
                config,
                split="val",
                batch_size=global_batch_size,
                rank=0,
                world_size=1,
                distributed_train=False,
            )

        with _quiet_non_rank0():
            model, criterion, optimizer, scheduler, prepared_device = prepare_training_objects(config)
        if prepared_device != device:
            raise RuntimeError(f"Prepared device {prepared_device} does not match DDP device {device}")
        compile_enabled = _cfg_bool(config.get("training", {}).get("use_compile", False), False)
        compile_order = str(config.get("training", {}).get("compile_order", "before_ddp"))
        if compile_enabled and compile_order == "before_ddp":
            model = _compile_inner_model(model)
            if rank == 0:
                print("[Compile] enabled=True order=before_ddp")
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=_cfg_bool(
                config.get("ddp", {}).get("find_unused_parameters", True),
                True,
            ),
        )
        if compile_enabled and compile_order == "after_ddp":
            if hasattr(torch, "compile"):
                ddp_model = torch.compile(ddp_model)
                if rank == 0:
                    print("[Compile] enabled=True order=after_ddp")
            elif rank == 0:
                print("[Compile] requested=True order=after_ddp available=False")
        elif not compile_enabled and rank == 0:
            print("[Compile] enabled=False order=none")
        if rank == 0:
            print(
                "[DDP Model] DistributedDataParallel wrapped with "
                f"find_unused_parameters={_cfg_bool(config.get('ddp', {}).get('find_unused_parameters', True), True)}"
            )

        trainer = DDPPhase1Trainer(
            model=ddp_model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            output_root=output_root,
        )
        result = trainer.fit(
            train_loader=train_loader,
            train_sampler=train_sampler,
            val_loader=val_loader,
            epochs=int(config.get("training", {}).get("epochs", 15)),
            max_train_batches=config.get("training", {}).get("max_train_batches"),
            max_val_batches=config.get("training", {}).get("max_val_batches"),
        )
        if rank == 0:
            print(f"Training done best_epoch={result['best_epoch']} best_metric={result['best_metric']:.6f}")
            try:
                log_experiment(output_root)
            except Exception as exc:
                print(f"Failed to log experiment: {exc}")
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
