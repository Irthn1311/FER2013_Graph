"""Persist runtime and sampler diagnostics for parity audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _hist_to_list(hist: Dict[int, int], num_classes: int = 7) -> List[int]:
    return [int(hist.get(i, 0)) for i in range(num_classes)]


def _dataset_label(dataset, idx: int) -> Optional[int]:
    if hasattr(dataset, "label_at_index"):
        return int(dataset.label_at_index(int(idx)))
    nested = getattr(dataset, "dataset", None)
    if nested is not None and hasattr(nested, "label_at_index"):
        return int(nested.label_at_index(int(idx)))
    return None


def _batch_indices_from_loader(loader) -> Iterable[List[int]]:
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is None:
        return []
    return batch_sampler


def _summarize_index_batches(loader) -> Dict[str, Any]:
    dataset = getattr(loader, "dataset", None)
    label_hist: Dict[int, int] = {}
    batch_sizes: List[int] = []
    yielded = 0
    label_available = dataset is not None and (
        hasattr(dataset, "label_at_index")
        or hasattr(getattr(dataset, "dataset", None), "label_at_index")
    )
    if dataset is None:
        return {
            "number_of_train_batches_per_rank": None,
            "unique_batch_sizes_per_rank": [],
            "number_of_samples_yielded_by_sampler": None,
            "per_rank_sampled_label_histogram": None,
            "label_histogram_available": False,
        }

    for batch in _batch_indices_from_loader(loader):
        indices = [int(idx) for idx in batch]
        batch_sizes.append(len(indices))
        yielded += len(indices)
        if label_available:
            for idx in indices:
                label = _dataset_label(dataset, idx)
                if label is not None:
                    label_hist[label] = label_hist.get(label, 0) + 1

    return {
        "number_of_train_batches_per_rank": len(batch_sizes),
        "unique_batch_sizes_per_rank": sorted(set(batch_sizes)),
        "number_of_samples_yielded_by_sampler": int(yielded),
        "per_rank_sampled_label_histogram": _hist_to_list(label_hist) if label_available else None,
        "label_histogram_available": bool(label_available),
    }


def _infer_runtime_mode(config: Dict[str, Any]) -> str:
    parity_cfg = config.get("runtime_parity", {}) or {}
    if parity_cfg.get("mode"):
        return str(parity_cfg["mode"])
    ddp_cfg = config.get("ddp", {}) or {}
    training_cfg = config.get("training", {}) or {}
    if _cfg_bool(ddp_cfg.get("enabled", False), False):
        amp = _cfg_bool(training_cfg.get("amp", False), False)
        compile_enabled = _cfg_bool(training_cfg.get("use_compile", training_cfg.get("torch_compile", False)), False)
        if amp and compile_enabled:
            return "ddp_amp_compile"
        if amp:
            return "ddp_amp"
        return "ddp_eager"
    if _cfg_bool(training_cfg.get("multi_gpu", False), False):
        return "legacy_dp"
    return "single_process"


def _maybe_all_reduce_hist(hist: Optional[List[int]], device: Optional[torch.device]) -> Optional[List[int]]:
    if hist is None:
        return None
    if not (dist.is_available() and dist.is_initialized()):
        return hist
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor(hist, dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [int(v) for v in tensor.detach().cpu().tolist()]


def write_sampler_diagnostics(
    *,
    config: Dict[str, Any],
    output_root: str | Path,
    train_loader,
    sampler: Any = None,
    rank: int = 0,
    world_size: int = 1,
    device: Optional[torch.device] = None,
    filename: Optional[str] = None,
) -> Path:
    """Write sampler/runtime diagnostics without touching model training."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_cfg = config.get("data", {}) or {}
    training_cfg = config.get("training", {}) or {}
    ddp_cfg = config.get("ddp", {}) or {}
    runtime_mode = _infer_runtime_mode(config)
    dataset = getattr(train_loader, "dataset", None)
    dataset_len = len(dataset) if dataset is not None and hasattr(dataset, "__len__") else None
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(1)

    batch_summary = _summarize_index_batches(train_loader)
    sampler_summary = None
    if sampler is not None and hasattr(sampler, "summary"):
        try:
            sampler_summary = sampler.summary(epoch=1)
        except TypeError:
            sampler_summary = sampler.summary()

    per_rank_hist = batch_summary.get("per_rank_sampled_label_histogram")
    global_hist = _maybe_all_reduce_hist(per_rank_hist, device=device)
    yielded = batch_summary.get("number_of_samples_yielded_by_sampler")
    global_yielded = None
    if yielded is not None:
        if dist.is_available() and dist.is_initialized():
            reduce_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            yielded_tensor = torch.tensor(int(yielded), dtype=torch.long, device=reduce_device)
            dist.all_reduce(yielded_tensor, op=dist.ReduceOp.SUM)
            global_yielded = int(yielded_tensor.detach().cpu().item())
        else:
            global_yielded = int(yielded)

    diagnostics: Dict[str, Any] = {
        "runtime_mode": runtime_mode,
        "world_size": int(world_size),
        "rank": int(rank),
        "global_batch_size": int(training_cfg.get("global_batch_size", training_cfg.get("batch_size", data_cfg.get("batch_size", 0))) or 0),
        "per_rank_batch_size": int(training_cfg.get("per_rank_batch_size", data_cfg.get("batch_size", 0)) or 0),
        "data_batch_size": int(data_cfg.get("batch_size", 0) or 0),
        "training_amp": _cfg_bool(training_cfg.get("amp", False), False),
        "training_multi_gpu": _cfg_bool(training_cfg.get("multi_gpu", False), False),
        "training_use_compile": _cfg_bool(training_cfg.get("use_compile", False), False),
        "training_torch_compile": _cfg_bool(training_cfg.get("torch_compile", training_cfg.get("use_compile", False)), False),
        "compile_order": training_cfg.get("compile_order", ddp_cfg.get("compile_order")),
        "ddp_enabled": _cfg_bool(ddp_cfg.get("enabled", False), False),
        "fixed_batch_size": _cfg_bool(data_cfg.get("fixed_batch_size", False), False),
        "drop_incomplete_batches": _cfg_bool(data_cfg.get("drop_incomplete_batches", False), False),
        "carry_over_leftovers": _cfg_bool(data_cfg.get("carry_over_leftovers", False), False),
        "ddp_drop_last_batches": _cfg_bool(data_cfg.get("ddp_drop_last_batches", False), False),
        "ddp_chunk_aware": _cfg_bool(data_cfg.get("ddp_chunk_aware", False), False),
        "chunk_aware_shuffle": _cfg_bool(data_cfg.get("chunk_aware_shuffle", False), False),
        "train_dataset_length_before_sampler": dataset_len,
        "number_of_train_batches_per_rank": batch_summary.get("number_of_train_batches_per_rank"),
        "unique_batch_sizes_per_rank": batch_summary.get("unique_batch_sizes_per_rank"),
        "number_of_samples_yielded_by_sampler": yielded,
        "global_number_of_samples_yielded_by_sampler": global_yielded,
        "dropped_sample_count_per_rank": (
            max(int(dataset_len) - int(yielded), 0)
            if dataset_len is not None and yielded is not None and int(world_size) <= 1
            else None
        ),
        "carry_over_leftover_count_per_rank": None,
        "per_rank_sampled_label_histogram": per_rank_hist,
        "global_sampled_label_histogram": global_hist if int(rank) == 0 else None,
        "class_0_exposure_count": int(per_rank_hist[0]) if per_rank_hist else None,
        "class_1_exposure_count": int(per_rank_hist[1]) if per_rank_hist else None,
        "global_class_0_exposure_count": int(global_hist[0]) if global_hist and int(rank) == 0 else None,
        "global_class_1_exposure_count": int(global_hist[1]) if global_hist and int(rank) == 0 else None,
        "sampler_type": type(sampler).__name__ if sampler is not None else type(getattr(train_loader, "sampler", None)).__name__,
        "batch_sampler_type": type(getattr(train_loader, "batch_sampler", None)).__name__,
        "sampler_summary": sampler_summary,
    }

    if sampler_summary:
        diagnostics["dropped_sample_count_per_rank"] = sampler_summary.get("dropped_samples_per_rank")
        diagnostics["unique_batch_sizes_per_rank_all"] = sampler_summary.get("unique_batch_sizes_per_rank")
        diagnostics["number_of_train_batches_all_ranks"] = sampler_summary.get("batches_after_balance")
        diagnostics["target_class_repeat_factors"] = sampler_summary.get("target_class_repeat_factors")
        diagnostics["expanded_sample_count"] = sampler_summary.get("expanded_total_samples")
        diagnostics["repeated_sample_count"] = sampler_summary.get("repeated_num_indices_total")

    out_name = filename or ("sampler_diagnostics.json" if int(world_size) <= 1 else f"sampler_diagnostics_rank{rank}.json")
    path = output_root / out_name
    with path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    if int(rank) == 0 and int(world_size) > 1:
        rank0_path = output_root / "sampler_diagnostics.json"
        with rank0_path.open("w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        return rank0_path
    return path
