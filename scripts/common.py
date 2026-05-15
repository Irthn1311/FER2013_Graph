"""Shared helpers for D5 command-line scripts."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from data.full_graph_dataset import ChunkAwareBatchSampler, FullGraphDataset, collate_fn_full_graph
from data.graph_config import GraphConfig
from models.registry import build_model
from training.losses import build_loss
from training.optimizer import build_optimizer, build_scheduler
from training.trainer import D5Trainer, set_seed


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path, environment: str | None = None) -> Dict[str, Any]:
    path = resolve_existing_path(config_path)
    cfg = _load_config_tree(path)
    cfg = resolve_environment_config(cfg, environment=environment)
    run_cfg = dict(cfg.get("run", {}))
    run_cfg.setdefault("config_name", path.stem)
    cfg["run"] = run_cfg
    return cfg


def _load_config_tree(config_path: str | Path, seen: Optional[set[Path]] = None) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"Circular config inheritance detected at: {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    inherited = cfg.pop("inherits", [])
    if isinstance(inherited, (str, Path)):
        inherited = [inherited]

    merged: Dict[str, Any] = {}
    for parent in inherited:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = deep_update(merged, _load_config_tree(parent_path, seen))
    seen.remove(path)
    return deep_update(merged, cfg)


def resolve_environment_config(config: Dict[str, Any], environment: str | None = None) -> Dict[str, Any]:
    cfg = dict(config)
    env = environment or cfg.get("environment") or os.environ.get("D5_ENV")
    if not env:
        return cfg
    env = str(env).lower()
    profiles = cfg.get("environments", {})
    if profiles and env not in profiles:
        raise ValueError(f"Unknown environment={env!r}. Available: {sorted(profiles)}")
    if profiles:
        cfg = deep_update(cfg, profiles.get(env, {}) or {})
    cfg["environment"] = env
    return cfg


def save_config(config: Dict[str, Any], output_root: str | Path) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def make_run_output_root(config: Dict[str, Any]) -> Path:
    paths = config.get("paths", {})
    run_cfg = config.get("run", {})
    base = resolve_path(paths.get("output_root", "outputs"))
    config_name = str(run_cfg.get("config_name") or "run").replace("\\", "_").replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(base) / config_name / timestamp
    suffix = 2
    while root.exists():
        root = Path(base) / config_name / f"{timestamp}_{suffix:02d}"
        suffix += 1
    return root


def output_root_from_checkpoint(checkpoint_path: str | Path) -> Optional[Path]:
    path = Path(checkpoint_path)
    parts = path.parts
    if "checkpoints" in parts:
        return path.parents[len(parts) - 1 - parts.index("checkpoints")]
    if path.parent.name == "checkpoints":
        return path.parent.parent
    return None


def resolve_existing_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Path not found: {path_like}")


def resolve_path(path_like: str | Path | None, default: Optional[str] = None) -> Optional[Path]:
    if path_like is None:
        path_like = default
    if path_like is None:
        return None
    value = str(path_like)
    if value.lower() == "auto":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_csv_root(csv_root: str | Path | None = "auto") -> Path:
    if csv_root is not None and str(csv_root).lower() != "auto":
        path = Path(csv_root)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if _has_split_csvs(path):
            return path
        raise FileNotFoundError(f"CSV root does not contain train/val/test CSVs: {path}")

    candidates = [
        Path.cwd(),
        Path.cwd() / "data",
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "data",
    ]
    for candidate in candidates:
        if _has_split_csvs(candidate):
            return candidate.resolve()

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for train_csv in kaggle_input.rglob("train.csv"):
            candidate = train_csv.parent
            if _has_split_csvs(candidate):
                return candidate.resolve()

    for train_csv in PROJECT_ROOT.parent.rglob("train.csv"):
        candidate = train_csv.parent
        if _has_split_csvs(candidate):
            return candidate.resolve()
    raise FileNotFoundError("Could not auto-find train.csv, val.csv, and test.csv")


def _has_split_csvs(path: Path) -> bool:
    return all((path / f"{split}.csv").exists() for split in ("train", "val", "test"))


def split_csv_paths(csv_root: str | Path | None = "auto") -> Dict[str, Path]:
    root = find_csv_root(csv_root)
    return {split: root / f"{split}.csv" for split in ("train", "val", "test")}


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(config)
    environment = getattr(args, "environment", None)
    if environment is not None:
        cfg = resolve_environment_config(cfg, environment=environment)
    paths = dict(cfg.get("paths", {}))
    data = dict(cfg.get("data", {}))
    training = dict(cfg.get("training", {}))
    logging_cfg = dict(cfg.get("logging", {}))

    for attr in ("csv_root", "graph_repo_path", "output_root"):
        value = getattr(args, attr, None)
        if value is not None:
            paths[attr] = value
            if attr == "output_root":
                paths["resolved_output_root"] = value
    if getattr(args, "batch_size", None) is not None:
        data["batch_size"] = int(args.batch_size)
        training["batch_size"] = int(args.batch_size)
    if getattr(args, "epochs", None) is not None:
        training["epochs"] = int(args.epochs)
    if getattr(args, "device", None) is not None:
        training["device"] = str(args.device)
    for attr in ("max_train_batches", "max_val_batches", "max_test_batches"):
        value = getattr(args, attr, None)
        if value is not None:
            training[attr] = int(value)
    if getattr(args, "no_wandb", False):
        logging_cfg["use_wandb"] = False
    if getattr(args, "wandb", False):
        logging_cfg["use_wandb"] = True
    if getattr(args, "wandb_project", None) is not None:
        logging_cfg["project"] = str(args.wandb_project)
    if getattr(args, "wandb_entity", None) is not None:
        logging_cfg["entity"] = str(args.wandb_entity)

    # --- DataLoader overrides ---
    if getattr(args, "num_workers", None) is not None:
        data["num_workers"] = int(args.num_workers)
        training["num_workers"] = int(args.num_workers)
    # pin_memory: CLI passes string "true"/"false" or bool
    _pin = getattr(args, "pin_memory", None)
    if _pin is not None:
        if isinstance(_pin, str):
            data["pin_memory"] = _pin.strip().lower() in ("1", "true", "yes")
        else:
            data["pin_memory"] = bool(_pin)
        training["pin_memory"] = data["pin_memory"]
    _pw = getattr(args, "persistent_workers", None)
    if _pw is not None:
        if isinstance(_pw, str):
            data["persistent_workers"] = _pw.strip().lower() in ("1", "true", "yes")
        else:
            data["persistent_workers"] = bool(_pw)
        training["persistent_workers"] = data["persistent_workers"]
    if getattr(args, "prefetch_factor", None) is not None:
        data["prefetch_factor"] = int(args.prefetch_factor)
        training["prefetch_factor"] = int(args.prefetch_factor)
    if getattr(args, "chunk_cache_size", None) is not None:
        data["chunk_cache_size"] = int(args.chunk_cache_size)
        data.pop("graph_cache_chunks", None)
    elif getattr(args, "graph_cache_chunks", None) is not None:
        data["chunk_cache_size"] = int(args.graph_cache_chunks)
        data.pop("graph_cache_chunks", None)
    if getattr(args, "chunk_aware_shuffle", False):
        data["chunk_aware_shuffle"] = True
    if getattr(args, "no_chunk_aware_shuffle", False):
        data["chunk_aware_shuffle"] = False

    # --- Training performance overrides ---
    if getattr(args, "profile_batches", None) is not None:
        training["profile_batches"] = int(args.profile_batches)
    # AMP: --amp sets True, --no_amp sets False
    if getattr(args, "amp", False):
        training["amp"] = True
    if getattr(args, "no_amp", False):
        training["amp"] = False

    cfg["paths"] = paths
    cfg["data"] = data
    cfg["training"] = training
    cfg["logging"] = logging_cfg
    return cfg


def build_dataloader(
    config: Dict[str, Any],
    split: str,
    shuffle: bool = False,
) -> DataLoader:
    paths = config.get("paths", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    repo = resolve_path(paths.get("graph_repo_path", "artifacts/graph_repo"))

    def _cfg_value(key: str, default: Any = None) -> Any:
        if key in data_cfg:
            return data_cfg.get(key)
        return training_cfg.get(key, default)

    def _cfg_bool(key: str, default: bool = False) -> bool:
        value = _cfg_value(key, default)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(value)

    chunk_cache_size = int(data_cfg.get("chunk_cache_size", data_cfg.get("graph_cache_chunks", 0)) or 0)
    graph_cache_chunks = data_cfg.get("graph_cache_chunks", None)
    if graph_cache_chunks is not None and chunk_cache_size <= 0:
        chunk_cache_size = int(graph_cache_chunks or 0)
    dataset = FullGraphDataset(
        repo_root=repo,
        split=split,
        chunk_cache_size=chunk_cache_size,
        graph_cache_chunks=int(graph_cache_chunks) if graph_cache_chunks is not None else None,
    )
    num_workers = int(_cfg_value("num_workers", 0) or 0)
    pin_memory = _cfg_bool("pin_memory", False)
    persistent_workers_cfg = _cfg_bool("persistent_workers", False)
    persistent_workers = persistent_workers_cfg and num_workers > 0
    if persistent_workers_cfg and num_workers == 0:
        print("[DataLoader] WARNING: persistent_workers=True ignored because num_workers=0")
    prefetch_factor_cfg = _cfg_value("prefetch_factor", None)
    prefetch_factor = int(prefetch_factor_cfg) if (prefetch_factor_cfg is not None and num_workers > 0) else None
    if prefetch_factor_cfg is not None and num_workers == 0:
        print("[DataLoader] WARNING: prefetch_factor ignored because num_workers=0")
    batch_size = int(data_cfg.get("batch_size", training_cfg.get("batch_size", 16)) or 16)
    chunk_aware_shuffle = _cfg_bool("chunk_aware_shuffle", False)
    shuffle_chunks = _cfg_bool("shuffle_chunks", True)
    shuffle_within_chunk = _cfg_bool("shuffle_within_chunk", True)
    use_chunk_aware_sampler = bool(split == "train" and shuffle and chunk_aware_shuffle)
    loader_kwargs: Dict[str, Any] = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn_full_graph,
    )
    if use_chunk_aware_sampler:
        loader_kwargs["batch_sampler"] = ChunkAwareBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle_chunks=shuffle_chunks,
            shuffle_within_chunk=shuffle_within_chunk,
        )
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = bool(shuffle)
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    batches_per_epoch = len(loader_kwargs["batch_sampler"]) if use_chunk_aware_sampler else math.ceil(len(dataset) / batch_size)
    print(
        f"[DataLoader split={split}] "
        f"num_samples={len(dataset)} "
        f"batch_size={batch_size} "
        f"num_workers={num_workers} "
        f"pin_memory={pin_memory} "
        f"persistent_workers={persistent_workers} "
        f"prefetch_factor={prefetch_factor} "
        f"chunk_cache_size={chunk_cache_size} "
        f"chunk_aware_sampler={use_chunk_aware_sampler} "
        f"shuffle_chunks={shuffle_chunks if use_chunk_aware_sampler else 'n/a'} "
        f"shuffle_within_chunk={shuffle_within_chunk if use_chunk_aware_sampler else 'n/a'} "
        f"batches_per_epoch={batches_per_epoch}"
    )
    return DataLoader(dataset, **loader_kwargs)


def resolve_device(device_arg: str | None = None, config: Optional[Dict[str, Any]] = None) -> torch.device:
    config = config or {}
    requested = (
        device_arg
        or config.get("training", {}).get("device")
        or config.get("device")
        or "auto"
    )
    requested = "auto" if requested is None else str(requested).strip()
    if requested == "" or requested.lower() == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif requested.lower().startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        device = torch.device("cuda:0" if requested.lower() == "cuda" else requested)
    else:
        device = torch.device(requested)

    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def log_device_info(device: torch.device | str) -> None:
    device = torch.device(device)
    print(f"--- torch version: {torch.__version__}")
    print(f"--- cuda available: {torch.cuda.is_available()}")
    print(f"--- cuda device count: {torch.cuda.device_count()}")
    print(f"--- selected device: {device}")
    if torch.cuda.is_available():
        idx = device.index if device.type == "cuda" and device.index is not None else torch.cuda.current_device()
        print(f"--- gpu name: {torch.cuda.get_device_name(idx)}")
        print(f"--- current cuda device: {torch.cuda.current_device()}")


def infer_device(config: Dict[str, Any]) -> torch.device:
    return resolve_device(config=config)


def prepare_training_objects(config: Dict[str, Any]):
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = infer_device(config)
    log_device_info(device)
    graph_cfg = GraphConfig.from_dict(config.get("graph", {}))
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("height", graph_cfg.height)
    model_cfg.setdefault("width", graph_cfg.width)
    model_cfg.setdefault("connectivity", graph_cfg.connectivity)
    model = build_model(model_cfg).to(device)
    loss_cfg = dict(config.get("loss", {}))
    if config.get("attention_regularization") is not None:
        loss_cfg["attention_regularization"] = config.get("attention_regularization")
    loss_cfg.setdefault("height", graph_cfg.height)
    loss_cfg.setdefault("width", graph_cfg.width)
    criterion = build_loss(loss_cfg).to(device)
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}))
    return model, criterion, optimizer, scheduler, device


def create_trainer(config: Dict[str, Any]) -> D5Trainer:
    model, criterion, optimizer, scheduler, device = prepare_training_objects(config)
    paths = config.get("paths", {})
    logging_cfg = config.get("logging", {})
    training_cfg = config.get("training", {})
    output_root = resolve_path(paths.get("resolved_output_root")) or make_run_output_root(config)
    config.setdefault("paths", {})["resolved_output_root"] = str(output_root)
    save_config(config, output_root)
    print(f"[Output] run_dir={output_root}")
    return D5Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_root=output_root,
        config=config,
        use_wandb=bool(logging_cfg.get("use_wandb", False)),
        wandb_project=logging_cfg.get("project"),
        wandb_entity=logging_cfg.get("entity"),
        wandb_run_name=logging_cfg.get("run_name"),
        grad_clip_norm=training_cfg.get("grad_clip_norm", 5.0),
        amp=bool(training_cfg.get("amp", False)),
        amp_init_scale=float(training_cfg.get("amp_init_scale", 65536.0)),
        profile_batches=int(training_cfg.get("profile_batches", 0)),
        multi_gpu=bool(training_cfg.get("multi_gpu", False)),
        use_compile=bool(training_cfg.get("use_compile", False)),
        val_frequency=int(training_cfg.get("val_frequency", 1)),
    )


def load_checkpoint_model(config: Dict[str, Any], checkpoint_path: str | Path):
    model, _, _, _, device = prepare_training_objects(config)
    ckpt_path = resolve_existing_path(checkpoint_path)
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device, checkpoint


def dump_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        return str(value)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=default)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
