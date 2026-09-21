"""Atomic, hash-verified, full-state resume bundles for MPG-FER v2.1."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
import torch

from .config import MPGConfig


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_config(config: MPGConfig, include_runtime_safe: bool = False) -> dict:
    data = asdict(config)
    safe = set(config.runtime_safe_resume_fields) | {"runtime_safe_resume_fields"}
    return data if include_runtime_safe else {k: v for k, v in data.items() if k not in safe}


def config_hash(config: MPGConfig) -> str:
    payload = json.dumps(canonical_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture_rng_state(loader_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dataloader_generator": loader_generator.get_state(),
    }


def restore_rng_state(state: dict, loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    loader_generator.set_state(state["dataloader_generator"])


def build_resume_bundle(
    *, config: MPGConfig, run_id: str, source_hash: str, completed_epoch: int,
    global_optimizer_step: int, model: torch.nn.Module, ema: Any,
    optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any,
    best_comparator_state: dict | None, best_epoch: int | None,
    best_metrics: dict | None, early_stop_counter: int, history: list[dict],
    loader_generator: torch.Generator, consistency_state: dict,
    sampler_state: dict | None = None,
    augmentation_state: dict | None = None,
) -> dict:
    temperature_state = None
    if hasattr(model, "motif_composer"):
        temperature_state = {
            "epoch": completed_epoch,
            "current_tau": float(model.motif_composer.temperature),
        }
    return {
        "resume_schema_version": config.resume_schema_version,
        "run_id": run_id,
        "source_hash": source_hash,
        "config_hash": config_hash(config),
        "scientific_config": canonical_config(config),
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "global_optimizer_step": global_optimizer_step,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict() if scaler is not None else {},
        "best_comparator_state": best_comparator_state,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "early_stop_counter": early_stop_counter,
        "early_stop_monitor_state": {
            "start_epoch": config.early_stop_monitor_start_epoch,
            "active": completed_epoch >= config.early_stop_monitor_start_epoch,
            "patience": early_stop_counter,
        },
        "temperature_state": temperature_state,
        "history": history,
        "rng_state": capture_rng_state(loader_generator),
        "consistency_state": consistency_state,
        "sampler_state": sampler_state,
        "augmentation_state": augmentation_state,
    }


def atomic_save_resume(bundle: dict, output_dir: str | Path, status: str = "TRAINING") -> tuple[Path, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "resume_latest.pt"
    temporary = output / "resume_latest.tmp"
    with temporary.open("wb") as handle:
        torch.save(bundle, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    digest = sha256_file(target)
    metadata = {
        "run_id": bundle["run_id"],
        "epoch": bundle["completed_epoch"],
        "next_epoch": bundle["next_epoch"],
        "checkpoint_sha256": digest,
        "source_hash": bundle["source_hash"],
        "config_hash": bundle["config_hash"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    meta_tmp = output / "resume_latest.json.tmp"
    meta_target = output / "resume_latest.json"
    with meta_tmp.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(meta_tmp, meta_target)
    return target, digest


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_periodic_snapshot(latest: Path, epoch: int, interval: int, keep: int) -> Path | None:
    if interval <= 0 or epoch % interval:
        return None
    snapshot = latest.with_name(f"resume_epoch_{epoch:03d}.pt")
    if not snapshot.exists():
        temporary = snapshot.with_suffix(".tmp")
        with latest.open("rb") as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, snapshot)
        bundle = torch.load(snapshot, map_location="cpu", weights_only=False)
        _atomic_json(
            snapshot.with_suffix(".json"),
            {
                "checkpoint_sha256": sha256_file(snapshot),
                "resume_schema_version": bundle["resume_schema_version"],
                "run_id": bundle["run_id"],
                "source_hash": bundle["source_hash"],
                "config_hash": bundle["config_hash"],
                "completed_epoch": bundle["completed_epoch"],
                "next_epoch": bundle["next_epoch"],
                "status": "IMMUTABLE_FALLBACK",
            },
        )
    snapshots = sorted(latest.parent.glob("resume_epoch_*.pt"))
    for old in snapshots[:-max(keep, 2)]:
        old.unlink()
        old.with_suffix(".json").unlink(missing_ok=True)
    return snapshot


def find_latest_valid_snapshot(
    directory: str | Path,
    expected_identity: dict | None = None,
) -> tuple[Path, str] | None:
    """Report the newest independently hash-valid immutable fallback."""
    root = Path(directory)
    for metadata_path in sorted(root.glob("resume_epoch_*.json"), reverse=True):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if expected_identity is not None and any(
                expected_identity.get(key) is not None
                and metadata.get(key) != expected_identity[key]
                for key in (
                    "resume_schema_version", "run_id", "source_hash", "config_hash"
                )
            ):
                continue
            checkpoint = metadata_path.with_suffix(".pt")
            expected = metadata["checkpoint_sha256"]
            if checkpoint.is_file() and sha256_file(checkpoint) == expected:
                return checkpoint, expected
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return None


def load_resume_bundle(
    path: str | Path, *, expected_sha256: str | None, config: MPGConfig,
    run_id: str | None, source_hash: str,
) -> dict:
    checkpoint = Path(path)
    if expected_sha256 and sha256_file(checkpoint) != expected_sha256:
        raise RuntimeError("Resume SHA-256 mismatch")
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if bundle.get("resume_schema_version") != config.resume_schema_version:
        raise RuntimeError("Resume schema mismatch")
    if run_id is not None and bundle.get("run_id") != run_id:
        raise RuntimeError("Resume run_id mismatch")
    if bundle.get("source_hash") != source_hash:
        raise RuntimeError("Resume source hash mismatch")
    if bundle.get("config_hash") != config_hash(config):
        raise RuntimeError("Resume scientific/architecture config mismatch")
    monitor = bundle.get("early_stop_monitor_state")
    if monitor != {
        "start_epoch": config.early_stop_monitor_start_epoch,
        "active": bundle["completed_epoch"] >= config.early_stop_monitor_start_epoch,
        "patience": bundle["early_stop_counter"],
    }:
        raise RuntimeError("Resume early-stop monitor state mismatch")
    temperature = bundle.get("temperature_state")
    if temperature is not None:
        state_tau = bundle["model_state_dict"].get("motif_composer.current_tau")
        if (
            temperature.get("epoch") != bundle["completed_epoch"]
            or state_tau is None
            or not torch.isclose(
                torch.as_tensor(state_tau).cpu(),
                torch.tensor(float(temperature["current_tau"])),
                rtol=0.0,
                atol=1e-7,
            )
        ):
            raise RuntimeError("Resume scheduled temperature state mismatch")
    return bundle


def restore_training_state(
    bundle: dict, *, model: torch.nn.Module, ema: Any,
    optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any,
    loader_generator: torch.Generator,
    sampler: Any | None = None,
    dataset: Any | None = None,
) -> None:
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    ema.load_state_dict(bundle["ema_state_dict"])
    optimizer.load_state_dict(bundle["optimizer_state_dict"])
    scheduler.load_state_dict(bundle["scheduler_state_dict"])
    if scaler is not None and bundle["grad_scaler_state_dict"]:
        scaler.load_state_dict(bundle["grad_scaler_state_dict"])
    restore_rng_state(bundle["rng_state"], loader_generator)
    if sampler is not None and bundle.get("sampler_state") is not None:
        sampler.load_state_dict(bundle["sampler_state"])
    if dataset is not None and bundle.get("augmentation_state") is not None:
        dataset.load_state_dict(bundle["augmentation_state"])
