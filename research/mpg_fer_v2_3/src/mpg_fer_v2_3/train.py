"""EMA-selected training and exact epoch-boundary continuation for MPG-FER v2.3."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import torch
import torch.nn as nn
from torch.optim import AdamW
import torchvision.transforms.functional as TF

from .checkpoint import (
    atomic_save_resume,
    build_resume_bundle,
    load_resume_bundle,
    restore_training_state,
    save_periodic_snapshot,
    sha256_file,
)
from .config import MPGConfig
from .data import (
    FER2013Dataset,
    create_private_dataloader,
    create_training_dataloaders,
    validate_split_path,
)
from .ema import ModelEMA
from .evaluate import evaluate_raw_and_tta
from .losses import supervised_contrastive_loss, symmetric_js_divergence
from .model import MPGFER
from .utils import set_seed


MOTIF_DIAGNOSTICS = (
    "tau", "H_local_raw", "H_local_normalized", "H_global_raw",
    "H_global_normalized", "L_MI", "mean_entropy", "effective_motif_count",
    "min_utilization", "max_utilization", "std_utilization",
    "mean_top1_probability", "mean_top2_probability", "mean_top1_top2_margin",
    "mean_offdiag_prototype_cosine", "mean_alpha_8", "mean_alpha_12",
    "mean_alpha_16", "std_alpha_8", "std_alpha_12", "std_alpha_16",
)
ROUTING_DIAGNOSTIC_FIELDS = (
    "selected_k", "entropy", "top1_mass", "boundary_tie_count",
    "local_share", "meso_share", "far_share", "edge_universe_coverage",
)
ROUTING_DIAGNOSTICS = tuple(
    f"motif_l{layer}_{field}"
    for layer in range(1, 6)
    for field in ROUTING_DIAGNOSTIC_FIELDS
)
FIXED_ROUTING_DIAGNOSTIC_INDICES = tuple(range(16))
ROUTING_DIAGNOSTIC_CADENCE_EPOCHS = 1
OFFICIAL_PHYSICAL_BATCH = 16
OFFICIAL_GRADIENT_ACCUMULATION = 2


def validate_official_batch_contract(config: MPGConfig) -> None:
    pair = (config.batch_size, config.gradient_accumulation_steps)
    expected = (OFFICIAL_PHYSICAL_BATCH, OFFICIAL_GRADIENT_ACCUMULATION)
    if pair != expected:
        raise RuntimeError(
            f"OFFICIAL_BATCH_CONTRACT_VIOLATION: expected {expected}, got {pair}"
        )


def build_fixed_routing_diagnostic_batch(
    dataset: FER2013Dataset,
) -> tuple[tuple[int, ...], torch.Tensor]:
    """Materialize fixed non-augmented Train images without advancing any RNG."""
    indices = FIXED_ROUTING_DIAGNOSTIC_INDICES
    if len(dataset) <= indices[-1]:
        raise RuntimeError("Train split is too small for the fixed routing batch")
    images = torch.from_numpy(dataset.images[list(indices)].copy()).float().unsqueeze(1)
    return indices, images.div_(255.0)


def _support_jaccard(
    current: torch.Tensor, previous: torch.Tensor | None,
) -> tuple[float | None, float | None]:
    if previous is None:
        return None, None
    if current.shape != previous.shape:
        raise RuntimeError("Previous routing support shape is incompatible")
    intersection = (
        current.unsqueeze(-1)
        .eq(previous.unsqueeze(-2))
        .any(dim=-1)
        .sum()
        .item()
    )
    union = current.numel() + previous.numel() - intersection
    jaccard = float(intersection / union) if union else 1.0
    return jaccard, 1.0 - jaccard


def collect_fixed_routing_diagnostics(
    model: nn.Module,
    images: torch.Tensor,
    device: str | torch.device,
    previous_supports: dict[str, torch.Tensor] | None = None,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Read routing on a fixed batch while preserving RNG and module modes."""
    resolved_device = torch.device(device)
    cuda_devices: list[int] = []
    if resolved_device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device()
            if resolved_device.index is None
            else resolved_device.index
        ]
    modes = [(module, module.training) for module in model.modules()]
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            model.eval()
            with torch.no_grad():
                _, outputs = model(
                    images.to(resolved_device), return_routing_supports=True
                )
    finally:
        for module, training in modes:
            module.training = training

    layers: dict[str, dict] = {}
    current_supports: dict[str, torch.Tensor] = {}
    for layer in range(1, 6):
        prefix = f"motif_l{layer}"
        support = outputs[f"{prefix}_selected_indices"].detach().cpu()
        prior = None if previous_supports is None else previous_supports.get(prefix)
        jaccard, turnover = _support_jaccard(support, prior)
        current_supports[prefix] = support
        layers[prefix] = {
            "selected_k": int(outputs[f"{prefix}_selected_k"].detach()),
            "entropy": float(outputs[f"{prefix}_entropy"].detach()),
            "top1_mass": float(outputs[f"{prefix}_top1_mass"].detach()),
            "boundary_tie_count": int(
                outputs[f"{prefix}_boundary_tie_count"].detach()
            ),
            "local_share": float(outputs[f"{prefix}_local_share"].detach()),
            "meso_share": float(outputs[f"{prefix}_meso_share"].detach()),
            "far_share": float(outputs[f"{prefix}_far_share"].detach()),
            "edge_universe_coverage": float(
                outputs[f"{prefix}_edge_universe_coverage"].detach()
            ),
            "support_jaccard_previous_epoch": jaccard,
            "support_turnover_previous_epoch": turnover,
        }
    return {"layers": layers}, current_supports


class WarmupCosineScheduler:
    """Warm up, decay to an independent horizon, then hold the LR floor."""

    def __init__(
        self, optimizer, base_lr: float, warmup_epochs: int,
        lr_decay_end_epoch: int, max_epochs: int, eta_min: float,
    ) -> None:
        if not 0 < warmup_epochs < lr_decay_end_epoch <= max_epochs:
            raise ValueError(
                "require 0 < warmup_epochs < lr_decay_end_epoch <= max_epochs"
            )
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.lr_decay_end_epoch = int(lr_decay_end_epoch)
        self.max_epochs = int(max_epochs)
        self.eta_min = float(eta_min)
        self.last_epoch = 0

    def lr_for_epoch(self, epoch: int) -> float:
        if not 1 <= epoch <= self.max_epochs:
            raise ValueError(f"epoch must be in [1,{self.max_epochs}]")
        if epoch <= self.warmup_epochs:
            return self.base_lr * epoch / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / (
            self.lr_decay_end_epoch - self.warmup_epochs
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + math.cos(math.pi * progress))

    def step(self, epoch: int) -> float:
        lr = self.lr_for_epoch(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_epoch = epoch
        return lr

    def state_dict(self) -> dict:
        return {
            "base_lr": self.base_lr, "warmup_epochs": self.warmup_epochs,
            "lr_decay_end_epoch": self.lr_decay_end_epoch,
            "max_epochs": self.max_epochs, "eta_min": self.eta_min,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        expected = {
            "base_lr": self.base_lr,
            "warmup_epochs": self.warmup_epochs,
            "lr_decay_end_epoch": self.lr_decay_end_epoch,
            "max_epochs": self.max_epochs,
            "eta_min": self.eta_min,
        }
        for name, value in expected.items():
            if state[name] != value:
                raise RuntimeError(f"Scheduler {name} mismatch")
        self.last_epoch = int(state["last_epoch"])


def source_tree_hash(package_dir: str | Path | None = None) -> str:
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def is_better_checkpoint(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    return (candidate["accuracy"], candidate["macro_f1"], -candidate["loss"]) > (
        incumbent["accuracy"], incumbent["macro_f1"], -incumbent["loss"]
    )


def update_early_stop_patience(
    epoch: int, improved: bool, patience_counter: int, config: MPGConfig,
) -> int:
    """Select checkpoints always, but consume patience only from epoch 85."""
    if improved or epoch < config.early_stop_monitor_start_epoch:
        return 0
    return patience_counter + 1


def should_early_stop(epoch: int, patience_counter: int, config: MPGConfig) -> bool:
    return (
        epoch >= config.early_stop_monitor_start_epoch
        and patience_counter >= config.early_stop_patience
    )


def should_end_segment(
    elapsed_seconds: float,
    estimated_next_epoch_seconds: float,
    config: MPGConfig,
) -> bool:
    safety_margin = config.segment_safety_margin_minutes * 60.0
    return (
        elapsed_seconds + estimated_next_epoch_seconds + safety_margin
        >= config.segment_soft_limit_hours * 3600.0
    )


def consistency_selected(seed: int, epoch: int, group_index: int, probability: float) -> bool:
    """Stateless decision, stable across process and resume boundaries."""
    value = random.Random(seed + epoch * 1_000_003 + group_index * 9_176).random()
    return value < probability


def compute_training_loss(
    logits: torch.Tensor, outputs: dict[str, torch.Tensor], targets: torch.Tensor,
    criterion: nn.Module, config: MPGConfig, flipped_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weighted = {
        "ce_final": criterion(logits, targets),
        "weighted_ce_motif": config.aux_motif_weight * criterion(outputs["motif_logits"], targets),
        "weighted_ce_pixel": config.aux_pixel_weight * criterion(outputs["pixel_logits"], targets),
        "weighted_diversity": config.lambda_div * outputs["loss_diversity"],
        "weighted_mi": config.lambda_mi * outputs["loss_mi"],
    }
    raw_consistency = logits.new_zeros(())
    if flipped_logits is not None:
        raw_consistency = symmetric_js_divergence(logits, flipped_logits)
    weighted["weighted_consistency"] = config.lambda_consistency * raw_consistency
    raw_supcon, supcon_stats = supervised_contrastive_loss(
        outputs["supcon_embeddings"], targets, config.supcon_temperature
    )
    weighted["supcon_loss_weighted"] = config.lambda_supcon * raw_supcon
    total = sum(weighted.values())
    components = {
        **weighted,
        "consistency_loss_raw": raw_consistency,
        "supcon_loss_raw": raw_supcon,
        **supcon_stats,
    }
    return total, components


def train_one_epoch(
    model: nn.Module, dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer, device: str | torch.device,
    scaler: torch.amp.GradScaler | None, config: MPGConfig, criterion: nn.Module,
    ema: ModelEMA | None = None, epoch: int = 1, global_optimizer_step: int = 0,
) -> tuple[dict, int]:
    model.train()
    if hasattr(model, "set_epoch_temperature"):
        model.set_epoch_temperature(epoch)
    if ema is not None and hasattr(ema.module, "set_epoch_temperature"):
        ema.module.set_epoch_temperature(epoch)
    optimizer.zero_grad(set_to_none=True)
    component_names = (
        "ce_final", "weighted_ce_motif", "weighted_ce_pixel",
        "weighted_diversity", "weighted_mi", "consistency_loss_raw",
        "weighted_consistency", "supcon_loss_raw", "supcon_loss_weighted",
        "valid_supcon_anchor_fraction", "mean_positive_count",
    )
    totals = {
        name: 0.0
        for name in (
            "train_loss", *component_names, *MOTIF_DIAGNOSTICS,
            *ROUTING_DIAGNOSTICS,
        )
    }
    correct = samples = 0
    accum_steps = config.gradient_accumulation_steps
    use_amp = config.use_amp and torch.cuda.is_available()
    num_batches = len(dataloader)
    selected_groups: list[int] = []

    for step, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        group_index = step // accum_steps
        group_start = group_index * accum_steps
        group_size = min(accum_steps, num_batches - group_start)
        use_consistency = consistency_selected(
            config.seed, epoch, group_index, config.consistency_probability
        )
        if use_consistency and group_index not in selected_groups:
            selected_groups.append(group_index)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, outputs = model(images)
            flipped_logits = model(TF.hflip(images))[0] if use_consistency else None
            loss, components = compute_training_loss(
                logits, outputs, targets, criterion, config, flipped_logits
            )
            backward_loss = loss / group_size
        if scaler is not None:
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()

        end_group = (step + 1) % accum_steps == 0 or step + 1 == num_batches
        if end_group:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            successful = True
            if scaler is not None:
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                successful = scaler.get_scale() >= old_scale
            else:
                optimizer.step()
            if successful:
                global_optimizer_step += 1
                if ema is not None:
                    ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        batch_size = len(targets)
        totals["train_loss"] += float(loss.detach()) * batch_size
        for name, value in components.items():
            totals[name] += float(value.detach()) * batch_size
        for name in MOTIF_DIAGNOSTICS:
            totals[name] += float(outputs[name].detach()) * batch_size
        for name in ROUTING_DIAGNOSTICS:
            value = float(outputs[name].detach())
            if name.endswith("_boundary_tie_count"):
                totals[name] += value
            else:
                totals[name] += value * batch_size
        correct += int((logits.argmax(dim=-1) == targets).sum())
        samples += batch_size

    stats = {
        name: (
            value
            if name.endswith("_boundary_tie_count")
            else value / samples
        )
        for name, value in totals.items()
    }
    stats["train_accuracy"] = correct / samples
    stats["consistency_groups"] = selected_groups
    stats["consistency_group_fraction"] = len(selected_groups) / math.ceil(num_batches / accum_steps)
    return stats, global_optimizer_step


def run_micro_overfit_preflight(
    train_csv: str | Path, config: MPGConfig | None = None,
    device: str | torch.device | None = None, raise_on_failure: bool = True,
) -> dict:
    config = config or MPGConfig()
    path = validate_split_path(train_csv, "train")
    device = torch.device(device or (config.device if torch.cuda.is_available() else "cpu"))
    set_seed(config.seed)
    dataset = FER2013Dataset(path, split="train", augment=False)
    images = torch.stack([dataset[i][0] for i in range(config.micro_overfit_samples)]).to(device)
    targets = torch.tensor([dataset[i][1] for i in range(config.micro_overfit_samples)], device=device)
    model = MPGFER(config).to(device)
    model.set_epoch_temperature(1)
    optimizer = AdamW(model.parameters(), lr=config.micro_overfit_learning_rate, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    accuracy, final_loss, steps = 0.0, float("nan"), 0
    for steps in range(1, config.micro_overfit_max_steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits, outputs = model(images)
        loss, _ = compute_training_loss(logits, outputs, targets, criterion, config)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip); optimizer.step()
        if steps % 5 == 0 or steps == config.micro_overfit_max_steps:
            model.eval()
            with torch.no_grad():
                logits, outputs = model(images)
                final_loss = float(compute_training_loss(logits, outputs, targets, criterion, config)[0])
                accuracy = float((logits.argmax(dim=-1) == targets).float().mean())
            if accuracy >= config.micro_overfit_target:
                break
    result = {"samples": config.micro_overfit_samples, "accuracy": accuracy,
              "target": config.micro_overfit_target, "steps": steps,
              "final_loss": final_loss, "passed": accuracy >= config.micro_overfit_target}
    if not result["passed"] and raise_on_failure:
        raise RuntimeError(f"16-example micro-overfit preflight failed: {result}")
    return result


def run_bounded_runtime_preflight(
    train_csv: str | Path, config: MPGConfig | None = None,
    device: str | torch.device | None = None,
) -> dict:
    config = config or MPGConfig()
    path = validate_split_path(train_csv, "train")
    device = torch.device(device or config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the bounded runtime preflight")
    set_seed(config.seed)
    dataset = FER2013Dataset(path, split="train", augment=False)
    images = torch.stack([dataset[i][0] for i in range(config.batch_size)]).to(device)
    targets = torch.tensor([dataset[i][1] for i in range(config.batch_size)], device=device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    model = MPGFER(config).to(device).train()
    model.set_epoch_temperature(1)
    ema = ModelEMA(model, config.ema_decay)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    step_started = time.perf_counter()
    with torch.amp.autocast("cuda", enabled=True):
        logits, outputs = model(images)
        flipped_logits, _ = model(TF.hflip(images))
        loss, components = compute_training_loss(
            logits, outputs, targets, criterion, config, flipped_logits
        )
    if not torch.isfinite(loss) or not all(
        torch.isfinite(value).all() for value in components.values()
    ):
        raise FloatingPointError("Non-finite bounded preflight loss")
    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    gradient_names = {
        "assignment_query": model.motif_composer.assignment_query.weight,
        "prototype_key": model.motif_composer.prototype_key.weight,
        "prototypes": model.motif_composer.prototypes,
        "scale_gate": model.motif_composer.scale_gate.weight,
        "pixel_projection": model.pixel_proj[0].weight,
        "pixel_readout": model.pixel_readout_proj[0].weight,
        "motif_readout": model.motif_readout_proj[0].weight,
        "supcon_projection": model.supcon_head[0].weight,
        "classifier": model.classifier[-1].weight,
    }
    gradients = {name: bool(p.grad is not None and torch.isfinite(p.grad).all() and torch.any(p.grad != 0)) for name, p in gradient_names.items()}
    if not all(gradients.values()):
        raise FloatingPointError(f"Invalid gradients: {gradients}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    old_scale = scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    optimizer_step_succeeded = scaler.get_scale() >= old_scale
    if optimizer_step_succeeded:
        ema.update(model)
    torch.cuda.synchronize(device)
    step_time_seconds = time.perf_counter() - step_started
    return {
        "physical_batch": config.batch_size,
        "gradient_accumulation": config.gradient_accumulation_steps,
        "amp_enabled": True,
        "worst_case_consistency_forward": True,
        "supcon_enabled": True,
        "loss": float(loss.detach()),
        "loss_components": {name: float(value.detach()) for name, value in components.items()},
        "gradient_audits": gradients,
        "optimizer_step_succeeded": optimizer_step_succeeded,
        "ema_updates": ema.num_updates,
        "step_time_seconds": step_time_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "passed": True,
    }


def _write_history(history: list[dict], output: Path) -> None:
    json_path = output / "history.json"
    json_temporary = output / "history.json.tmp"
    with json_temporary.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(json_temporary, json_path)
    if not history:
        return
    rows = []
    for entry in history:
        flat = {k: v for k, v in entry.items() if not isinstance(v, (dict, list))}
        for view in ("val_raw", "val_tta"):
            for metric in ("loss", "accuracy", "macro_f1"):
                flat[f"{view}_{metric}"] = entry[view][metric]
        rows.append(flat)
    csv_path = output / "history.csv"
    csv_temporary = output / "history.csv.tmp"
    with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(csv_temporary, csv_path)


def _atomic_json_document(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_routing_diagnostics(
    history: list[dict], output: Path, best_epoch: int | None,
) -> None:
    trajectory = []
    for entry in history:
        fixed = entry.get("routing_diagnostics")
        if fixed is None:
            raise RuntimeError(
                f"Epoch {entry.get('epoch')} lacks fixed routing diagnostics"
            )
        ordinary = {}
        for layer in range(1, 6):
            prefix = f"motif_l{layer}"
            ordinary[prefix] = {
                field: entry[f"{prefix}_{field}"]
                for field in ROUTING_DIAGNOSTIC_FIELDS
            }
        trajectory.append({
            "epoch": entry["epoch"],
            "ordinary_train_batch_statistics": ordinary,
            "fixed_batch_statistics": fixed["layers"],
        })
    payload = {
        "schema_version": 1,
        "checkpoint_selection_used_routing_diagnostics": False,
        "model_observed": "online training model after each completed epoch",
        "cadence_epochs": ROUTING_DIAGNOSTIC_CADENCE_EPOCHS,
        "fixed_batch": {
            "split": "Train",
            "sample_indices": list(FIXED_ROUTING_DIAGNOSTIC_INDICES),
            "augmentation": False,
        },
        "spatial_bins": {
            "grid": "fixed 7x7 occurrence grid",
            "distance": "Chebyshev",
            "LOCAL": "d=1",
            "MESO": "d in {2,3}",
            "FAR": "d>=4",
        },
        "best_epoch": best_epoch,
        "final_epoch": None if not history else history[-1]["epoch"],
        "trajectory": trajectory,
    }
    _atomic_json_document(output / "routing_diagnostics.json", payload)


def _save_best_ema(path: Path, ema: ModelEMA, epoch: int, metrics: dict, config: MPGConfig, source_hash: str) -> None:
    temporary = path.with_suffix(".tmp")
    scheduled_tau = (
        float(ema.module.motif_composer.temperature)
        if hasattr(ema.module, "motif_composer")
        else None
    )
    with temporary.open("wb") as handle:
        torch.save({
            "checkpoint_type": "EMA_INFERENCE_ONLY", "weights_type": "EMA", "epoch": epoch,
            "model_state_dict": ema.module.state_dict(), "ema_state_dict": ema.state_dict(),
            "scheduled_tau": scheduled_tau,
            "val_metrics": metrics, "config": asdict(config), "source_hash": source_hash,
            "selection_metric": "EMA PublicTest flip-TTA accuracy",
            "selection_tiebreak": ["higher EMA TTA macro-F1", "lower EMA TTA loss"],
        }, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _segment_manifest(
    output: Path, *, run_id: str, segment_number: int, start_epoch: int,
    end_epoch: int, next_epoch: int, wallclock: float, resume_sha: str,
    best_epoch: int | None, best_metrics: dict | None, status: str,
) -> dict:
    manifest = {
        "run_id": run_id, "segment_number": segment_number,
        "start_epoch": start_epoch, "end_epoch": end_epoch, "next_epoch": next_epoch,
        "wallclock_seconds": wallclock, "resume_sha256": resume_sha,
        "best_epoch": best_epoch,
        "best_public_tta_accuracy": None if best_metrics is None else best_metrics["accuracy"],
        "status": status,
    }
    _atomic_json_document(output / "segment_manifest.json", manifest)
    return manifest


def run_training(
    train_csv: str | Path, val_csv: str | Path, output_dir: str | Path,
    preflight_result: dict, config: MPGConfig | None = None,
    resume_path: str | Path | None = None, resume_sha256: str | None = None,
) -> dict:
    """Train/select with EMA PublicTest metrics; never opens PrivateTest."""
    config = config or MPGConfig()
    validate_official_batch_contract(config)
    fresh_gate = (
        preflight_result.get("passed")
        and preflight_result.get("samples") == config.micro_overfit_samples
    )
    resume_gate = resume_path is not None and preflight_result.get(
        "resume_compatibility_passed"
    )
    if not (fresh_gate or resume_gate):
        raise RuntimeError(
            "Training refused: valid real-16 fresh gate or verified resume gate required"
        )
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    source_hash = source_tree_hash()
    generator = torch.Generator().manual_seed(config.seed)
    loaders = create_training_dataloaders(
        train_csv, val_csv, config.batch_size, config.num_workers,
        seed=config.seed, generator=generator,
    )
    fixed_routing_indices, fixed_routing_images = (
        build_fixed_routing_diagnostic_batch(loaders["train"].dataset)
    )
    if fixed_routing_indices != FIXED_ROUTING_DIAGNOSTIC_INDICES:
        raise RuntimeError("Fixed routing diagnostic indices changed unexpectedly")
    model = MPGFER(config).to(device)
    model_summary = model.model_summary(
        source_sha256=source_hash,
        source_git_commit=os.environ.get("MPG_FER_SOURCE_GIT_COMMIT"),
    )
    _atomic_json_document(output / "v23_model_summary.json", model_summary)
    print(json.dumps(model_summary, indent=2))
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer, config.learning_rate, config.warmup_epochs,
        config.lr_decay_end_epoch, config.max_epochs, config.min_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda") if config.use_amp and torch.cuda.is_available() else None
    ema = ModelEMA(model, config.ema_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    run_id = config.run_id or str(uuid.uuid4())
    start_epoch, global_step = 1, 0
    best_metrics = best_comparator = None
    best_epoch = None
    patience = 0
    history: list[dict] = []
    previous_routing_supports: dict[str, torch.Tensor] | None = None
    resumed_from = None
    if resume_path is not None:
        bundle = load_resume_bundle(
            resume_path, expected_sha256=resume_sha256, config=config,
            run_id=config.run_id, source_hash=source_hash,
        )
        restore_training_state(
            bundle, model=model, ema=ema, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, loader_generator=generator,
            sampler=loaders["train"].sampler,
            dataset=loaders["train"].dataset,
        )
        run_id = bundle["run_id"]
        start_epoch = int(bundle["next_epoch"])
        global_step = int(bundle["global_optimizer_step"])
        best_comparator = bundle["best_comparator_state"]
        best_metrics = bundle["best_metrics"]
        best_epoch = bundle["best_epoch"]
        patience = int(bundle["early_stop_counter"])
        history = list(bundle["history"])
        resumed_from = str(Path(resume_path).resolve())
        print("RESUMING EXISTING MPG-FER v2.3 RUN")
        print(f"run_id={run_id}")
        print(f"completed epoch={bundle['completed_epoch']}")
        print(f"next epoch={start_epoch}")
        print(f"restored LR={optimizer.param_groups[0]['lr']:.10f}")
        print(f"restored optimizer step={global_step}")
        print(f"restored best epoch={best_epoch}")
        print(f"restored patience={patience}")
        print(f"restored tau={float(model.motif_composer.temperature):.10f}")
        print(f"restored EMA updates={ema.num_updates}")
        print(
            "restored GradScaler scale="
            + ("none" if scaler is None else f"{scaler.get_scale():.10f}")
        )
        print(
            "early-stop monitor active="
            f"{start_epoch >= config.early_stop_monitor_start_epoch}"
        )
        _, previous_routing_supports = collect_fixed_routing_diagnostics(
            model, fixed_routing_images, device
        )

    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    segment_started = time.monotonic()
    segment_start_epoch = start_epoch
    status = "TRAINING_COMPLETED"
    resume_digest = ""
    end_epoch = start_epoch - 1
    checkpoint_path = output / "best_val_acc.pt"
    print(f"Starting MPG-FER v2.3 training on device: {device}; run_id={run_id}")

    for epoch in range(start_epoch, config.max_epochs + 1):
        loaders["train"].dataset.set_epoch(epoch)
        loaders["train"].sampler.set_epoch(epoch)
        lr = scheduler.step(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.monotonic()
        try:
            stats, global_step = train_one_epoch(
                model, loaders["train"], optimizer, device, scaler, config,
                criterion, ema=ema, epoch=epoch, global_optimizer_step=global_step,
            )
            ema.module.set_epoch_temperature(epoch)
            validation = evaluate_raw_and_tta(
                ema.module, loaders["val"], device, config.use_amp
            )
            tta = validation["tta"]
            improved = is_better_checkpoint(tta, best_comparator)
            routing_diagnostics, current_routing_supports = (
                collect_fixed_routing_diagnostics(
                    model, fixed_routing_images, device,
                    previous_supports=previous_routing_supports,
                )
            )
        except Exception:
            latest_resume = output / "resume_latest.pt"
            failed_sha = sha256_file(latest_resume) if latest_resume.exists() else ""
            _segment_manifest(
                output, run_id=run_id, segment_number=config.segment_number,
                start_epoch=segment_start_epoch, end_epoch=epoch - 1,
                next_epoch=epoch, wallclock=time.monotonic() - segment_started,
                resume_sha=failed_sha, best_epoch=best_epoch,
                best_metrics=best_comparator, status="FAILED",
            )
            raise
        duration = time.monotonic() - epoch_started
        if improved:
            best_comparator = dict(tta); best_metrics = {"raw": validation["raw"], "tta": tta}
            best_epoch = epoch
            _save_best_ema(checkpoint_path, ema, epoch, best_metrics, config, source_hash)
        patience = update_early_stop_patience(epoch, improved, patience, config)
        previous_routing_supports = current_routing_supports
        entry = {
            "epoch": epoch, "lr": lr, "epoch_duration_sec": duration,
            "val_raw": validation["raw"], "val_tta": tta,
            "global_optimizer_step": global_step,
            "early_stop_monitor_active": epoch >= config.early_stop_monitor_start_epoch,
            "early_stop_patience": patience,
            "routing_diagnostics": routing_diagnostics,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
            **stats,
        }
        history.append(entry); end_epoch = epoch; _write_history(history, output)
        _write_routing_diagnostics(history, output, best_epoch)
        print(
            f"Epoch {epoch:03d}/{config.max_epochs} [{duration:.1f}s] LR={lr:.8f} "
            f"TrainAcc={stats['train_accuracy']:.4f} RawAcc={validation['raw']['accuracy']:.4f} "
            f"EMATTAAcc={tta['accuracy']:.4f} EMATTAF1={tta['macro_f1']:.4f} "
            f"tau={stats['tau']:.4f} Hlocal={stats['H_local_normalized']:.4f} "
            f"Hglobal={stats['H_global_normalized']:.4f} alpha="
            f"({stats['mean_alpha_8']:.3f},{stats['mean_alpha_12']:.3f},{stats['mean_alpha_16']:.3f}) "
            f"ConsFrac={stats['consistency_group_fraction']:.3f} "
            f"SupCon={stats['supcon_loss_raw']:.4f} "
            f"SupConValid={stats['valid_supcon_anchor_fraction']:.3f}"
        )

        consistency_state = {
            "algorithm": "stateless_seed_epoch_group_v1",
            "last_completed_epoch": epoch,
            "selected_groups": stats["consistency_groups"],
        }
        bundle = build_resume_bundle(
            config=config, run_id=run_id, source_hash=source_hash,
            completed_epoch=epoch, global_optimizer_step=global_step,
            model=model, ema=ema, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, best_comparator_state=best_comparator,
            best_epoch=best_epoch, best_metrics=best_metrics,
            early_stop_counter=patience, history=history,
            loader_generator=generator, consistency_state=consistency_state,
            sampler_state=loaders["train"].sampler.state_dict(),
            augmentation_state=loaders["train"].dataset.state_dict(),
        )
        latest, resume_digest = atomic_save_resume(bundle, output)
        save_periodic_snapshot(
            latest, epoch, config.resume_snapshot_interval,
            config.resume_snapshots_to_keep,
        )
        if should_early_stop(epoch, patience, config):
            print(f"Early stopping triggered at epoch {epoch}")
            break
        elapsed = time.monotonic() - segment_started
        estimate = max(item["epoch_duration_sec"] for item in history[-3:])
        if (
            epoch < config.max_epochs
            and should_end_segment(elapsed, estimate, config)
        ):
            status = "NEEDS_RESUME"
            _, resume_digest = atomic_save_resume(bundle, output, status=status)
            break

    elapsed = time.monotonic() - segment_started
    if end_epoch >= segment_start_epoch and status == "TRAINING_COMPLETED":
        bundle = build_resume_bundle(
            config=config, run_id=run_id, source_hash=source_hash,
            completed_epoch=end_epoch, global_optimizer_step=global_step,
            model=model, ema=ema, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, best_comparator_state=best_comparator,
            best_epoch=best_epoch, best_metrics=best_metrics,
            early_stop_counter=patience, history=history,
            loader_generator=generator,
            consistency_state={"algorithm": "stateless_seed_epoch_group_v1", "last_completed_epoch": end_epoch},
            sampler_state=loaders["train"].sampler.state_dict(),
            augmentation_state=loaders["train"].dataset.state_dict(),
        )
        _, resume_digest = atomic_save_resume(bundle, output, status=status)
    segment = _segment_manifest(
        output, run_id=run_id, segment_number=config.segment_number,
        start_epoch=segment_start_epoch, end_epoch=end_epoch,
        next_epoch=end_epoch + 1, wallclock=elapsed, resume_sha=resume_digest,
        best_epoch=best_epoch, best_metrics=best_comparator, status=status,
    )
    result = {
        "run_id": run_id, "source_hash": source_hash, "status": status,
        "best_epoch": best_epoch, "best_metrics": best_metrics,
        "best_checkpoint": str(checkpoint_path.resolve()) if checkpoint_path.exists() else None,
        "best_checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.exists() else None,
        "resume_checkpoint": str((output / "resume_latest.pt").resolve()),
        "resume_sha256": resume_digest, "resumed_from": resumed_from,
        "segment": segment, "PRIVATE_EVALUATED": False,
    }
    _atomic_json_document(output / "execution_manifest.json", result)
    return result


def evaluate_private_once(
    test_csv: str | Path, output_dir: str | Path, config: MPGConfig | None = None,
) -> dict:
    config = config or MPGConfig()
    output = Path(output_dir)
    manifest_path = output / "execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "TRAINING_COMPLETED":
        raise RuntimeError("Private evaluation refused before training completion")
    if manifest.get("PRIVATE_EVALUATED"):
        raise RuntimeError("PrivateTest one-shot already recorded")
    checkpoint_path = Path(manifest["best_checkpoint"])
    if sha256_file(checkpoint_path) != manifest["best_checkpoint_sha256"]:
        raise RuntimeError("Frozen EMA checkpoint hash mismatch")
    loader = create_private_dataloader(test_csv, config.batch_size, config.num_workers)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_tau = checkpoint["model_state_dict"].get("motif_composer.current_tau")
    if state_tau is None or not math.isclose(
        float(state_tau), float(checkpoint["scheduled_tau"]), rel_tol=0.0, abs_tol=1e-7
    ):
        raise RuntimeError("Best checkpoint scheduled temperature mismatch")
    model = MPGFER(config).to(device); model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics = evaluate_raw_and_tta(model, loader, device, config.use_amp)
    _atomic_json_document(output / "private_metrics.json", metrics)
    manifest["private_metrics"] = metrics
    manifest["private_evaluated_only_after_freeze"] = True
    manifest["PRIVATE_EVALUATED"] = True
    _atomic_json_document(manifest_path, manifest)
    return metrics
