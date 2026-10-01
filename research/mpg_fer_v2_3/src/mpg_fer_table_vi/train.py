"""Authorized training entrypoint for the preregistered Table VI framework."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import uuid

import torch
import torch.nn as nn
from torch.optim import AdamW
import torchvision.transforms.functional as TF

from .model import AblationMPGFER, AblationMode
from .protocol import (
    AblationConfig,
    ablation_source_tree_hash,
    evaluate_canonical_private_fp32,
    evaluate_canonical_public_fp32,
    validate_ablation_data_paths,
    validate_final_recipe_lock,
    write_ablation_manifest,
    write_checksums,
    write_json,
)
from mpg_fer_v2_3.checkpoint import (
    atomic_save_resume,
    build_resume_bundle,
    load_resume_bundle,
    restore_training_state,
    save_periodic_snapshot,
    sha256_file,
)
from mpg_fer_v2_3.data import create_private_dataloader, create_training_dataloaders
from mpg_fer_v2_3.data import FER2013Dataset, validate_split_path
from mpg_fer_v2_3.ema import ModelEMA
from mpg_fer_v2_3.evaluate import evaluate_raw_and_tta
from mpg_fer_v2_3.losses import supervised_contrastive_loss, symmetric_js_divergence
from mpg_fer_v2_3 import train as baseline_train
from mpg_fer_v2_3.utils import set_seed


class RecipeEpochScheduler:
    """Small stateful epoch scheduler for recipe families not in the baseline."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: AblationConfig) -> None:
        self.optimizer = optimizer
        self.family = config.scheduler_family
        self.base_lr = float(config.learning_rate)
        self.min_lr = float(config.min_learning_rate)
        self.max_epochs = int(config.max_epochs)
        self.last_epoch = 0

    def step(self, epoch: int) -> float:
        self.last_epoch = int(epoch)
        if self.family == "constant":
            lr = self.base_lr
        elif self.family == "cosine_annealing":
            progress = min(max((epoch - 1) / max(self.max_epochs - 1, 1), 0.0), 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        else:  # pragma: no cover - constructor is reached only through factory
            raise RuntimeError(f"Unsupported scheduler family: {self.family}")
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "max_epochs": self.max_epochs,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "family": self.family,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "max_epochs": self.max_epochs,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Recipe scheduler state/config mismatch")
        self.last_epoch = int(state["last_epoch"])


def build_recipe_optimizer(
    model: nn.Module, config: AblationConfig
) -> torch.optim.Optimizer:
    """Construct only the optimizer family and kwargs frozen by the recipe."""
    family = config.optimizer_family
    kwargs = dict(config.optimizer_kwargs)
    forbidden = {"lr", "weight_decay"}.intersection(kwargs)
    if forbidden:
        raise RuntimeError(f"Optimizer kwargs duplicate locked fields: {sorted(forbidden)}")
    allowed = {
        "AdamW": {"betas", "eps", "amsgrad"},
        "Adam": {"betas", "eps", "amsgrad"},
        "SGD": {"momentum", "dampening", "nesterov"},
    }
    if family not in allowed:
        raise RuntimeError(f"Unresolved or unsupported optimizer family: {family!r}")
    extra = set(kwargs) - allowed[family]
    if extra:
        raise RuntimeError(f"Unsupported {family} kwargs: {sorted(extra)}")
    if "betas" in kwargs:
        kwargs["betas"] = tuple(kwargs["betas"])
    optimizer_type = {
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }[family]
    return optimizer_type(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        **kwargs,
    )


def build_recipe_scheduler(
    optimizer: torch.optim.Optimizer, config: AblationConfig
) -> Any:
    """Construct the exact scheduler family frozen by the final recipe."""
    if config.scheduler_kwargs:
        raise RuntimeError(
            f"Unsupported {config.scheduler_family} kwargs: "
            f"{sorted(config.scheduler_kwargs)}"
        )
    if config.scheduler_family == "linear_warmup_cosine_then_floor":
        return baseline_train.WarmupCosineScheduler(
            optimizer,
            config.learning_rate,
            config.warmup_epochs,
            config.lr_decay_end_epoch,
            config.max_epochs,
            config.min_learning_rate,
        )
    if config.scheduler_family in {"constant", "cosine_annealing"}:
        return RecipeEpochScheduler(optimizer, config)
    raise RuntimeError(
        f"Unresolved or unsupported scheduler family: {config.scheduler_family!r}"
    )


def _model_factory(config: AblationConfig) -> AblationMPGFER:
    return AblationMPGFER(config, AblationMode(config.ablation_mode))


def compute_ablation_training_loss(
    logits: torch.Tensor,
    outputs: dict[str, Any],
    targets: torch.Tensor,
    criterion: nn.Module,
    config: AblationConfig,
    flipped_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    """Apply the frozen weights while preserving explicit N/A components."""
    applicability = outputs.get("loss_applicability", {})
    weighted: dict[str, torch.Tensor | None] = {
        "ce_final": criterion(logits, targets),
        "weighted_ce_motif": config.aux_motif_weight
        * criterion(outputs["motif_logits"], targets),
        "weighted_ce_pixel": config.aux_pixel_weight
        * criterion(outputs["pixel_logits"], targets),
        "weighted_diversity": (
            config.lambda_div * outputs["loss_diversity"]
            if applicability.get("prototype_diversity", True)
            else None
        ),
        "weighted_mi": (
            config.lambda_mi * outputs["loss_mi"]
            if applicability.get("prototype_mi", True)
            else None
        ),
    }
    raw_consistency = logits.new_zeros(())
    if flipped_logits is not None:
        raw_consistency = symmetric_js_divergence(logits, flipped_logits)
    weighted["weighted_consistency"] = config.lambda_consistency * raw_consistency
    raw_supcon, supcon_stats = supervised_contrastive_loss(
        outputs["supcon_embeddings"], targets, config.supcon_temperature
    )
    weighted["supcon_loss_weighted"] = config.lambda_supcon * raw_supcon
    total = sum(value for value in weighted.values() if value is not None)
    return total, {
        **weighted,
        "consistency_loss_raw": raw_consistency,
        "supcon_loss_raw": raw_supcon,
        **supcon_stats,
    }


def train_ablation_one_epoch(
    model: AblationMPGFER,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    scaler: torch.amp.GradScaler | None,
    config: AblationConfig,
    criterion: nn.Module,
    *,
    ema: ModelEMA | None = None,
    epoch: int = 1,
    global_optimizer_step: int = 0,
) -> tuple[dict[str, Any], int]:
    model.train()
    model.set_epoch_temperature(epoch)
    if ema is not None:
        ema.module.set_epoch_temperature(epoch)
    optimizer.zero_grad(set_to_none=True)
    component_names = (
        "ce_final",
        "weighted_ce_motif",
        "weighted_ce_pixel",
        "weighted_diversity",
        "weighted_mi",
        "consistency_loss_raw",
        "weighted_consistency",
        "supcon_loss_raw",
        "supcon_loss_weighted",
        "valid_supcon_anchor_fraction",
        "mean_positive_count",
    )
    all_names = (
        "train_loss",
        *component_names,
        *baseline_train.MOTIF_DIAGNOSTICS,
        *baseline_train.ROUTING_DIAGNOSTICS,
    )
    totals: dict[str, float | None] = {name: 0.0 for name in all_names}
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
        use_consistency = baseline_train.consistency_selected(
            config.seed, epoch, group_index, config.consistency_probability
        )
        if use_consistency and group_index not in selected_groups:
            selected_groups.append(group_index)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, outputs = model(images)
            flipped_logits = model(TF.hflip(images))[0] if use_consistency else None
            loss, components = compute_ablation_training_loss(
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
        assert totals["train_loss"] is not None
        totals["train_loss"] += float(loss.detach()) * batch_size
        for name, value in components.items():
            if value is None:
                totals[name] = None
            elif totals[name] is not None:
                totals[name] += float(value.detach()) * batch_size
        for name in baseline_train.MOTIF_DIAGNOSTICS:
            value = outputs.get(name)
            if value is None:
                totals[name] = None
            elif totals[name] is not None:
                totals[name] += float(value.detach()) * batch_size
        for name in baseline_train.ROUTING_DIAGNOSTICS:
            value = outputs.get(name)
            if value is None:
                totals[name] = None
            elif totals[name] is not None:
                number = float(value.detach())
                totals[name] += (
                    number
                    if name.endswith("_boundary_tie_count")
                    else number * batch_size
                )
        correct += int((logits.argmax(dim=-1) == targets).sum())
        samples += batch_size
    stats = {
        name: (
            None
            if value is None
            else value if name.endswith("_boundary_tie_count") else value / samples
        )
        for name, value in totals.items()
    }
    stats["train_accuracy"] = correct / samples
    stats["consistency_groups"] = selected_groups
    stats["consistency_group_fraction"] = len(selected_groups) / math.ceil(
        num_batches / accum_steps
    )
    return stats, global_optimizer_step


def collect_ablation_routing_diagnostics(
    model: AblationMPGFER,
    images: torch.Tensor,
    device: str | torch.device,
    previous_supports: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not model.diagnostic_applicability["motif_routing_diagnostics"]:
        return {"applicable": False, "layers": None}, {}
    diagnostics, supports = baseline_train.collect_fixed_routing_diagnostics(
        model, images, device, previous_supports=previous_supports
    )
    diagnostics["applicable"] = True
    return diagnostics, supports


def _write_ablation_routing_diagnostics(
    history: list[dict[str, Any]], output: Path, best_epoch: int | None
) -> None:
    if history and history[0]["routing_diagnostics"].get("applicable") is False:
        baseline_train._atomic_json_document(
            output / "routing_diagnostics.json",
            {
                "schema_version": 1,
                "applicable": False,
                "layers": None,
                "checkpoint_selection_used_routing_diagnostics": False,
                "best_epoch": best_epoch,
                "final_epoch": history[-1]["epoch"],
            },
        )
        return
    baseline_train._write_routing_diagnostics(history, output, best_epoch)


def run_ablation_micro_overfit_preflight(
    train_csv: str | Path,
    config: AblationConfig,
    device: str | torch.device | None = None,
    *,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Run the existing real-16 gate on the selected ablation mode."""
    path = validate_split_path(train_csv, "train")
    resolved = torch.device(
        device or (config.device if torch.cuda.is_available() else "cpu")
    )
    set_seed(config.seed)
    dataset = FER2013Dataset(path, split="train", augment=False)
    images = torch.stack(
        [dataset[index][0] for index in range(config.micro_overfit_samples)]
    ).to(resolved)
    targets = torch.tensor(
        [dataset[index][1] for index in range(config.micro_overfit_samples)],
        device=resolved,
    )
    model = _model_factory(config).to(resolved)
    model.set_epoch_temperature(1)
    optimizer = AdamW(
        model.parameters(), lr=config.micro_overfit_learning_rate, weight_decay=0.0
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    accuracy, final_loss, steps = 0.0, None, 0
    for steps in range(1, config.micro_overfit_max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, outputs = model(images)
        loss, _ = compute_ablation_training_loss(
            logits, outputs, targets, criterion, config
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if steps % 5 == 0 or steps == config.micro_overfit_max_steps:
            model.eval()
            with torch.no_grad():
                logits, outputs = model(images)
                final_loss = float(
                    compute_ablation_training_loss(
                        logits, outputs, targets, criterion, config
                    )[0]
                )
                accuracy = float((logits.argmax(dim=-1) == targets).float().mean())
            if accuracy >= config.micro_overfit_target:
                break
    result = {
        "mode": config.ablation_mode,
        "samples": config.micro_overfit_samples,
        "accuracy": accuracy,
        "target": config.micro_overfit_target,
        "steps": steps,
        "final_loss": final_loss,
        "passed": accuracy >= config.micro_overfit_target,
    }
    if not result["passed"] and raise_on_failure:
        raise RuntimeError(f"Ablation real-16 preflight failed: {result}")
    return result


def _run_ablation_training_core(
    train_csv: str | Path,
    public_csv: str | Path,
    output_dir: str | Path,
    preflight_result: dict[str, Any],
    config: AblationConfig,
    *,
    resume_path: str | Path | None = None,
    resume_sha256: str | None = None,
) -> dict[str, Any]:
    """Ablation-aware copy of the frozen training lifecycle, with no private API."""
    baseline_train.validate_official_batch_contract(config)
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
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_hash = ablation_source_tree_hash()
    generator = torch.Generator().manual_seed(config.seed)
    loaders = create_training_dataloaders(
        train_csv,
        public_csv,
        config.batch_size,
        config.num_workers,
        seed=config.seed,
        generator=generator,
    )
    fixed_indices, fixed_images = baseline_train.build_fixed_routing_diagnostic_batch(
        loaders["train"].dataset
    )
    if fixed_indices != baseline_train.FIXED_ROUTING_DIAGNOSTIC_INDICES:
        raise RuntimeError("Fixed routing diagnostic indices changed unexpectedly")
    model = _model_factory(config).to(device)
    summary = model.model_summary(
        source_sha256=source_hash,
        source_git_commit=os.environ.get("MPG_FER_SOURCE_GIT_COMMIT"),
    )
    baseline_train._atomic_json_document(
        output / "ablation_model_summary.json", summary
    )
    optimizer = build_recipe_optimizer(model, config)
    scheduler = build_recipe_scheduler(optimizer, config)
    scaler = (
        torch.amp.GradScaler("cuda")
        if config.use_amp and torch.cuda.is_available()
        else None
    )
    ema = ModelEMA(model, config.ema_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    run_id = config.run_id or str(uuid.uuid4())
    start_epoch, global_step = 1, 0
    best_metrics = best_comparator = None
    best_epoch = None
    patience = 0
    history: list[dict[str, Any]] = []
    previous_routing_supports: dict[str, torch.Tensor] | None = None
    resumed_from = None
    if resume_path is not None:
        bundle = load_resume_bundle(
            resume_path,
            expected_sha256=resume_sha256,
            config=config,
            run_id=config.run_id,
            source_hash=source_hash,
        )
        restore_training_state(
            bundle,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader_generator=generator,
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
        _, previous_routing_supports = collect_ablation_routing_diagnostics(
            model, fixed_images, device
        )

    baseline_train._atomic_json_document(output / "config.json", asdict(config))
    segment_started = time.monotonic()
    segment_start_epoch = start_epoch
    status = "TRAINING_COMPLETED"
    resume_digest = ""
    end_epoch = start_epoch - 1
    checkpoint_path = output / "best_val_acc.pt"

    for epoch in range(start_epoch, config.max_epochs + 1):
        loaders["train"].dataset.set_epoch(epoch)
        loaders["train"].sampler.set_epoch(epoch)
        lr = scheduler.step(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.monotonic()
        try:
            stats, global_step = train_ablation_one_epoch(
                model,
                loaders["train"],
                optimizer,
                device,
                scaler,
                config,
                criterion,
                ema=ema,
                epoch=epoch,
                global_optimizer_step=global_step,
            )
            ema.module.set_epoch_temperature(epoch)
            validation = evaluate_raw_and_tta(
                ema.module, loaders["val"], device, config.use_amp
            )
            tta = validation["tta"]
            improved = baseline_train.is_better_checkpoint(tta, best_comparator)
            routing, current_supports = collect_ablation_routing_diagnostics(
                model,
                fixed_images,
                device,
                previous_supports=previous_routing_supports,
            )
        except Exception:
            latest = output / "resume_latest.pt"
            failed_sha = sha256_file(latest) if latest.exists() else ""
            baseline_train._segment_manifest(
                output,
                run_id=run_id,
                segment_number=config.segment_number,
                start_epoch=segment_start_epoch,
                end_epoch=epoch - 1,
                next_epoch=epoch,
                wallclock=time.monotonic() - segment_started,
                resume_sha=failed_sha,
                best_epoch=best_epoch,
                best_metrics=best_comparator,
                status="FAILED",
            )
            raise
        duration = time.monotonic() - epoch_started
        if improved:
            best_comparator = dict(tta)
            best_metrics = {"raw": validation["raw"], "tta": tta}
            best_epoch = epoch
            baseline_train._save_best_ema(
                checkpoint_path,
                ema,
                epoch,
                best_metrics,
                config,
                source_hash,
            )
        patience = baseline_train.update_early_stop_patience(
            epoch, improved, patience, config
        )
        previous_routing_supports = current_supports
        entry = {
            "epoch": epoch,
            "lr": lr,
            "epoch_duration_sec": duration,
            "val_raw": validation["raw"],
            "val_tta": tta,
            "global_optimizer_step": global_step,
            "early_stop_monitor_active": epoch >= config.early_stop_monitor_start_epoch,
            "early_stop_patience": patience,
            "routing_diagnostics": routing,
            "diagnostic_applicability": model.diagnostic_applicability,
            "loss_applicability": model.loss_applicability,
            "gpu_peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else 0.0
            ),
            "gpu_peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / 2**20
                if device.type == "cuda"
                else 0.0
            ),
            **stats,
        }
        history.append(entry)
        end_epoch = epoch
        baseline_train._write_history(history, output)
        _write_ablation_routing_diagnostics(history, output, best_epoch)

        consistency_state = {
            "algorithm": "stateless_seed_epoch_group_v1",
            "last_completed_epoch": epoch,
            "selected_groups": stats["consistency_groups"],
        }
        bundle = build_resume_bundle(
            config=config,
            run_id=run_id,
            source_hash=source_hash,
            completed_epoch=epoch,
            global_optimizer_step=global_step,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_comparator_state=best_comparator,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            early_stop_counter=patience,
            history=history,
            loader_generator=generator,
            consistency_state=consistency_state,
            sampler_state=loaders["train"].sampler.state_dict(),
            augmentation_state=loaders["train"].dataset.state_dict(),
        )
        latest, resume_digest = atomic_save_resume(bundle, output)
        save_periodic_snapshot(
            latest,
            epoch,
            config.resume_snapshot_interval,
            config.resume_snapshots_to_keep,
        )
        if baseline_train.should_early_stop(epoch, patience, config):
            break
        elapsed = time.monotonic() - segment_started
        estimate = max(item["epoch_duration_sec"] for item in history[-3:])
        if epoch < config.max_epochs and baseline_train.should_end_segment(
            elapsed, estimate, config
        ):
            status = "NEEDS_RESUME"
            _, resume_digest = atomic_save_resume(bundle, output, status=status)
            break

    elapsed = time.monotonic() - segment_started
    if end_epoch >= segment_start_epoch and status == "TRAINING_COMPLETED":
        bundle = build_resume_bundle(
            config=config,
            run_id=run_id,
            source_hash=source_hash,
            completed_epoch=end_epoch,
            global_optimizer_step=global_step,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_comparator_state=best_comparator,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            early_stop_counter=patience,
            history=history,
            loader_generator=generator,
            consistency_state={
                "algorithm": "stateless_seed_epoch_group_v1",
                "last_completed_epoch": end_epoch,
            },
            sampler_state=loaders["train"].sampler.state_dict(),
            augmentation_state=loaders["train"].dataset.state_dict(),
        )
        _, resume_digest = atomic_save_resume(bundle, output, status=status)
    segment = baseline_train._segment_manifest(
        output,
        run_id=run_id,
        segment_number=config.segment_number,
        start_epoch=segment_start_epoch,
        end_epoch=end_epoch,
        next_epoch=end_epoch + 1,
        wallclock=elapsed,
        resume_sha=resume_digest,
        best_epoch=best_epoch,
        best_metrics=best_comparator,
        status=status,
    )
    result = {
        "run_id": run_id,
        "ablation_mode": config.ablation_mode,
        "source_hash": source_hash,
        "status": status,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_checkpoint": (
            str(checkpoint_path.resolve()) if checkpoint_path.exists() else None
        ),
        "best_checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path.exists() else None
        ),
        "resume_checkpoint": str((output / "resume_latest.pt").resolve()),
        "resume_sha256": resume_digest,
        "resumed_from": resumed_from,
        "segment": segment,
        "PRIVATE_EVALUATED": False,
    }
    baseline_train._atomic_json_document(output / "execution_manifest.json", result)
    return result


def run_ablation_training(
    train_csv: str | Path,
    public_csv: str | Path,
    private_csv: str | Path,
    output_dir: str | Path,
    preflight_result: dict[str, Any],
    *,
    config: AblationConfig,
    final_recipe_lock: str | Path,
    design_lock: str | Path,
    resume_path: str | Path | None = None,
    resume_sha256: str | None = None,
) -> dict[str, Any]:
    """Train/select on Train/PublicTest, then report PrivateTest exactly once."""
    recipe = validate_final_recipe_lock(final_recipe_lock, design_lock, config)
    # This validates path roles only. PrivateTest content is not opened here.
    train_path, public_path, private_path = validate_ablation_data_paths(
        train_csv, public_csv, private_csv
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_sha = ablation_source_tree_hash()
    write_ablation_manifest(
        output,
        config,
        source_sha256=source_sha,
        recipe_sha256=recipe["sha256"],
    )
    result = _run_ablation_training_core(
        train_path,
        public_path,
        output,
        preflight_result,
        config=config,
        resume_path=resume_path,
        resume_sha256=resume_sha256,
    )
    if result["status"] != "TRAINING_COMPLETED":
        return result

    checkpoint = Path(result["best_checkpoint"])
    if sha256_file(checkpoint) != result["best_checkpoint_sha256"]:
        raise RuntimeError("Frozen ablation checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = _model_factory(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    loaders = create_training_dataloaders(
        train_path,
        public_path,
        config.batch_size,
        config.num_workers,
        seed=config.seed,
    )
    canonical = evaluate_canonical_public_fp32(
        model,
        loaders["val"],
        device,
        dataset_role="PublicTest",
        source_path=public_path,
    )
    canonical.update(
        {
            "ablation_mode": config.ablation_mode,
            "seed": config.seed,
            "weights_type": "EMA",
            "selected_epoch": payload["epoch"],
            "checkpoint_sha256": result["best_checkpoint_sha256"],
        }
    )
    write_json(output / "canonical_public_metrics.json", canonical)
    result["canonical_public_metrics"] = canonical
    execution = output / "execution_manifest.json"
    manifest = json.loads(execution.read_text(encoding="utf-8"))
    manifest.update(
        {
            "ablation_mode": config.ablation_mode,
            "final_recipe_lock_sha256": recipe["sha256"],
            "canonical_public_metrics_sha256": sha256_file(
                output / "canonical_public_metrics.json"
            ),
            "PRIVATE_EVALUATED": False,
            "private_test_selection_or_training_use": False,
        }
    )
    write_json(execution, manifest)
    private_metrics = evaluate_ablation_private_once(
        private_path,
        output,
        config=config,
        expected_checkpoint_sha256=result["best_checkpoint_sha256"],
    )
    result["canonical_private_metrics"] = private_metrics
    return result


def evaluate_ablation_private_once(
    private_csv: str | Path,
    output_dir: str | Path,
    *,
    config: AblationConfig,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Open PrivateTest only after completion and checkpoint SHA freeze."""
    output = Path(output_dir)
    execution = output / "execution_manifest.json"
    manifest = json.loads(execution.read_text(encoding="utf-8"))
    if manifest.get("status") != "TRAINING_COMPLETED":
        raise RuntimeError("FINAL_TEST_REFUSED: training is not complete")
    if manifest.get("PRIVATE_EVALUATED"):
        raise RuntimeError("FINAL_TEST_REFUSED: PrivateTest one-shot already recorded")
    public_metrics_path = output / "canonical_public_metrics.json"
    if not public_metrics_path.is_file():
        raise RuntimeError("FINAL_TEST_REFUSED: frozen PublicTest report is absent")
    checkpoint = Path(manifest["best_checkpoint"])
    checkpoint_sha = sha256_file(checkpoint)
    if (
        checkpoint_sha != expected_checkpoint_sha256
        or checkpoint_sha != manifest.get("best_checkpoint_sha256")
    ):
        raise RuntimeError("FINAL_TEST_REFUSED: frozen checkpoint SHA-256 mismatch")

    # This is the first operation that constructs a dataset from test.csv.
    loader = create_private_dataloader(
        private_csv, config.batch_size, config.num_workers
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = _model_factory(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    canonical = evaluate_canonical_private_fp32(
        model,
        loader,
        device,
        dataset_role="PrivateTest",
        source_path=private_csv,
    )
    canonical.update(
        {
            "ablation_mode": config.ablation_mode,
            "seed": config.seed,
            "weights_type": "EMA",
            "selected_epoch": payload["epoch"],
            "checkpoint_sha256": checkpoint_sha,
            "evaluated_only_after_training_completed": True,
            "evaluated_only_after_checkpoint_sha_freeze": True,
        }
    )
    private_metrics_path = write_json(
        output / "canonical_private_metrics.json", canonical
    )
    manifest.update(
        {
            "canonical_private_metrics_sha256": sha256_file(private_metrics_path),
            "private_test_selection_or_training_use": False,
            "private_evaluated_only_after_freeze": True,
            "PRIVATE_EVALUATED": True,
        }
    )
    write_json(execution, manifest)
    write_checksums(output)
    return canonical
