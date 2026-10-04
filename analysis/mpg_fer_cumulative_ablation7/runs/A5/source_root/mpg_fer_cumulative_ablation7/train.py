"""Training and evaluation pipeline for the 7-configuration cumulative ablation ladder."""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any
import uuid

import torch
import torch.nn as nn
from torch.optim import AdamW
import torchvision.transforms.functional as TF

from .model import (
    CUMULATIVE_REGISTRY,
    CumulativeAblationMode,
    CumulativeAblationMPGFER,
)
from .protocol import (
    CANONICAL_DATASET_HASHES,
    CANONICAL_DATASET_ROWS,
    EXPECTED_A6_METRICS,
    sha256_file,
    validate_all_splits,
    validate_split_identity,
)
from mpg_fer_table_vi.protocol import (
    AblationConfig,
    evaluate_canonical_private_fp32,
    evaluate_canonical_public_fp32,
)
from mpg_fer_table_vi.train import (
    RecipeEpochScheduler,
    build_recipe_optimizer,
    build_recipe_scheduler,
)
from mpg_fer_v2_3.checkpoint import (
    atomic_save_resume,
    build_resume_bundle,
    load_resume_bundle,
    restore_training_state,
    save_periodic_snapshot,
)
from mpg_fer_v2_3.data import create_private_dataloader, create_training_dataloaders
from mpg_fer_v2_3.ema import ModelEMA
from mpg_fer_v2_3.evaluate import evaluate_raw_and_tta
from mpg_fer_v2_3.losses import supervised_contrastive_loss, symmetric_js_divergence
from mpg_fer_v2_3 import train as baseline_train
from mpg_fer_v2_3.utils import set_seed


def compute_cumulative_training_loss(
    logits: torch.Tensor,
    outputs: dict[str, Any],
    targets: torch.Tensor,
    criterion: nn.Module,
    config: AblationConfig,
    mode: CumulativeAblationMode,
    flipped_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    spec = CUMULATIVE_REGISTRY[mode]
    losses_spec = spec.applicable_losses

    weighted: dict[str, torch.Tensor | None] = {
        "ce_final": criterion(logits, targets),
        "weighted_ce_pixel": config.aux_pixel_weight * criterion(outputs["pixel_logits"], targets),
    }

    if losses_spec.get("motif_aux", 0.0) > 0.0:
        weighted["weighted_ce_motif"] = config.aux_motif_weight * criterion(outputs["motif_logits"], targets)
    else:
        weighted["weighted_ce_motif"] = None

    if losses_spec.get("lambda_div", 0.0) > 0.0 and outputs.get("loss_diversity") is not None:
        weighted["weighted_diversity"] = config.lambda_div * outputs["loss_diversity"]
    else:
        weighted["weighted_diversity"] = None

    if losses_spec.get("lambda_mi", 0.0) > 0.0 and outputs.get("loss_mi") is not None:
        weighted["weighted_mi"] = config.lambda_mi * outputs["loss_mi"]
    else:
        weighted["weighted_mi"] = None

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


def train_cumulative_one_epoch(
    model: CumulativeAblationMPGFER,
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
    if hasattr(model, "set_epoch_temperature"):
        model.set_epoch_temperature(epoch)
    if ema is not None and hasattr(ema.module, "set_epoch_temperature"):
        ema.module.set_epoch_temperature(epoch)

    optimizer.zero_grad(set_to_none=True)
    accum_steps = config.gradient_accumulation_steps
    use_amp = config.use_amp and torch.cuda.is_available()
    num_batches = len(dataloader)
    selected_groups: list[int] = []

    totals: dict[str, float | None] = {"train_loss": 0.0}
    correct = samples = 0

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
            loss, components = compute_cumulative_training_loss(
                logits, outputs, targets, criterion, config, model.cumulative_mode, flipped_logits
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
            elif totals.get(name) is not None:
                totals[name] += float(value.detach()) * batch_size
            elif name not in totals:
                totals[name] = float(value.detach()) * batch_size

        correct += int((logits.argmax(dim=-1) == targets).sum())
        samples += batch_size

    stats = {
        name: (None if value is None else value / samples)
        for name, value in totals.items()
    }
    stats["train_accuracy"] = correct / samples
    stats["consistency_groups"] = selected_groups
    return stats, global_optimizer_step


def run_cumulative_micro_overfit(
    train_path: Path | str,
    config: AblationConfig,
    mode: CumulativeAblationMode,
) -> dict[str, Any]:
    """16-example micro-overfit sanity check."""
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = CumulativeAblationMPGFER(config=config, mode=mode).to(device)
    optimizer = AdamW(model.parameters(), lr=config.micro_overfit_learning_rate)
    criterion = nn.CrossEntropyLoss()

    dataset = baseline_train.FER2013Dataset(train_path, augment=False)
    samples = [dataset[i] for i in range(min(config.micro_overfit_samples, len(dataset)))]
    images = torch.stack([s[0] for s in samples]).to(device)
    targets = torch.tensor([s[1] for s in samples], dtype=torch.long, device=device)

    model.train()
    target_accuracy = config.micro_overfit_target
    step = 0
    final_acc = 0.0
    for step in range(1, config.micro_overfit_max_steps + 1):
        optimizer.zero_grad()
        logits, outputs = model(images)
        loss, _ = compute_cumulative_training_loss(logits, outputs, targets, criterion, config, mode)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=-1)
        final_acc = float((preds == targets).float().mean().item())
        if final_acc >= target_accuracy:
            break

    passed = final_acc >= target_accuracy
    return {
        "passed": passed,
        "steps": step,
        "final_accuracy": final_acc,
        "target_accuracy": target_accuracy,
    }


def run_cumulative_training_job(
    mode: CumulativeAblationMode,
    run_id: str,
    train_csv: Path | str,
    val_csv: Path | str,
    test_csv: Path | str,
    output_dir: Path | str,
    *,
    config: AblationConfig | None = None,
) -> dict[str, Any]:
    """Execute complete end-to-end training and evaluation job for one cumulative mode."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Dataset verification: Fail closed on dataset identity
    split_audit = validate_all_splits(Path(train_csv), Path(val_csv), Path(test_csv))
    print("Dataset verification passed:", split_audit)

    cfg = config or AblationConfig()
    cfg.seed = 42
    cfg.optimizer_family = "AdamW"
    cfg.scheduler_family = "linear_warmup_cosine_then_floor"
    cfg.learning_rate = 0.0003
    cfg.weight_decay = 0.0005
    cfg.warmup_epochs = 5
    cfg.lr_decay_end_epoch = 85
    cfg.max_epochs = 120
    cfg.early_stop_monitor_start_epoch = 85
    cfg.early_stop_patience = 15
    cfg.ema_decay = 0.999
    cfg.batch_size = 16
    cfg.gradient_accumulation_steps = 2
    cfg.use_amp = True
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    spec = CUMULATIVE_REGISTRY[mode]
    print(f"Starting cumulative job {run_id} for {mode.value}: {spec.paper_name}")

    # Micro-overfit preflight
    preflight = run_cumulative_micro_overfit(train_csv, cfg, mode)
    print("Micro-overfit preflight result:", preflight)
    if not preflight["passed"]:
        raise RuntimeError(f"Micro-overfit preflight failed for {mode.value}: {preflight}")

    # Data loaders
    loaders = create_training_dataloaders(
        Path(train_csv),
        Path(val_csv),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    model = CumulativeAblationMPGFER(config=cfg, mode=mode).to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay)
    optimizer = build_recipe_optimizer(model, cfg)
    scheduler = build_recipe_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and torch.cuda.is_available())

    checkpoint_path = out / "best_val_acc.pt"
    best_comparator = {"accuracy": -1.0, "macro_f1": -1.0, "loss": float("inf")}
    best_metrics: dict[str, Any] = {}
    best_epoch = -1
    patience = cfg.early_stop_patience
    history: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.monotonic()
        stats, global_step = train_cumulative_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            scaler,
            cfg,
            criterion,
            ema=ema,
            epoch=epoch,
            global_optimizer_step=global_step,
        )
        lr = scheduler.step(epoch)

        # Validation on PublicTest using EMA module
        if hasattr(ema.module, "set_epoch_temperature"):
            ema.module.set_epoch_temperature(epoch)
        validation = evaluate_raw_and_tta(ema.module, loaders["val"], device, cfg.use_amp)
        tta = validation["tta"]
        improved = baseline_train.is_better_checkpoint(tta, best_comparator)

        if improved:
            best_comparator = dict(tta)
            best_metrics = {"raw": validation["raw"], "tta": tta}
            best_epoch = epoch
            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ema.module.state_dict(),
                    "best_metrics": best_metrics,
                    "config": asdict(cfg),
                    "mode": mode.value,
                },
                checkpoint_path,
            )

        patience = baseline_train.update_early_stop_patience(epoch, improved, patience, cfg)
        duration = time.monotonic() - t0

        entry = {
            "epoch": epoch,
            "lr": lr,
            "epoch_duration_sec": duration,
            "val_raw": validation["raw"],
            "val_tta": tta,
            "global_optimizer_step": global_step,
            "early_stop_patience": patience,
            **stats,
        }
        history.append(entry)

        # Print progress every 5 epochs or on improvement
        if epoch % 5 == 0 or improved or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Train Loss: {stats['train_loss']:.4f} | "
                f"Val TTA Acc: {tta['accuracy']*100:.2f}% (Best: {best_comparator['accuracy']*100:.2f}%) | "
                f"Patience: {patience}"
            )

        if baseline_train.should_early_stop(epoch, patience, cfg):
            print(f"Early stopping triggered at epoch {epoch}!")
            break

    # Save training history
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_tta_acc", "val_tta_f1"])
        for h in history:
            writer.writerow([
                h["epoch"], h["lr"], h["train_loss"], h["train_accuracy"],
                h["val_tta"]["accuracy"], h["val_tta"]["macro_f1"]
            ])

    # Checkpoint SHA256 freeze
    ckpt_sha = sha256_file(checkpoint_path)
    print(f"Selected Best Checkpoint Epoch {best_epoch} SHA256: {ckpt_sha}")

    # Canonical PublicTest Evaluation
    eval_model = CumulativeAblationMPGFER(config=cfg, mode=mode).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    eval_model.load_state_dict(payload["model_state_dict"], strict=True)

    public_metrics = evaluate_canonical_public_fp32(
        eval_model, loaders["val"], device, dataset_role="PublicTest", source_path=val_csv
    )
    public_metrics.update({
        "configuration": mode.value,
        "seed": cfg.seed,
        "weights_type": "EMA",
        "selected_epoch": best_epoch,
        "checkpoint_sha256": ckpt_sha,
    })
    with open(out / "canonical_public_metrics.json", "w", encoding="utf-8") as f:
        json.dump(public_metrics, f, indent=2)

    # Canonical PrivateTest Evaluation (evaluated ONCE after checkpoint freeze)
    private_loader = create_private_dataloader(
        Path(test_csv), batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    private_metrics = evaluate_canonical_private_fp32(
        eval_model, private_loader, device, source_path=test_csv
    )
    private_metrics.update({
        "configuration": mode.value,
        "seed": cfg.seed,
        "weights_type": "EMA",
        "selected_epoch": best_epoch,
        "checkpoint_sha256": ckpt_sha,
    })
    with open(out / "canonical_private_metrics.json", "w", encoding="utf-8") as f:
        json.dump(private_metrics, f, indent=2)

    # Execution manifest
    execution_manifest = {
        "run_id": run_id,
        "configuration": mode.value,
        "display_name": spec.paper_name,
        "status": "TRAINING_COMPLETED",
        "best_epoch": best_epoch,
        "checkpoint_sha256": ckpt_sha,
        "split_sha256": {
            "train.csv": CANONICAL_DATASET_HASHES["train.csv"],
            "val.csv": CANONICAL_DATASET_HASHES["val.csv"],
            "test.csv": CANONICAL_DATASET_HASHES["test.csv"],
        },
        "public_metrics": public_metrics,
        "private_metrics": private_metrics,
    }
    with open(out / "execution_manifest.json", "w", encoding="utf-8") as f:
        json.dump(execution_manifest, f, indent=2)

    # Checksums
    checksums = {}
    for p in sorted(out.glob("*")):
        if p.is_file() and p.name != "checksums.sha256":
            checksums[p.name] = sha256_file(p)
    with open(out / "checksums.sha256", "w", encoding="utf-8") as f:
        for fname, digest in sorted(checksums.items()):
            f.write(f"{digest}  {fname}\n")

    # Zip archive
    archive_base = out.parent / f"{run_id}-artifacts"
    archive_zip = shutil.make_archive(str(archive_base), "zip", root_dir=out)
    print(f"Artifact archive created at {archive_zip}")

    return execution_manifest
