"""Canonical O1 runner with an operational epoch-65 stop outside config identity."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW

from .protocol import (
    O1_REGISTRY,
    SCREEN_STOP_EPOCH,
    o1_source_tree_hash,
    resolve_o1_config,
    right_censor_decision,
    validate_design_lock,
    validate_train_public_paths,
    verify_single_delta,
    write_hpo_manifest,
    write_json,
    write_resolved_config,
    write_run_checksums,
)
from mpg_fer_v2_3 import train as baseline_train
from mpg_fer_v2_3.checkpoint import (
    atomic_save_resume,
    build_resume_bundle,
    config_hash,
    load_resume_bundle,
    restore_training_state,
    save_periodic_snapshot,
    sha256_file,
)
from mpg_fer_v2_3.data import create_training_dataloaders
from mpg_fer_v2_3.ema import ModelEMA
from mpg_fer_v2_3.evaluate import evaluate_raw_and_tta
from mpg_fer_v2_3.model import MPGFER
from mpg_fer_v2_3.utils import set_seed


def _resume_bundle(
    *,
    config,
    run_id,
    source_hash,
    epoch,
    global_step,
    model,
    ema,
    optimizer,
    scheduler,
    scaler,
    best_comparator,
    best_epoch,
    best_metrics,
    patience,
    history,
    generator,
    loaders,
    selected_groups,
):
    return build_resume_bundle(
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
        consistency_state={
            "algorithm": "stateless_seed_epoch_group_v1",
            "last_completed_epoch": epoch,
            "selected_groups": selected_groups,
        },
        sampler_state=loaders["train"].sampler.state_dict(),
        augmentation_state=loaders["train"].dataset.state_dict(),
    )


def run_o1_training(
    train_csv: str | Path,
    public_csv: str | Path,
    output_dir: str | Path,
    preflight_result: dict[str, Any],
    *,
    config_id: str,
    design_lock: str | Path,
    notebook_sha256: str,
    run_id: str,
    resume_path: str | Path | None = None,
    resume_sha256: str | None = None,
    segment_number: int = 1,
) -> dict[str, Any]:
    """Run one reviewed O1 config through epoch 65; no PrivateTest API exists."""
    if config_id not in O1_REGISTRY:
        raise RuntimeError("O1_EXECUTION_REFUSED: selector not in immutable registry")
    source_hash = o1_source_tree_hash()
    validate_design_lock(
        design_lock,
        expected_source_sha256=source_hash,
        expected_notebook_sha256=notebook_sha256,
    )
    config = resolve_o1_config(
        config_id,
        run_id=run_id,
        output_dir=str(output_dir),
        resume_path=None if resume_path is None else str(resume_path),
        segment_number=segment_number,
    )
    verify_single_delta(config_id, config)
    baseline_train.validate_official_batch_contract(config)
    fresh_gate = (
        preflight_result.get("passed")
        and preflight_result.get("samples") == config.micro_overfit_samples
    )
    resume_gate = resume_path is not None and preflight_result.get(
        "resume_compatibility_passed"
    )
    if not (fresh_gate or resume_gate):
        raise RuntimeError("O1 execution refused: real-16 or exact-resume gate required")
    train_path, public_path = validate_train_public_paths(train_csv, public_csv)

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(config.seed)
    loaders = create_training_dataloaders(
        train_path,
        public_path,
        config.batch_size,
        config.num_workers,
        seed=config.seed,
        generator=generator,
    )
    fixed_indices, fixed_images = baseline_train.build_fixed_routing_diagnostic_batch(
        loaders["train"].dataset
    )
    if fixed_indices != baseline_train.FIXED_ROUTING_DIAGNOSTIC_INDICES:
        raise RuntimeError("O1 fixed routing diagnostic identity mismatch")
    model = MPGFER(config).to(device)
    write_json(
        output / "o1_model_summary.json",
        model.model_summary(
            source_sha256=source_hash,
            source_git_commit=os.environ.get("MPG_FER_SOURCE_GIT_COMMIT"),
        ),
    )
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = baseline_train.WarmupCosineScheduler(
        optimizer,
        config.learning_rate,
        config.warmup_epochs,
        config.lr_decay_end_epoch,
        config.max_epochs,
        config.min_learning_rate,
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if config.use_amp and torch.cuda.is_available()
        else None
    )
    ema = ModelEMA(model, config.ema_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    start_epoch, global_step = 1, 0
    best_metrics = best_comparator = None
    best_epoch = None
    patience = 0
    history: list[dict[str, Any]] = []
    previous_supports = None
    resumed_from = None
    if resume_path is not None:
        bundle = load_resume_bundle(
            resume_path,
            expected_sha256=resume_sha256,
            config=config,
            run_id=run_id,
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
        start_epoch = int(bundle["next_epoch"])
        global_step = int(bundle["global_optimizer_step"])
        best_comparator = bundle["best_comparator_state"]
        best_metrics = bundle["best_metrics"]
        best_epoch = bundle["best_epoch"]
        patience = int(bundle["early_stop_counter"])
        history = list(bundle["history"])
        resumed_from = str(Path(resume_path).resolve())
        _, previous_supports = baseline_train.collect_fixed_routing_diagnostics(
            model, fixed_images, device
        )
    if start_epoch > SCREEN_STOP_EPOCH:
        raise RuntimeError("O1 resume is already beyond the operational screen limit")

    write_resolved_config(output / "resolved_config.json", config)
    write_hpo_manifest(
        output,
        config_id=config_id,
        config=config,
        source_sha256=source_hash,
        notebook_sha256=notebook_sha256,
    )
    segment_started = time.monotonic()
    segment_start = start_epoch
    status = "SCREENING_COMPLETED"
    resume_digest = ""
    end_epoch = start_epoch - 1
    checkpoint_path = output / "best_val_acc.pt"

    for epoch in range(start_epoch, SCREEN_STOP_EPOCH + 1):
        loaders["train"].dataset.set_epoch(epoch)
        loaders["train"].sampler.set_epoch(epoch)
        lr = scheduler.step(epoch)
        epoch_started = time.monotonic()
        try:
            stats, global_step = baseline_train.train_one_epoch(
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
            routing, current_supports = baseline_train.collect_fixed_routing_diagnostics(
                model,
                fixed_images,
                device,
                previous_supports=previous_supports,
            )
        except Exception:
            latest = output / "resume_latest.pt"
            baseline_train._segment_manifest(
                output,
                run_id=run_id,
                segment_number=segment_number,
                start_epoch=segment_start,
                end_epoch=epoch - 1,
                next_epoch=epoch,
                wallclock=time.monotonic() - segment_started,
                resume_sha=sha256_file(latest) if latest.exists() else "",
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
                checkpoint_path, ema, epoch, best_metrics, config, source_hash
            )
        patience = baseline_train.update_early_stop_patience(
            epoch, improved, patience, config
        )
        previous_supports = current_supports
        entry = {
            "epoch": epoch,
            "lr": lr,
            "epoch_duration_sec": duration,
            "val_raw": validation["raw"],
            "val_tta": tta,
            "global_optimizer_step": global_step,
            "early_stop_monitor_active": epoch
            >= config.early_stop_monitor_start_epoch,
            "early_stop_patience": patience,
            "routing_diagnostics": routing,
            **stats,
        }
        history.append(entry)
        end_epoch = epoch
        baseline_train._write_history(history, output)
        baseline_train._write_routing_diagnostics(history, output, best_epoch)
        bundle = _resume_bundle(
            config=config,
            run_id=run_id,
            source_hash=source_hash,
            epoch=epoch,
            global_step=global_step,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_comparator=best_comparator,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            patience=patience,
            history=history,
            generator=generator,
            loaders=loaders,
            selected_groups=stats["consistency_groups"],
        )
        latest, resume_digest = atomic_save_resume(bundle, output)
        save_periodic_snapshot(
            latest,
            epoch,
            config.resume_snapshot_interval,
            config.resume_snapshots_to_keep,
        )
        elapsed = time.monotonic() - segment_started
        estimate = max(item["epoch_duration_sec"] for item in history[-3:])
        if epoch < SCREEN_STOP_EPOCH and baseline_train.should_end_segment(
            elapsed, estimate, config
        ):
            status = "NEEDS_RESUME"
            _, resume_digest = atomic_save_resume(bundle, output, status=status)
            break

    elapsed = time.monotonic() - segment_started
    if status == "SCREENING_COMPLETED" and end_epoch == SCREEN_STOP_EPOCH:
        _, resume_digest = atomic_save_resume(
            bundle, output, status="SCREENING_COMPLETED"
        )
    segment = baseline_train._segment_manifest(
        output,
        run_id=run_id,
        segment_number=segment_number,
        start_epoch=segment_start,
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
        "config_id": config_id,
        "source_sha256": source_hash,
        "scientific_config_sha256": config_hash(config),
        "status": status,
        "screen_stop_epoch": SCREEN_STOP_EPOCH,
        "scientific_max_epochs": config.max_epochs,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_checkpoint": str(checkpoint_path.resolve())
        if checkpoint_path.exists()
        else None,
        "best_checkpoint_sha256": sha256_file(checkpoint_path)
        if checkpoint_path.exists()
        else None,
        "resume_checkpoint": str((output / "resume_latest.pt").resolve()),
        "resume_sha256": resume_digest,
        "resumed_from": resumed_from,
        "segment": segment,
        "PRIVATE_EVALUATED": False,
    }
    if status == "SCREENING_COMPLETED":
        if end_epoch != SCREEN_STOP_EPOCH or best_metrics is None or best_epoch is None:
            raise RuntimeError("O1 screen completion artifact state is inconsistent")
        comparator_trajectory = [
            {"epoch": item["epoch"], **item["val_tta"]} for item in history
        ]
        censor = right_censor_decision(
            selected_epoch=best_epoch,
            comparator_trajectory=comparator_trajectory,
        )
        write_json(
            output / "selected_public_metrics.json",
            {
                "schema_version": 1,
                "config_id": config_id,
                "selected_epoch": best_epoch,
                "selected_learning_rate": history[best_epoch - 1]["lr"],
                "public": best_metrics,
                "right_censor": censor,
                "checkpoint_sha256": result["best_checkpoint_sha256"],
                "PRIVATE_EVALUATED": False,
            },
        )
    write_json(output / "execution_manifest.json", result)
    if status == "SCREENING_COMPLETED":
        write_run_checksums(output)
    return result
