"""Generate the source-locked fresh/resume MPG-FER v2 Kaggle notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "mpg_fer_v2"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "MPG_FER_v2_Kaggle_T4.ipynb"
SOURCE_FILES = (
    "config.py", "features.py", "graph.py", "losses.py", "motif.py",
    "model.py", "ema.py", "checkpoint.py", "data.py", "evaluate.py",
    "utils.py", "train.py", "kaggle.py", "__init__.py",
)


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def _code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(True),
    }


def build_notebook() -> dict:
    sources = {name: (SOURCE_ROOT / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
    hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in sources.items()}
    bootstrap = f'''# Generated from reviewed Issue #92 sources. Do not hand-edit embedded code.
import sys
from pathlib import Path

EMBEDDED_SOURCES = {sources!r}
EMBEDDED_ROOT = Path("/kaggle/working/mpg_fer_v2_embedded")
EMBEDDED_PACKAGE = EMBEDDED_ROOT / "mpg_fer_v2"
EMBEDDED_PACKAGE.mkdir(parents=True, exist_ok=True)
for relative_name, source_text in EMBEDDED_SOURCES.items():
    (EMBEDDED_PACKAGE / relative_name).write_text(source_text, encoding="utf-8")
sys.path.insert(0, str(EMBEDDED_ROOT))
print(f"Materialized reviewed MPG-FER v2 package at {{EMBEDDED_PACKAGE}}")
'''
    cells = [
        _markdown(
            "# MPG-FER v2 — official segmented Kaggle T4 execution\n\n"
            "Generated from Issue #92 reviewed sources. The scientific package is "
            "source-locked; this notebook adds only runtime gates, resume verification, "
            "and artifact finalization.\n"
        ),
        _code(
            '''# STAGING TOOL EDITS ONLY THESE EXECUTION VALUES.
RESUME_MODE = "fresh"       # "fresh", "auto", or "required"
RESUME_PATH = None           # optional explicit /kaggle/input/.../resume_latest.pt
SEGMENT_NUMBER = 1
OUTPUT_DIR = "/kaggle/working/mpg_fer_v2_run"
ACCOUNT = "irthn1311"
KERNEL_REF = "irthn1311/mpg-fer-v2-final-t4-2026-09-21"
GIT_COMMIT_SHA = "SET_BY_STAGING"
EXPECTED_SOURCE_SHA = "f9a06cd4f7482c6a6e37d58844c1a30022602e9fc825eff73240f71740b0af1c"
EXPECTED_PARAMETERS = 2_238_609
'''
        ),
        _code(bootstrap),
        _code(
            '''# Environment, source/model/data gates, and explicit v2-only resume resolution.
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
import torchvision.transforms.functional as TF

from mpg_fer_v2.checkpoint import config_hash, load_resume_bundle, sha256_file
from mpg_fer_v2.config import MPGConfig
from mpg_fer_v2.data import FER2013Dataset, validate_dataset_gate
from mpg_fer_v2.ema import ModelEMA
from mpg_fer_v2.kaggle import resolve_kaggle_splits, resolve_resume_artifact
from mpg_fer_v2.model import MPGFER
from mpg_fer_v2.train import (
    compute_training_loss, evaluate_private_once, run_micro_overfit_preflight,
    run_training, source_tree_hash,
)
from mpg_fer_v2.utils import set_seed

def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
cfg = MPGConfig(segment_number=SEGMENT_NUMBER, output_dir=str(output_dir), resume_path=RESUME_PATH)
cuda_available = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_available else "NONE"
environment = {
    "torch_cuda_available": cuda_available,
    "gpu_name": gpu_name,
    "cuda_version": torch.version.cuda,
    "torch_version": torch.__version__,
}
print(json.dumps(environment, indent=2))
if not cuda_available or "t4" not in gpu_name.lower():
    atomic_json(output_dir / "preflight_report.json", {
        "status": "WRONG_GPU", "environment": environment,
        "kernel_ref": KERNEL_REF, "segment_number": SEGMENT_NUMBER,
    })
    raise RuntimeError(f"WRONG_GPU: expected NVIDIA Tesla T4, got {gpu_name}")
device = torch.device("cuda")

reviewed_source_sha = source_tree_hash()
if reviewed_source_sha != EXPECTED_SOURCE_SHA:
    raise RuntimeError(
        f"SOURCE_HASH_MISMATCH: expected {EXPECTED_SOURCE_SHA}, got {reviewed_source_sha}"
    )
model_for_count = MPGFER(cfg)
parameter_count = sum(p.numel() for p in model_for_count.parameters() if p.requires_grad)
del model_for_count
if parameter_count != EXPECTED_PARAMETERS:
    raise RuntimeError(
        f"PARAMETER_COUNT_MISMATCH: expected {EXPECTED_PARAMETERS}, got {parameter_count}"
    )

train_csv, val_csv, test_csv = resolve_kaggle_splits()
dataset_gate = validate_dataset_gate(train_csv, val_csv, test_csv)
resume_checkpoint, resume_sha = resolve_resume_artifact(RESUME_MODE, RESUME_PATH)

# The only allowed scientific fallback is selected before the fresh official run.
# A resumed bundle declares which of the two preregistered batch/accumulation pairs it used.
if resume_checkpoint is not None:
    inspected = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
    resumed_scientific = inspected.get("scientific_config", {})
    resumed_pair = (
        int(resumed_scientific.get("batch_size", cfg.batch_size)),
        int(resumed_scientific.get("gradient_accumulation_steps", cfg.gradient_accumulation_steps)),
    )
    if resumed_pair == (8, 4):
        cfg.batch_size, cfg.gradient_accumulation_steps = resumed_pair
    elif resumed_pair != (16, 2):
        raise RuntimeError(f"Forbidden resumed batch/accumulation pair: {resumed_pair}")
    del inspected

print(json.dumps({
    "account": ACCOUNT, "kernel_ref": KERNEL_REF, "git_commit": GIT_COMMIT_SHA,
    "segment_number": SEGMENT_NUMBER, "source_sha256": reviewed_source_sha,
    "parameter_count": parameter_count, "dataset": dataset_gate,
    "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
}, indent=2))
'''
        ),
        _code(
            '''# Fresh T4 preflight or strict resume-state verification.
def run_execution_bounded_preflight(train_path, config, device):
    set_seed(config.seed)
    dataset = FER2013Dataset(train_path, split="train", augment=False)
    images = torch.stack([dataset[index][0] for index in range(config.batch_size)]).to(device)
    targets = torch.tensor(
        [dataset[index][1] for index in range(config.batch_size)],
        dtype=torch.long, device=device,
    )
    del dataset
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = MPGFER(config).to(device).train()
    ema = ModelEMA(model, config.ema_decay)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    finite = {}
    hooks = []

    def finite_hook(name):
        def record(_module, _inputs, output):
            finite[name] = bool(torch.isfinite(output).all().item())
        return record

    hooks.append(model.pixel_extractor.register_forward_hook(finite_hook("features")))
    for index, layer in enumerate(model.pixel_gnn):
        hooks.append(layer.attn_dropout.register_forward_hook(finite_hook(f"pixel_attention_{index}")))
    for index, layer in enumerate(model.motif_gnn):
        hooks.append(layer.attn_dropout.register_forward_hook(finite_hook(f"motif_attention_{index}")))

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        logits, outputs = model(images)
        flipped_logits, flipped_outputs = model(TF.hflip(images))
        loss, components = compute_training_loss(
            logits, outputs, targets, criterion, config, flipped_logits
        )
    tensors = {
        "final_logits": logits, "flipped_logits": flipped_logits,
        "motif_assignments": outputs["motif_assignments"],
        "motif_geometry": outputs["motif_geometry"],
        "temperature": outputs["tau"], "mi_loss": outputs["loss_mi"],
        "consistency_loss": components["weighted_consistency"], "total_loss": loss,
    }
    finite.update({name: bool(torch.isfinite(value).all().item()) for name, value in tensors.items()})
    if not all(finite.values()):
        raise FloatingPointError(f"Non-finite T4 preflight tensor: {finite}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    audited = {
        "pixel_projection": model.pixel_proj[0].weight,
        "pixel_gnn": model.pixel_gnn[0].q_proj.weight,
        "raw_tau": model.motif_composer.raw_tau,
        "assignment_query": model.motif_composer.assignment_query.weight,
        "prototypes": model.motif_composer.prototypes,
        "scale_gate": model.motif_composer.scale_gate.weight,
        "scale_saliency_8": model.motif_composer.scale_saliency["8"].weight,
        "motif_geometry_attention": model.motif_gnn[0].geom_proj.weight,
        "classifier": model.classifier[-1].weight,
    }
    gradients = {
        name: bool(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all().item()
            and torch.any(parameter.grad != 0).item()
        )
        for name, parameter in audited.items()
    }
    if not all(gradients.values()):
        raise FloatingPointError(f"Invalid T4 preflight gradient: {gradients}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer_step_succeeded = scaler.get_scale() >= old_scale
    if not optimizer_step_succeeded:
        raise FloatingPointError("GradScaler skipped the bounded optimizer step")
    ema.update(model)
    torch.cuda.synchronize(device)
    result = {
        "passed": True,
        "physical_batch": config.batch_size,
        "gradient_accumulation": config.gradient_accumulation_steps,
        "effective_batch": config.batch_size * config.gradient_accumulation_steps,
        "amp_enabled": True,
        "worst_case_consistency_forward": True,
        "loss": float(loss.detach()),
        "loss_components": {name: float(value.detach()) for name, value in components.items()},
        "finite_audits": finite,
        "gradient_audits": gradients,
        "ema_updates": ema.num_updates,
        "motif_diagnostics": {
            name: float(outputs[name].detach())
            for name in (
                "tau", "H_local_raw", "H_local_normalized", "H_global_raw",
                "H_global_normalized", "L_MI", "mean_entropy",
                "effective_motif_count", "min_utilization", "max_utilization",
                "std_utilization", "mean_top1_probability", "mean_top2_probability",
                "mean_top1_top2_margin", "mean_offdiag_prototype_cosine",
                "mean_alpha_8", "mean_alpha_12", "mean_alpha_16",
            )
        },
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    for hook in hooks:
        hook.remove()
    del model, ema, optimizer, scaler, images, targets, logits, flipped_logits, outputs, flipped_outputs, loss
    torch.cuda.empty_cache()
    return result

resume_verification = None
if resume_checkpoint is None:
    try:
        bounded = run_execution_bounded_preflight(train_csv, cfg, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        cfg.batch_size = 8
        cfg.gradient_accumulation_steps = 4
        bounded = run_execution_bounded_preflight(train_csv, cfg, device)
        bounded["oom_fallback_applied"] = True
    micro = run_micro_overfit_preflight(
        train_csv, cfg, device=device, raise_on_failure=False
    )
    if not micro["passed"]:
        preflight = {
            "status": "PREFLIGHT_FAILED", "passed": False,
            "environment": environment, "dataset": dataset_gate,
            "bounded_gpu": bounded, "micro_overfit": micro,
        }
        atomic_json(output_dir / "preflight_report.json", preflight)
        raise RuntimeError(f"REAL_MICRO_OVERFIT_FAILED: {micro}")
    preflight = {
        **micro, "status": "PASS", "environment": environment,
        "dataset": dataset_gate, "parameter_count": parameter_count,
        "source_sha256": reviewed_source_sha, "bounded_gpu": bounded,
        "fresh_reset_ready": True,
    }
else:
    metadata = json.loads(resume_checkpoint.with_name("resume_latest.json").read_text(encoding="utf-8"))
    bundle = load_resume_bundle(
        resume_checkpoint, expected_sha256=resume_sha, config=cfg,
        run_id=metadata["run_id"], source_hash=reviewed_source_sha,
    )
    previous_segment_path = resume_checkpoint.with_name("segment_manifest.json")
    if not previous_segment_path.is_file():
        raise RuntimeError("Attached resume artifact lacks segment_manifest.json")
    previous_segment = json.loads(previous_segment_path.read_text(encoding="utf-8"))
    checks = {
        "run_id": bundle["run_id"] == metadata["run_id"] == previous_segment["run_id"],
        "source_hash": bundle["source_hash"] == metadata["source_hash"] == reviewed_source_sha,
        "config_hash": bundle["config_hash"] == metadata["config_hash"] == config_hash(cfg),
        "epoch_boundary": bundle["next_epoch"] == previous_segment["next_epoch"],
        "history_length": len(bundle["history"]) == bundle["completed_epoch"],
        "previous_status": previous_segment["status"] == "NEEDS_RESUME",
    }
    if not all(checks.values()):
        raise RuntimeError(f"RESUME_STATE_MISMATCH: {checks}")
    best_source = resume_checkpoint.with_name("best_val_acc.pt")
    best_metadata_path = resume_checkpoint.with_name("best_val_acc.json")
    if bundle["best_epoch"] is not None:
        if not best_source.is_file() or not best_metadata_path.is_file():
            raise RuntimeError("Resume artifact lacks globally best EMA checkpoint/metadata")
        best_metadata = json.loads(best_metadata_path.read_text(encoding="utf-8"))
        best_checks = {
            "sha256": sha256_file(best_source) == best_metadata["checkpoint_sha256"],
            "run_id": best_metadata["run_id"] == bundle["run_id"],
            "source_hash": best_metadata["source_hash"] == reviewed_source_sha,
            "config_hash": best_metadata["config_hash"] == bundle["config_hash"],
            "best_epoch": best_metadata["best_epoch"] == bundle["best_epoch"],
        }
        if not all(best_checks.values()):
            raise RuntimeError(f"BEST_CHECKPOINT_RESUME_MISMATCH: {best_checks}")
        shutil.copy2(best_source, output_dir / "best_val_acc.pt")
    raw_tau = bundle["model_state_dict"]["motif_composer.raw_tau"].item()
    restored_tau = cfg.tau_min + (cfg.tau_max - cfg.tau_min) / (1.0 + math.exp(-raw_tau))
    scaler_state = bundle["grad_scaler_state_dict"]
    resume_verification = {
        "checks": checks,
        "run_id": bundle["run_id"],
        "segment_number": SEGMENT_NUMBER,
        "source_hash": bundle["source_hash"],
        "scientific_config_hash": bundle["config_hash"],
        "completed_epoch": bundle["completed_epoch"],
        "next_epoch": bundle["next_epoch"],
        "restored_lr": bundle["optimizer_state_dict"]["param_groups"][0]["lr"],
        "restored_optimizer_step": bundle["global_optimizer_step"],
        "restored_grad_scaler_scale": scaler_state.get("scale"),
        "restored_ema_updates": bundle["ema_state_dict"]["num_updates"],
        "restored_tau": restored_tau,
        "best_epoch": bundle["best_epoch"],
        "best_public_tta_accuracy": None if bundle["best_comparator_state"] is None else bundle["best_comparator_state"]["accuracy"],
        "patience_counter": bundle["early_stop_counter"],
        "history_length": len(bundle["history"]),
    }
    print("RESUME_VERIFICATION")
    print(json.dumps(resume_verification, indent=2))
    preflight = {
        "status": "RESUME_VERIFIED", "resume_compatibility_passed": True,
        "resume_checkpoint_sha256": resume_sha, "resume_verification": resume_verification,
        "environment": environment, "dataset": dataset_gate,
        "parameter_count": parameter_count, "source_sha256": reviewed_source_sha,
    }
    del bundle

atomic_json(output_dir / "preflight_report.json", preflight)
print(json.dumps(preflight, indent=2))
'''
        ),
        _code(
            '''# Fresh official run or exact continuation from next_epoch.
manifest = run_training(
    train_csv=train_csv, val_csv=val_csv, output_dir=output_dir,
    preflight_result=preflight, config=cfg,
    resume_path=resume_checkpoint, resume_sha256=resume_sha,
)

# Complete the segment provenance contract and preserve the global best EMA checkpoint.
resume_metadata = json.loads((output_dir / "resume_latest.json").read_text(encoding="utf-8"))
segment_path = output_dir / "segment_manifest.json"
segment_manifest = json.loads(segment_path.read_text(encoding="utf-8"))
segment_manifest.update({
    "source_hash": resume_metadata["source_hash"],
    "config_hash": resume_metadata["config_hash"],
    "gpu": gpu_name,
    "kernel_ref": KERNEL_REF,
    "git_commit_sha": GIT_COMMIT_SHA,
    "parameter_count": parameter_count,
    "resume_verification": resume_verification,
})
atomic_json(segment_path, segment_manifest)

best_path = output_dir / "best_val_acc.pt"
if not best_path.is_file():
    raise RuntimeError("Global best EMA checkpoint is missing at segment end")
best_metadata = {
    "checkpoint_sha256": sha256_file(best_path),
    "weights_type": "EMA",
    "run_id": manifest["run_id"],
    "source_hash": resume_metadata["source_hash"],
    "config_hash": resume_metadata["config_hash"],
    "best_epoch": manifest["best_epoch"],
    "best_public_tta_accuracy": manifest["best_metrics"]["tta"]["accuracy"],
}
atomic_json(output_dir / "best_val_acc.json", best_metadata)
print(json.dumps({"manifest": manifest, "segment_manifest": segment_manifest,
                  "best_checkpoint": best_metadata}, indent=2))
'''
        ),
        _code(
            '''# Final metrics/artifacts only after TRAINING_COMPLETED; otherwise clean resume archive.
if manifest["status"] == "TRAINING_COMPLETED":
    private_metrics = evaluate_private_once(test_csv, output_dir, cfg)
    execution_manifest = json.loads((output_dir / "execution_manifest.json").read_text(encoding="utf-8"))
    public_metrics = execution_manifest["best_metrics"]
    atomic_json(output_dir / "public_metrics.json", public_metrics)

    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    best_entry = next(entry for entry in history if entry["epoch"] == execution_manifest["best_epoch"])
    motif_names = [
        "tau", "H_local_raw", "H_local_normalized", "H_global_raw",
        "H_global_normalized", "L_MI", "mean_entropy", "effective_motif_count",
        "min_utilization", "max_utilization", "std_utilization",
        "mean_top1_probability", "mean_top2_probability",
        "mean_top1_top2_margin", "mean_offdiag_prototype_cosine",
        "mean_alpha_8", "mean_alpha_12", "mean_alpha_16",
        "std_alpha_8", "std_alpha_12", "std_alpha_16",
    ]
    motif_diagnostics = {
        "best_epoch": execution_manifest["best_epoch"],
        "best_epoch_diagnostics": {name: best_entry[name] for name in motif_names},
        "final_epoch": history[-1]["epoch"],
        "final_epoch_diagnostics": {name: history[-1][name] for name in motif_names},
    }
    atomic_json(output_dir / "motif_diagnostics.json", motif_diagnostics)
    selection = {
        "run_id": execution_manifest["run_id"],
        "weights_type": "EMA",
        "best_epoch": execution_manifest["best_epoch"],
        "checkpoint_sha256": sha256_file(output_dir / "best_val_acc.pt"),
        "selection_metric": "EMA Public flip-TTA accuracy",
        "selection_tiebreak": ["higher EMA TTA macro-F1", "lower EMA TTA loss"],
        "private_evaluated_only_after_freeze": True,
    }
    atomic_json(output_dir / "final_selection_manifest.json", selection)

    import matplotlib.pyplot as plt
    def save_confusion(metrics, path, title):
        matrix = metrics["confusion_matrix"]
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(title=title, xlabel="Predicted", ylabel="True")
        axis.set_xticks(range(7)); axis.set_yticks(range(7))
        for row in range(7):
            for column in range(7):
                axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
        figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)

    save_confusion(public_metrics["tta"], output_dir / "public_confusion_matrix.png", "PublicTest EMA TTA")
    save_confusion(private_metrics["tta"], output_dir / "private_confusion_matrix.png", "PrivateTest EMA TTA")
    epochs = [entry["epoch"] for entry in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], label="train")
    axes[0].plot(epochs, [entry["val_tta"]["loss"] for entry in history], label="EMA Public TTA")
    axes[1].plot(epochs, [entry["train_accuracy"] for entry in history], label="train")
    axes[1].plot(epochs, [entry["val_tta"]["accuracy"] for entry in history], label="EMA Public TTA")
    for axis in axes:
        axis.legend(); axis.grid(alpha=0.2); axis.set_xlabel("Epoch")
    axes[0].set_title("Loss"); axes[1].set_title("Accuracy")
    figure.tight_layout(); figure.savefig(output_dir / "training_curves.png", dpi=160); plt.close(figure)
    print(json.dumps({"public": public_metrics, "private": private_metrics,
                      "motif": motif_diagnostics, "selection": selection}, indent=2))
else:
    print("Segment ended NEEDS_RESUME normally; PrivateTest remains unopened.")

archive = shutil.make_archive("/kaggle/working/mpg_fer_v2_artifacts", "zip", root_dir=output_dir)
print(f"Artifacts ZIP: {archive}")
'''
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "mpg_fer_v2_source_manifest": hashes,
            "mpg_fer_v2_issue": 92,
            "mpg_fer_notebook_generator": "tools/sync_notebook.py",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")
