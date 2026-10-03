"""Run Phase 0 and Stage 1 of the NPF conditional pixel-residual experiment."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np
import scipy
import sklearn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.functional as TF

from model import ResidualBranch, ResidualVariant
from protocol import (
    change_counts,
    disagreement,
    gate_summary,
    is_better,
    learning_rate_for_epoch,
    metrics,
    sha256_file,
    state_sha256,
    write_csv,
    write_json,
)


EXPECTED = {
    "train_sha256": "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8",
    "public_sha256": "412036d077c6ec203047b2935ab14bc858d8136ee26e8db3e23023f1fc9dee08",
    "private_sha256": "be385344b93606c75cb65cac3d39b13cdedb5b91621640ee08f73b9a551fbd9d",
    "config_sha256": "8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2",
    "full_checkpoint_sha256": "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e",
    "npf_checkpoint_sha256": "f301895cd174f8adf998d7f622279510e45bf210e2cc52ebc25376db49e5f972",
    "full_checkpoint_source": "1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87",
    "npf_checkpoint_source": "f1e11eab85361061065baa87906749c0a31b19e08e1b121bfd25aa5cd298d59c",
    "full_epoch": 57,
    "npf_epoch": 41,
}
VARIANTS = ("R1", "R2")
CLASS_NAMES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[4]
    checkpoint_repo = repo.parent if repo.name.startswith(".codex-") else repo
    staging = Path(r"D:\KaggleStaging\mpg-fer-ablation7-20261002")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--source-root", type=Path, default=repo / "research/mpg_fer_v2_3/src")
    parser.add_argument("--train-csv", type=Path, default=staging / "datasets/existing_profile_3/train.csv")
    parser.add_argument("--public-csv", type=Path, default=staging / "datasets/existing_profile_3/val.csv")
    parser.add_argument("--private-csv", type=Path, default=staging / "datasets/existing_profile_3/test.csv")
    parser.add_argument("--full-checkpoint", type=Path, default=checkpoint_repo / "research/mpg_fer_v2_3/official_runs/segment_02/mpg_fer_v2_3_run/best_val_acc.pt")
    parser.add_argument("--npf-checkpoint", type=Path, default=staging / "resume_segment2/verified_outputs/NO_PIXEL_FUSION/mpg-fer-table-vi/mpgfer-ablation-no-pixel-fusion-s42/best_val_acc.pt")
    parser.add_argument("--output", type=Path, default=staging / "analysis/npf_conditional_pixel_residual")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--backbone-microbatch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--resume-phase0", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(path: Path, mode: Any, expected_sha: str, expected_source: str, expected_epoch: int, device: torch.device, model_type: Any, config_type: Any, config_hash_fn: Any) -> tuple[torch.nn.Module, dict[str, Any], Any]:
    if sha256_file(path) != expected_sha:
        raise RuntimeError(f"checkpoint SHA mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["source_hash"] != expected_source or int(checkpoint["epoch"]) != expected_epoch:
        raise RuntimeError(f"checkpoint internal identity mismatch: {path}")
    allowed = {field.name for field in fields(config_type)}
    config = config_type(**{key: value for key, value in checkpoint["config"].items() if key in allowed})
    if config_hash_fn(config) != EXPECTED["config_sha256"]:
        raise RuntimeError("scientific config mismatch")
    model = model_type(config, mode=mode).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint load failed")
    model.eval().requires_grad_(False)
    return model, checkpoint, config


def extract_frozen_views(model: torch.nn.Module, loader: DataLoader, device: torch.device, *, readouts: bool) -> dict[str, Any]:
    model.eval()
    before = state_sha256(model)
    result: dict[str, list[np.ndarray]] = {"labels": [], "raw_logits": [], "flip_logits": []}
    if readouts:
        for name in ("pixel_raw", "pixel_flip", "motif_raw", "motif_flip"):
            result[name] = []
    prior_matmul = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn = torch.backends.cudnn.allow_tf32
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, dtype=torch.float32)
                result["labels"].append(labels.numpy())
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    raw, raw_outputs = model(images)
                    flip, flip_outputs = model(TF.hflip(images))
                result["raw_logits"].append(raw.float().cpu().numpy())
                result["flip_logits"].append(flip.float().cpu().numpy())
                if readouts:
                    result["pixel_raw"].append(raw_outputs["h_pixel_readout"].float().cpu().numpy())
                    result["pixel_flip"].append(flip_outputs["h_pixel_readout"].float().cpu().numpy())
                    result["motif_raw"].append(raw_outputs["h_motif_readout"].float().cpu().numpy())
                    result["motif_flip"].append(flip_outputs["h_motif_readout"].float().cpu().numpy())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul
        torch.backends.cudnn.allow_tf32 = prior_cudnn
    after = state_sha256(model)
    if before != after:
        raise RuntimeError("frozen model changed during feature extraction")
    arrays = {name: np.concatenate(parts) for name, parts in result.items()}
    arrays["tta_logits"] = (0.5 * (arrays["raw_logits"] + arrays["flip_logits"])).astype(np.float32)
    arrays["state_sha256_before"] = before
    arrays["state_sha256_after"] = after
    arrays["runtime_seconds"] = time.monotonic() - started
    return arrays


def phase0_report(labels: np.ndarray, full: dict[str, Any], npf: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result: dict[str, Any] = {"metrics": {}, "views": {}}
    sweep_rows = []
    for name, values in (("FULL", full), ("NO_PIXEL_FUSION", npf)):
        result["metrics"][name] = {view: metrics(labels, values[f"{view}_logits"]) for view in ("raw", "tta")}
    for view in ("raw", "tta"):
        full_logits, npf_logits = full[f"{view}_logits"], npf[f"{view}_logits"]
        full_prediction, npf_prediction = full_logits.argmax(axis=1), npf_logits.argmax(axis=1)
        rows = []
        for step in range(101):
            alpha = step / 100.0
            item = metrics(labels, alpha * full_logits + (1.0 - alpha) * npf_logits)
            row = {"view": view, "alpha_full": alpha, **item}
            rows.append(row)
            sweep_rows.append(row)
        maximum = max(row["accuracy"] for row in rows)
        result["views"][view] = {
            "disagreement": disagreement(labels, full_prediction, npf_prediction),
            "equal_logit_fusion": metrics(labels, 0.5 * (full_logits + npf_logits)),
            "scalar_fusion": {
                "label": "DESCRIPTIVE_IN_SAMPLE_FUSION_UPPER_DIAGNOSTIC",
                "maximum_accuracy": maximum,
                "tied_alpha_full": [row["alpha_full"] for row in rows if row["accuracy"] == maximum],
                "macro_f1_at_tied_alphas": [row["macro_f1"] for row in rows if row["accuracy"] == maximum],
            },
        }
    return result, sweep_rows


def residual_from_cached(branch: ResidualBranch, cached: dict[str, Any], device: torch.device, batch_size: int = 512) -> dict[str, Any]:
    branch.eval()
    output: dict[str, list[np.ndarray]] = {"raw_logits": [], "flip_logits": [], "raw_delta": [], "flip_delta": [], "raw_gate": [], "flip_gate": []}
    count = len(cached["labels"])
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            for view in ("raw", "flip"):
                base = torch.from_numpy(cached[f"{view}_logits"][start:stop]).to(device)
                pixel = torch.from_numpy(cached[f"pixel_{view}"][start:stop]).to(device)
                motif = torch.from_numpy(cached[f"motif_{view}"][start:stop]).to(device)
                logits, diagnostics = branch(base, pixel, motif)
                output[f"{view}_logits"].append(logits.float().cpu().numpy())
                output[f"{view}_delta"].append(diagnostics["delta"].float().cpu().numpy())
                if diagnostics["gate"] is not None:
                    output[f"{view}_gate"].append(diagnostics["gate"].float().cpu().numpy())
    arrays: dict[str, Any] = {}
    for name, parts in output.items():
        arrays[name] = None if not parts else np.concatenate(parts)
    arrays["tta_logits"] = (0.5 * (arrays["raw_logits"] + arrays["flip_logits"])).astype(np.float32)
    arrays["delta_mean_l2"] = float(np.mean(np.linalg.norm(np.concatenate((arrays["raw_delta"], arrays["flip_delta"])), axis=1)))
    arrays["delta_mean_absolute_logit"] = float(np.mean(np.abs(np.concatenate((arrays["raw_delta"], arrays["flip_delta"])))))
    gate = None if arrays["raw_gate"] is None else np.concatenate((arrays["raw_gate"], arrays["flip_gate"]))
    arrays["gate_summary"] = gate_summary(gate)
    return arrays


def save_checkpoint(path: Path, *, variant: str, epoch: int, selector: dict[str, Any], branch: ResidualBranch, backbone: torch.nn.Module, config_sha: str) -> None:
    torch.save(
        {
            "schema_version": 1,
            "variant": variant,
            "epoch": epoch,
            "selector": selector,
            "training_config_sha256": config_sha,
            "npf_checkpoint_sha256": EXPECTED["npf_checkpoint_sha256"],
            "frozen_backbone_state_sha256": state_sha256(backbone),
            "backbone_state_dict": {name: value.detach().cpu() for name, value in backbone.state_dict().items()},
            "residual_state_dict": {name: value.detach().cpu() for name, value in branch.state_dict().items()},
        },
        path,
    )


def main() -> None:
    args = parse_args()
    phase0_files = {
        "PUBLIC_FULL_NPF_AUDIT.json",
        "PUBLIC_FULL_NPF_AUDIT.md",
        "PUBLIC_FULL_NPF_OUTPUTS.npz",
        "PUBLIC_FULL_NPF_FUSION_SWEEP.csv",
    }
    if args.output.exists() and not args.resume_phase0:
        raise FileExistsError(f"fail-closed output exists: {args.output}")
    if args.resume_phase0 and (
        not args.output.is_dir()
        or not phase0_files.issubset({path.name for path in args.output.iterdir()})
    ):
        raise RuntimeError("resume-phase0 requires the complete Phase-0 artifact set")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.max_epochs != 30 or args.batch_size != 16:
        raise RuntimeError("registered Stage-1 horizon/batch contract changed")
    for path in (args.train_csv, args.public_csv, args.private_csv, args.full_checkpoint, args.npf_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_dataset = {
        "train_sha256": sha256_file(args.train_csv),
        "public_sha256": sha256_file(args.public_csv),
    }
    if actual_dataset != {key: EXPECTED[key] for key in actual_dataset}:
        raise RuntimeError(f"dataset identity mismatch: {actual_dataset}")
    set_seed(42)
    device = torch.device("cuda")
    sys.path.insert(0, str(args.source_root))
    from mpg_fer_table_vi.model import AblationMPGFER, AblationMode
    from mpg_fer_v2_3.checkpoint import config_hash
    from mpg_fer_v2_3.config import MPGConfig
    from mpg_fer_v2_3.data import FER2013Dataset, create_private_dataloader, create_training_dataloaders, inspect_split_file

    dataset_gate = {
        "train": inspect_split_file(args.train_csv, "train", validate_content=True),
        "public": inspect_split_file(args.public_csv, "val", validate_content=True),
        "private_before_selection": {"role": "test", "path": str(args.private_csv.resolve()), "accessed": False},
    }
    full_model, _, config = load_model(args.full_checkpoint, AblationMode.FULL, EXPECTED["full_checkpoint_sha256"], EXPECTED["full_checkpoint_source"], EXPECTED["full_epoch"], device, AblationMPGFER, MPGConfig, config_hash)
    npf_model, _, npf_config = load_model(args.npf_checkpoint, AblationMode.NO_PIXEL_FUSION, EXPECTED["npf_checkpoint_sha256"], EXPECTED["npf_checkpoint_source"], EXPECTED["npf_epoch"], device, AblationMPGFER, MPGConfig, config_hash)
    if config_hash(config) != config_hash(npf_config):
        raise RuntimeError("FULL and NPF scientific configs differ")
    backbone_before = state_sha256(npf_model)

    if args.resume_phase0:
        del full_model
        torch.cuda.empty_cache()
        with np.load(args.output / "PUBLIC_FULL_NPF_OUTPUTS.npz") as saved:
            npf_public = {
                "labels": saved["true_label"].copy(),
                "raw_logits": saved["npf_raw_logits"].copy(),
                "flip_logits": saved["npf_flip_logits"].copy(),
                "tta_logits": saved["npf_tta_logits"].copy(),
                "pixel_raw": saved["npf_pixel_readout_raw"].copy(),
                "pixel_flip": saved["npf_pixel_readout_flip"].copy(),
                "motif_raw": saved["npf_motif_readout_raw"].copy(),
                "motif_flip": saved["npf_motif_readout_flip"].copy(),
            }
        labels = npf_public["labels"].astype(np.int64)
        phase0 = json.loads((args.output / "PUBLIC_FULL_NPF_AUDIT.json").read_text(encoding="utf-8"))
        print(json.dumps({"phase": "PHASE_0", "status": "REUSED_VALIDATED_ARTIFACTS"}), flush=True)
    else:
        public_dataset = FER2013Dataset(args.public_csv, split="val", augment=False)
        public_loader = DataLoader(public_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        print(json.dumps({"phase": "PHASE_0", "model": "FULL", "status": "STARTED"}), flush=True)
        full_public = extract_frozen_views(full_model, public_loader, device, readouts=False)
        print(json.dumps({"phase": "PHASE_0", "model": "FULL", "status": "COMPLETE", "runtime_seconds": full_public["runtime_seconds"]}), flush=True)
        del full_model
        torch.cuda.empty_cache()
        print(json.dumps({"phase": "PHASE_0", "model": "NO_PIXEL_FUSION", "status": "STARTED"}), flush=True)
        npf_public = extract_frozen_views(npf_model, public_loader, device, readouts=True)
        print(json.dumps({"phase": "PHASE_0", "model": "NO_PIXEL_FUSION", "status": "COMPLETE", "runtime_seconds": npf_public["runtime_seconds"]}), flush=True)
        labels = npf_public["labels"].astype(np.int64)
        if not np.array_equal(labels, full_public["labels"]):
            raise RuntimeError("Public row alignment mismatch")
        phase0, sweep_rows = phase0_report(labels, full_public, npf_public)
        full_reference = json.loads((args.repo / "research/mpg_fer_v2_3/FULL_BASELINE_REFERENCE.json").read_text(encoding="utf-8"))
        for view, key in (("raw", "public_raw"), ("tta", "public_tta")):
            for metric_name in ("accuracy", "macro_f1"):
                if abs(phase0["metrics"]["FULL"][view][metric_name] - full_reference["canonical_fp32"][key][metric_name]) > 1e-9:
                    raise RuntimeError(f"FULL Public parity failure: {view}/{metric_name}")
        args.output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            args.output / "PUBLIC_FULL_NPF_OUTPUTS.npz",
            row_index=np.arange(len(labels), dtype=np.int64), true_label=labels,
            full_raw_logits=full_public["raw_logits"], full_flip_logits=full_public["flip_logits"], full_tta_logits=full_public["tta_logits"],
            npf_raw_logits=npf_public["raw_logits"], npf_flip_logits=npf_public["flip_logits"], npf_tta_logits=npf_public["tta_logits"],
            npf_pixel_readout_raw=npf_public["pixel_raw"], npf_pixel_readout_flip=npf_public["pixel_flip"],
            npf_motif_readout_raw=npf_public["motif_raw"], npf_motif_readout_flip=npf_public["motif_flip"],
        )
        phase0.update({"schema_version": 1, "status": "PASS", "scope": {"training": False, "dataset_role": "PublicTest", "selection_use": True, "private_opened": False}, "identity": {**actual_dataset, "full_checkpoint_sha256": EXPECTED["full_checkpoint_sha256"], "npf_checkpoint_sha256": EXPECTED["npf_checkpoint_sha256"], "config_sha256": EXPECTED["config_sha256"], "row_order": "canonical PublicTest CSV order"}, "immutability": {"FULL": full_public["state_sha256_before"] == full_public["state_sha256_after"], "NO_PIXEL_FUSION": npf_public["state_sha256_before"] == npf_public["state_sha256_after"]}, "runtime_seconds": {"FULL": full_public["runtime_seconds"], "NO_PIXEL_FUSION": npf_public["runtime_seconds"]}})
        write_json(args.output / "PUBLIC_FULL_NPF_AUDIT.json", phase0)
        write_csv(args.output / "PUBLIC_FULL_NPF_FUSION_SWEEP.csv", sweep_rows)
        (args.output / "PUBLIC_FULL_NPF_AUDIT.md").write_text("# PublicTest FULL-vs-NPF frozen audit\n\n" f"Status: **PASS**\n\nFULL TTA Accuracy/Macro-F1: **{100*phase0['metrics']['FULL']['tta']['accuracy']:.4f}% / {100*phase0['metrics']['FULL']['tta']['macro_f1']:.4f}%**.\n\n" f"NPF TTA Accuracy/Macro-F1: **{100*phase0['metrics']['NO_PIXEL_FUSION']['tta']['accuracy']:.4f}% / {100*phase0['metrics']['NO_PIXEL_FUSION']['tta']['macro_f1']:.4f}%**.\n\n" f"TTA disagreement: `{phase0['views']['tta']['disagreement']}`.\n\nScalar fusion is descriptive in-sample analysis only. PrivateTest was not opened.\n", encoding="utf-8")

    training_config = {
        "schema_version": 1,
        "seed": 42,
        "variants": ["R1", "R2"],
        "anchor": "NO_PIXEL_FUSION",
        "npf_checkpoint_sha256": EXPECTED["npf_checkpoint_sha256"],
        "frozen_backbone_state_sha256": backbone_before,
        "optimizer": {"family": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-3},
        "loss": {"cross_entropy": True, "lambda_delta": 1e-4, "delta_term": "mean(sum(delta**2, dim=class))"},
        "epochs": 30,
        "warmup_epochs": 2,
        "scheduler": "linear_warmup_then_cosine_to_zero",
        "early_stop_patience": 8,
        "checkpoint_selection": "highest PublicTest horizontal-flip TTA Accuracy, tie-break Macro-F1",
        "batch_size": 16,
        "frozen_backbone_microbatch_size": args.backbone_microbatch_size,
        "gradient_accumulation_steps": 2,
        "training_amp": True,
        "evaluation": "FP32, autocast off, TF32 off, raw and logit-average horizontal-flip TTA",
        "private_test_policy": "opened once only after R1 and R2 best checkpoints are selected and frozen",
        "dataset": dataset_gate,
        "dataset_sha256": {**actual_dataset, "private_sha256_expected_not_read": EXPECTED["private_sha256"]},
    }
    write_json(args.output / "TRAINING_CONFIG.json", training_config)
    training_config_sha = sha256_file(args.output / "TRAINING_CONFIG.json")

    branches = {variant: ResidualBranch(ResidualVariant(variant)).to(device) for variant in VARIANTS}
    zero_init = {}
    for variant, branch in branches.items():
        initial = residual_from_cached(branch, npf_public, device)
        zero_init[variant] = {
            "raw_exact": bool(np.array_equal(initial["raw_logits"], npf_public["raw_logits"])),
            "flip_exact": bool(np.array_equal(initial["flip_logits"], npf_public["flip_logits"])),
            "tta_exact": bool(np.array_equal(initial["tta_logits"], npf_public["tta_logits"])),
        }
    if not all(all(values.values()) for values in zero_init.values()):
        raise RuntimeError(f"zero-init NPF parity failure: {zero_init}")

    loaders = create_training_dataloaders(args.train_csv, args.public_csv, batch_size=args.batch_size, num_workers=args.num_workers, seed=42)
    optimizers = {variant: torch.optim.AdamW(branches[variant].parameters(), lr=3e-4, weight_decay=1e-3) for variant in VARIANTS}
    scalers = {variant: torch.amp.GradScaler("cuda", enabled=True) for variant in VARIANTS}
    histories: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    best: dict[str, dict[str, Any] | None] = {variant: None for variant in VARIANTS}
    patience = {variant: 0 for variant in VARIANTS}
    active = set(VARIANTS)
    training_started = time.monotonic()
    npf_model.eval().requires_grad_(False)
    for epoch in range(1, 31):
        loaders["train"].dataset.set_epoch(epoch)
        loaders["train"].sampler.set_epoch(epoch)
        lr = learning_rate_for_epoch(epoch)
        for variant in active:
            for group in optimizers[variant].param_groups:
                group["lr"] = lr
            branches[variant].train()
            optimizers[variant].zero_grad(set_to_none=True)
        loss_sums = {variant: 0.0 for variant in VARIANTS}
        sample_sums = {variant: 0 for variant in VARIANTS}
        delta_l2_sums = {variant: 0.0 for variant in VARIANTS}
        delta_abs_sums = {variant: 0.0 for variant in VARIANTS}
        for step, (images, targets) in enumerate(loaders["train"], start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            base_parts, pixel_parts, motif_parts = [], [], []
            with torch.no_grad():
                for start in range(0, len(images), args.backbone_microbatch_size):
                    microbatch = images[start : start + args.backbone_microbatch_size]
                    with torch.amp.autocast("cuda", enabled=True):
                        base_micro, outputs = npf_model(microbatch)
                    base_parts.append(base_micro)
                    pixel_parts.append(outputs["h_pixel_readout"])
                    motif_parts.append(outputs["h_motif_readout"])
            base_logits = torch.cat(base_parts, dim=0)
            pixel = torch.cat(pixel_parts, dim=0)
            motif = torch.cat(motif_parts, dim=0)
            for variant in tuple(active):
                with torch.amp.autocast("cuda", enabled=True):
                    final_logits, diagnostic = branches[variant](base_logits.detach(), pixel.detach(), motif.detach())
                    delta = diagnostic["delta"]
                    loss = F.cross_entropy(final_logits.float(), targets) + 1e-4 * delta.float().pow(2).sum(dim=1).mean()
                scalers[variant].scale(loss / 2).backward()
                count = len(targets)
                loss_sums[variant] += float(loss.detach()) * count
                sample_sums[variant] += count
                delta_l2_sums[variant] += float(delta.detach().float().norm(dim=1).sum())
                delta_abs_sums[variant] += float(delta.detach().float().abs().sum())
            if step % 2 == 0 or step == len(loaders["train"]):
                for variant in tuple(active):
                    scalers[variant].unscale_(optimizers[variant])
                    torch.nn.utils.clip_grad_norm_(branches[variant].parameters(), 1.0)
                    scalers[variant].step(optimizers[variant])
                    scalers[variant].update()
                    optimizers[variant].zero_grad(set_to_none=True)
        if any(parameter.grad is not None for parameter in npf_model.parameters()):
            raise RuntimeError("frozen NPF parameter received a gradient")
        if state_sha256(npf_model) != backbone_before:
            raise RuntimeError("frozen NPF changed during training")

        for variant in tuple(active):
            evaluated = residual_from_cached(branches[variant], npf_public, device)
            val_metrics = metrics(labels, evaluated["tta_logits"])
            changes = change_counts(labels, npf_public["tta_logits"].argmax(axis=1), evaluated["tta_logits"].argmax(axis=1))
            row = {
                "epoch": epoch,
                "learning_rate": lr,
                "train_loss": loss_sums[variant] / sample_sums[variant],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "mean_delta_l2_train": delta_l2_sums[variant] / sample_sums[variant],
                "mean_absolute_delta_logit_train": delta_abs_sums[variant] / (sample_sums[variant] * 7),
                "mean_delta_l2_val_views": evaluated["delta_mean_l2"],
                "mean_absolute_delta_logit_val_views": evaluated["delta_mean_absolute_logit"],
                "gate_statistics": None if evaluated["gate_summary"] is None else json.dumps(evaluated["gate_summary"], separators=(",", ":")),
                **changes,
            }
            histories[variant].append(row)
            candidate = {"accuracy": val_metrics["accuracy"], "macro_f1": val_metrics["macro_f1"], "loss": val_metrics["loss"]}
            if is_better(candidate, None if best[variant] is None else best[variant]["selector"]):
                best[variant] = {"epoch": epoch, "selector": candidate}
                patience[variant] = 0
                save_checkpoint(args.output / f"{variant}_best.pt", variant=variant, epoch=epoch, selector=candidate, branch=branches[variant], backbone=npf_model, config_sha=training_config_sha)
            else:
                patience[variant] += 1
                if patience[variant] >= 8:
                    active.remove(variant)
            print(json.dumps({"epoch": epoch, "variant": variant, "active": variant in active, "patience": patience[variant], "selector": candidate, "best": best[variant]}, separators=(",", ":")), flush=True)
        if not active:
            break

    for variant in VARIANTS:
        write_csv(args.output / f"{variant}_HISTORY.csv", histories[variant])
    if state_sha256(npf_model) != backbone_before:
        raise RuntimeError("frozen NPF changed after Stage 1")

    # Only now, after both variant selectors are frozen, open PrivateTest once.
    private_sha = sha256_file(args.private_csv)
    if private_sha != EXPECTED["private_sha256"]:
        raise RuntimeError("PrivateTest SHA mismatch after selector freeze")
    actual_dataset["private_sha256"] = private_sha
    private_loader = create_private_dataloader(args.private_csv, batch_size=args.eval_batch_size, num_workers=0)
    npf_private = extract_frozen_views(npf_model, private_loader, device, readouts=True)
    private_labels = npf_private["labels"].astype(np.int64)
    baseline_private = {view: metrics(private_labels, npf_private[f"{view}_logits"]) for view in ("raw", "tta")}
    frozen_reference = {"raw": (0.6815268877124547, 0.6653503035453028), "tta": (0.706603510727222, 0.6940030403992327)}
    for view in ("raw", "tta"):
        if abs(baseline_private[view]["accuracy"] - frozen_reference[view][0]) > 1e-9 or abs(baseline_private[view]["macro_f1"] - frozen_reference[view][1]) > 1e-9:
            raise RuntimeError(f"frozen NPF Private parity failure: {view}")

    final: dict[str, Any] = {}
    evaluated_variants: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        checkpoint = torch.load(args.output / f"{variant}_best.pt", map_location="cpu", weights_only=False)
        if checkpoint["frozen_backbone_state_sha256"] != backbone_before:
            raise RuntimeError("selected checkpoint backbone identity mismatch")
        branches[variant].load_state_dict(checkpoint["residual_state_dict"], strict=True)
        evaluated = residual_from_cached(branches[variant], npf_private, device)
        evaluated_variants[variant] = evaluated
        final[variant] = {
            "selected_epoch": checkpoint["epoch"],
            "checkpoint_sha256": sha256_file(args.output / f"{variant}_best.pt"),
            "selector": checkpoint["selector"],
            "private": {view: metrics(private_labels, evaluated[f"{view}_logits"]) for view in ("raw", "tta")},
            "private_change_counts_tta": change_counts(private_labels, npf_private["tta_logits"].argmax(axis=1), evaluated["tta_logits"].argmax(axis=1)),
            "private_delta_mean_l2_views": evaluated["delta_mean_l2"],
            "private_delta_mean_absolute_logit_views": evaluated["delta_mean_absolute_logit"],
            "private_gate_summary": evaluated["gate_summary"],
        }
    selected_variant = max(VARIANTS, key=lambda variant: (final[variant]["selector"]["accuracy"], final[variant]["selector"]["macro_f1"]))
    public_baseline = phase0["metrics"]["NO_PIXEL_FUSION"]["tta"]["accuracy"]
    validation_gain_pp = 100.0 * (final[selected_variant]["selector"]["accuracy"] - public_baseline)
    private_accuracy = final[selected_variant]["private"]["tta"]["accuracy"]
    go = validation_gain_pp >= 0.40 and private_accuracy > 0.7105043187517415
    if go and private_accuracy >= 0.72:
        signal = "VERY_STRONG_SIGNAL"
    elif go and private_accuracy >= 0.715:
        signal = "STRONG_SIGNAL"
    elif go:
        signal = "GO_STAGE_2"
    else:
        signal = "NO_GO_STOP_STAGE_1"

    sample_rows = []
    baseline_raw = npf_private["raw_logits"].argmax(axis=1)
    baseline_tta = npf_private["tta_logits"].argmax(axis=1)
    for index, label in enumerate(private_labels):
        row: dict[str, Any] = {"row_index": index, "true_label": int(label), "class_name": CLASS_NAMES[int(label)], "npf_raw_prediction": int(baseline_raw[index]), "npf_tta_prediction": int(baseline_tta[index]), "npf_tta_correct": bool(baseline_tta[index] == label)}
        for variant in VARIANTS:
            evaluated = evaluated_variants[variant]
            raw_prediction = int(evaluated["raw_logits"][index].argmax())
            tta_prediction = int(evaluated["tta_logits"][index].argmax())
            row.update({f"{variant.lower()}_raw_prediction": raw_prediction, f"{variant.lower()}_tta_prediction": tta_prediction, f"{variant.lower()}_tta_correct": bool(tta_prediction == label), f"{variant.lower()}_tta_changed_vs_npf": bool(tta_prediction != baseline_tta[index]), f"{variant.lower()}_npf_wrong_to_correct": bool(baseline_tta[index] != label and tta_prediction == label), f"{variant.lower()}_npf_correct_to_wrong": bool(baseline_tta[index] == label and tta_prediction != label), f"{variant.lower()}_raw_delta_l2": float(np.linalg.norm(evaluated["raw_delta"][index])), f"{variant.lower()}_flip_delta_l2": float(np.linalg.norm(evaluated["flip_delta"][index]))})
        sample_rows.append(row)
    write_csv(args.output / "SELECTED_SAMPLE_DIAGNOSTICS.csv", sample_rows)

    summary = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": "NPF_ANCHORED_CONDITIONAL_PIXEL_RESIDUAL_STAGE_1",
        "identity": {**actual_dataset, "npf_checkpoint_sha256": EXPECTED["npf_checkpoint_sha256"], "full_checkpoint_sha256": EXPECTED["full_checkpoint_sha256"], "training_config_sha256": training_config_sha},
        "strict_checks": {"public_full_parity": True, "zero_initialization_exact_npf_parity": zero_init, "frozen_backbone_before": backbone_before, "frozen_backbone_after": state_sha256(npf_model), "frozen_backbone_bitwise_unchanged": state_sha256(npf_model) == backbone_before, "only_residual_parameters_optimized": True, "private_opened_only_after_both_selectors_frozen": True},
        "phase0": phase0,
        "baseline_private": baseline_private,
        "variants": final,
        "selected_variant_by_public": selected_variant,
        "selected_public_gain_pp": validation_gain_pp,
        "selected_private_tta_accuracy": private_accuracy,
        "motif_head_reference_tta_accuracy": 0.7105043187517415,
        "go_stage_2": go,
        "decision": signal,
        "answer": "YES" if go else "NO",
        "answer_detail": "The selected NPF-trajectory residual recovered sufficient complementary pixel information and beat the motif-head reference." if go else "The selected NPF-trajectory residual did not satisfy both the registered PublicTest gain and 71.0504% frozen Test reference gates.",
        "runtime_seconds": {"training_stage1": time.monotonic() - training_started},
        "stage2": "NOT_RUN_PENDING_REGISTERED_GO" if go else "NOT_RUN_NO_GO",
    }
    write_json(args.output / "RESIDUAL_EXPERIMENT_SUMMARY.json", summary)
    md = f"""# NPF conditional pixel-residual experiment

Status: **PASS**
Decision: **{signal}**

## Stage 1 selection

| Variant | Selected epoch | Public TTA Acc. | Private raw Acc. | Private TTA Acc. | Private TTA Macro-F1 |
|---|---:|---:|---:|---:|---:|
"""
    for variant in VARIANTS:
        item = final[variant]
        md += f"| {variant} | {item['selected_epoch']} | {100*item['selector']['accuracy']:.4f}% | {100*item['private']['raw']['accuracy']:.4f}% | {100*item['private']['tta']['accuracy']:.4f}% | {100*item['private']['tta']['macro_f1']:.4f}% |\n"
    md += f"""

Selected by PublicTest only: **{selected_variant}**, gain over frozen Public NPF **{validation_gain_pp:+.4f} pp**.

The registered GO decision is **{go}**. {summary['answer_detail']}

No backbone parameter changed. PrivateTest was opened only after both best checkpoints were selected and frozen.
"""
    (args.output / "RESIDUAL_EXPERIMENT_SUMMARY.md").write_text(md, encoding="utf-8")
    environment = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "command": " ".join(sys.argv)}
    write_json(args.output / "environment.json", environment)
    outputs = [path.name for path in args.output.iterdir() if path.is_file() and path.name != "checksums.sha256"]
    with (args.output / "checksums.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        for name in sorted(outputs):
            handle.write(f"{sha256_file(args.output / name)}  {name}\n")
    print(json.dumps({"status": "PASS", "decision": signal, "selected_variant": selected_variant, "go_stage_2": go, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
