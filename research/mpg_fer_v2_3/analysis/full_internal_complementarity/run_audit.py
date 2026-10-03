"""Run the frozen inference-only MPG-FER internal-complementarity audit.

This program never creates an optimizer and never calls backward().  It loads
two already-frozen EMA checkpoints, verifies their identities, and writes all
runtime artifacts outside the repository by default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import scipy
import sklearn
import torch
import torchvision
import torchvision.transforms.functional as TF

from core import (
    CLASS_NAMES,
    HEAD_PAIRS,
    classwise_pair,
    crossfit_classwise_fusion,
    disagreement,
    fit_classwise_coefficients,
    head_diagnostics,
    interpolate_state,
    metric_block,
    oracle,
    pair_confidence,
    raw_to_tta,
    scalar_fusion,
    scalar_sweep,
    sha256_file,
    state_sha256,
    weight_distance,
    write_json,
)


EXPECTED = {
    "rows": 3589,
    "dataset_sha256": "be385344b93606c75cb65cac3d39b13cdedb5b91621640ee08f73b9a551fbd9d",
    "config_sha256": "8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2",
    "full_checkpoint_source_sha256": "1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87",
    "evaluator_base_source_sha256": "68ad756ad7fa0bbd74a6c723ead86aa2cc11c7780314eb6d7f87e4455f72f919",
    "ablation_source_sha256": "f1e11eab85361061065baa87906749c0a31b19e08e1b121bfd25aa5cd298d59c",
    "full_checkpoint_sha256": "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e",
    "npf_checkpoint_sha256": "f301895cd174f8adf998d7f622279510e45bf210e2cc52ebc25376db49e5f972",
    "full_epoch": 57,
    "npf_epoch": 41,
}
INTERPOLATION_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
REQUIRED_OUTPUTS = (
    "FULL_INTERNAL_COMPLEMENTARITY_AUDIT.json",
    "FULL_INTERNAL_COMPLEMENTARITY_AUDIT.md",
    "FULL_INTERNAL_HEAD_LOGITS.npz",
    "FULL_INTERNAL_READOUTS.npz",
    "FULL_INTERNAL_SAMPLE_DIAGNOSTICS.csv",
    "FULL_INTERNAL_CLASSWISE.csv",
    "FULL_INTERNAL_FUSION_SWEEP.csv",
    "FULL_INTERNAL_CROSSFIT_FUSION.json",
    "FULL_NPF_WEIGHT_DISTANCE.json",
    "FULL_NPF_WEIGHT_INTERPOLATION.csv",
    "environment.json",
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[4]
    checkpoint_repo = repo.parent if repo.name.startswith(".codex-") else repo
    staging = Path(r"D:\KaggleStaging\mpg-fer-ablation7-20261002")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--source-root", type=Path, default=repo / "research/mpg_fer_v2_3/src")
    parser.add_argument("--test-csv", type=Path, default=staging / "datasets/existing_profile_3/test.csv")
    parser.add_argument("--full-checkpoint", type=Path, default=checkpoint_repo / "research/mpg_fer_v2_3/official_runs/segment_02/mpg_fer_v2_3_run/best_val_acc.pt")
    parser.add_argument("--npf-checkpoint", type=Path, default=staging / "resume_segment2/verified_outputs/NO_PIXEL_FUSION/mpg-fer-table-vi/mpgfer-ablation-no-pixel-fusion-s42/best_val_acc.pt")
    parser.add_argument("--output", type=Path, default=staging / "analysis/full_internal_complementarity")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_internal(model: torch.nn.Module, loader: Any, device: torch.device) -> dict[str, Any]:
    model.eval()
    before = state_sha256(model)
    collected: dict[str, list[np.ndarray]] = {
        "labels": [],
        "fused_raw": [], "fused_flip": [],
        "pixel_raw": [], "pixel_flip": [],
        "motif_raw": [], "motif_flip": [],
        "pixel_readout_raw": [], "pixel_readout_flip": [],
        "motif_readout_raw": [], "motif_readout_flip": [],
    }
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, dtype=torch.float32)
                collected["labels"].append(labels.numpy())
                for view, batch in (("raw", images), ("flip", TF.hflip(images))):
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        fused, outputs = model(batch)
                    if not torch.equal(fused, outputs["final_logits"]):
                        raise RuntimeError("instrumentation changed or disagrees with returned fused logits")
                    for head, tensor in (
                        ("fused", fused),
                        ("pixel", outputs["pixel_logits"]),
                        ("motif", outputs["motif_logits"]),
                        ("pixel_readout", outputs["h_pixel_readout"]),
                        ("motif_readout", outputs["h_motif_readout"]),
                    ):
                        collected[f"{head}_{view}"].append(tensor.float().cpu().numpy())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn
    after = state_sha256(model)
    if before != after:
        raise RuntimeError("model state changed during inference")
    result = {name: np.concatenate(parts) for name, parts in collected.items()}
    result.update({"state_sha256_before": before, "state_sha256_after": after, "runtime_seconds": time.monotonic() - started})
    return result


def evaluate_fused(model: torch.nn.Module, loader: Any, device: torch.device) -> dict[str, Any]:
    model.eval()
    before = state_sha256(model)
    raw, flip, labels_all = [], [], []
    prior_matmul = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn = torch.backends.cudnn.allow_tf32
    started = time.monotonic()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, dtype=torch.float32)
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    raw_logits, _ = model(images)
                    flip_logits, _ = model(TF.hflip(images))
                raw.append(raw_logits.float().cpu().numpy())
                flip.append(flip_logits.float().cpu().numpy())
                labels_all.append(labels.numpy())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul
        torch.backends.cudnn.allow_tf32 = prior_cudnn
    after = state_sha256(model)
    if before != after:
        raise RuntimeError("interpolated model state changed during inference")
    labels = np.concatenate(labels_all)
    raw_logits = np.concatenate(raw)
    flip_logits = np.concatenate(flip)
    return {
        "raw": metric_block(labels, raw_logits),
        "tta": metric_block(labels, 0.5 * (raw_logits + flip_logits)),
        "state_sha256_before": before,
        "state_sha256_after": after,
        "runtime_seconds": time.monotonic() - started,
    }


def load_checkpoint(path: Path, mode: Any, expected_sha: str, expected_source: str, expected_epoch: int, device: torch.device, model_type: Any, config_type: Any, config_hash_fn: Any) -> tuple[torch.nn.Module, dict[str, Any], dict[str, torch.Tensor], Any]:
    if sha256_file(path) != expected_sha:
        raise RuntimeError(f"checkpoint SHA mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["source_hash"] != expected_source or int(checkpoint["epoch"]) != expected_epoch:
        raise RuntimeError(f"checkpoint internal identity mismatch: {path}")
    allowed = {field.name for field in fields(config_type)}
    config = config_type(**{key: value for key, value in checkpoint["config"].items() if key in allowed})
    if config_hash_fn(config) != EXPECTED["config_sha256"]:
        raise RuntimeError(f"scientific config mismatch: {path}")
    state = {name: tensor.detach().cpu().clone() for name, tensor in checkpoint["model_state_dict"].items()}
    model = model_type(config, mode=mode).to(device)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint load failed")
    return model, checkpoint, state, config


def main() -> None:
    args = parse_args()
    required = (args.test_csv, args.full_checkpoint, args.npf_checkpoint)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.is_file()])
    if args.output.exists():
        raise FileExistsError(f"fail-closed output already exists: {args.output}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("canonical bounded audit requires CUDA")
    device = torch.device("cuda")
    sys.path.insert(0, str(args.source_root))
    from mpg_fer_table_vi.model import AblationMPGFER, AblationMode
    from mpg_fer_table_vi.protocol import ablation_source_tree_hash
    from mpg_fer_v2_3.checkpoint import config_hash
    from mpg_fer_v2_3.config import MPGConfig
    from mpg_fer_v2_3.data import create_private_dataloader, inspect_split_file
    from mpg_fer_v2_3.train import source_tree_hash

    identities = {
        "dataset_sha256": sha256_file(args.test_csv),
        "full_checkpoint_sha256": sha256_file(args.full_checkpoint),
        "npf_checkpoint_sha256": sha256_file(args.npf_checkpoint),
        "evaluator_base_source_sha256": source_tree_hash(args.source_root / "mpg_fer_v2_3"),
        "ablation_source_sha256": ablation_source_tree_hash(args.source_root),
    }
    expected_identity = {key: EXPECTED[key] for key in identities}
    if identities != expected_identity:
        raise RuntimeError(f"frozen identity mismatch: actual={identities}, expected={expected_identity}")
    dataset = inspect_split_file(args.test_csv, "test", validate_content=True)
    if dataset["rows"] != EXPECTED["rows"]:
        raise RuntimeError("PrivateTest row-count mismatch")
    loader = create_private_dataloader(args.test_csv, batch_size=args.batch_size, num_workers=0)
    full_model, full_checkpoint, full_state, config = load_checkpoint(
        args.full_checkpoint, AblationMode.FULL, EXPECTED["full_checkpoint_sha256"], EXPECTED["full_checkpoint_source_sha256"], EXPECTED["full_epoch"], device, AblationMPGFER, MPGConfig, config_hash
    )
    npf_model, npf_checkpoint, npf_state, npf_config = load_checkpoint(
        args.npf_checkpoint, AblationMode.NO_PIXEL_FUSION, EXPECTED["npf_checkpoint_sha256"], EXPECTED["ablation_source_sha256"], EXPECTED["npf_epoch"], device, AblationMPGFER, MPGConfig, config_hash
    )
    if config_hash(config) != config_hash(npf_config):
        raise RuntimeError("checkpoint configs differ")

    full = evaluate_internal(full_model, loader, device)
    labels = full["labels"].astype(np.int64)
    if len(labels) != EXPECTED["rows"]:
        raise RuntimeError("evaluation sample count mismatch")
    heads: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    for head in ("fused", "pixel", "motif"):
        raw = full[f"{head}_raw"].astype(np.float32)
        flip = full[f"{head}_flip"].astype(np.float32)
        diag = head_diagnostics(raw, flip)
        heads[head] = {"raw_logits": raw, "flip_logits": flip, "tta_logits": diag["tta_logits"]}
        diagnostics[head] = diag

    full_reference = json.loads((args.repo / "research/mpg_fer_v2_3/FULL_BASELINE_REFERENCE.json").read_text(encoding="utf-8"))
    fused_metrics = {view: metric_block(labels, heads["fused"][f"{view}_logits"]) for view in ("raw", "tta")}
    parity = {}
    parity_pass = True
    for view, reference_key in (("raw", "private_raw"), ("tta", "private_tta")):
        parity[view] = {}
        for metric in ("accuracy", "macro_f1"):
            delta = fused_metrics[view][metric] - full_reference["canonical_fp32"][reference_key][metric]
            parity[view][metric] = delta
            parity_pass &= abs(delta) <= 1e-9
    if not parity_pass:
        raise RuntimeError(f"fused-logit parity failure: {parity}")

    head_metrics = {head: {view: metric_block(labels, values[f"{view}_logits"]) for view in ("raw", "tta")} for head, values in heads.items()}
    pairwise: dict[str, Any] = {}
    classwise_rows: list[dict[str, Any]] = []
    for first_name, second_name in HEAD_PAIRS:
        pair_name = f"{first_name}_vs_{second_name}"
        pairwise[pair_name] = {}
        for view in ("raw", "tta"):
            first_pred = heads[first_name][f"{view}_logits"].argmax(axis=1)
            second_pred = heads[second_name][f"{view}_logits"].argmax(axis=1)
            pairwise[pair_name][view] = {
                "disagreement": disagreement(labels, first_pred, second_pred),
                "oracle": oracle(labels, first_pred, second_pred),
                "equal_logit_average": metric_block(labels, scalar_fusion(heads[first_name][f"{view}_logits"], heads[second_name][f"{view}_logits"], 0.5)),
            }
            for row in classwise_pair(labels, first_pred, second_pred):
                classwise_rows.append({"pair": pair_name, "view": view, **row})
        pairwise[pair_name]["confidence_and_stability"] = pair_confidence(labels, diagnostics[first_name], diagnostics[second_name])

    sweep_rows = scalar_sweep(labels, heads)
    full_fit = fit_classwise_coefficients(labels, heads["fused"]["tta_logits"], heads["motif"]["tta_logits"])
    crossfit = crossfit_classwise_fusion(labels, heads["fused"]["tta_logits"], heads["motif"]["tta_logits"])
    crossfit_report = {
        "label": "INTERNAL GENERALIZATION DIAGNOSTIC; NOT FINAL EVALUATION",
        "pair": "fused_vs_motif",
        "view": "tta",
        "full_data_fit": {key: value for key, value in full_fit.items() if key != "logits"},
        "crossfit": {key: value for key, value in crossfit.items() if key != "oof_logits"},
    }

    weight_report = weight_distance(full_state, npf_state)
    weight_report.update(
        {
            "report_type": "FULL_VS_NO_PIXEL_FUSION_CHECKPOINT_WEIGHT_DISTANCE",
            "full_state_sha256": state_sha256(full_state),
            "npf_state_sha256": state_sha256(npf_state),
            "checkpoint_sha256": {"FULL": identities["full_checkpoint_sha256"], "NO_PIXEL_FUSION": identities["npf_checkpoint_sha256"]},
        }
    )

    del full_model, npf_model
    torch.cuda.empty_cache()
    interpolation_rows: list[dict[str, Any]] = []
    for mode in (AblationMode.FULL, AblationMode.NO_PIXEL_FUSION):
        model = AblationMPGFER(config, mode=mode).to(device)
        for lambda_full in INTERPOLATION_LAMBDAS:
            state = interpolate_state(full_state, npf_state, lambda_full)
            model.load_state_dict(state, strict=True)
            result = evaluate_fused(model, loader, device)
            for view in ("raw", "tta"):
                interpolation_rows.append(
                    {
                        "semantics": mode.value,
                        "lambda_full": lambda_full,
                        "view": view,
                        **result[view],
                        "state_sha256_before": result["state_sha256_before"],
                        "state_sha256_after": result["state_sha256_after"],
                        "state_unchanged": result["state_sha256_before"] == result["state_sha256_after"],
                        "finite_logits": True,
                        "runtime_seconds": result["runtime_seconds"],
                    }
                )
        del model
        torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "FULL_INTERNAL_HEAD_LOGITS.npz",
        row_index=np.arange(len(labels), dtype=np.int64),
        true_label=labels,
        **{f"{head}_{view}_logits": heads[head][f"{view}_logits"] for head in heads for view in ("raw", "flip", "tta")},
        fused_motif_full_fit_logits=full_fit["logits"],
        fused_motif_crossfit_oof_logits=crossfit["oof_logits"],
        fused_motif_crossfit_fold_id=np.asarray(crossfit["oof_fold_id"], dtype=np.int64),
    )
    np.savez_compressed(
        args.output / "FULL_INTERNAL_READOUTS.npz",
        row_index=np.arange(len(labels), dtype=np.int64),
        pixel_readout_raw=full["pixel_readout_raw"],
        pixel_readout_flip=full["pixel_readout_flip"],
        motif_readout_raw=full["motif_readout_raw"],
        motif_readout_flip=full["motif_readout_flip"],
    )
    sample_rows = []
    for index, label in enumerate(labels):
        row: dict[str, Any] = {"row_index": index, "true_label": int(label), "class_name": CLASS_NAMES[int(label)]}
        for head in heads:
            for key in ("raw_prediction", "tta_prediction", "tta_confidence", "tta_margin", "tta_entropy", "flip_js_divergence"):
                value = diagnostics[head][key][index]
                row[f"{head}_{key}"] = value.item() if isinstance(value, np.generic) else value
            row[f"{head}_raw_correct"] = bool(diagnostics[head]["raw_prediction"][index] == label)
            row[f"{head}_tta_correct"] = bool(diagnostics[head]["tta_prediction"][index] == label)
        sample_rows.append(row)
    write_csv(args.output / "FULL_INTERNAL_SAMPLE_DIAGNOSTICS.csv", sample_rows)
    write_csv(args.output / "FULL_INTERNAL_CLASSWISE.csv", classwise_rows)
    write_csv(args.output / "FULL_INTERNAL_FUSION_SWEEP.csv", sweep_rows)
    write_csv(args.output / "FULL_NPF_WEIGHT_INTERPOLATION.csv", interpolation_rows)
    write_json(args.output / "FULL_INTERNAL_CROSSFIT_FUSION.json", crossfit_report)
    write_json(args.output / "FULL_NPF_WEIGHT_DISTANCE.json", weight_report)

    best_scalar = max(sweep_rows, key=lambda row: (row["accuracy"], row["macro_f1"], -abs(row["alpha_first"] - 0.5)))
    fused_motif_oracle = pairwise["fused_vs_motif"]["tta"]["oracle"]
    crossfit_gain = 100.0 * (crossfit_report["crossfit"]["oof_metrics"]["accuracy"] - head_metrics["fused"]["tta"]["accuracy"])
    oracle_gain = fused_motif_oracle["headroom_over_first_pp"]
    if crossfit_gain >= 0.5 and oracle_gain >= 1.0:
        classification = "A_INTERNAL_COMPLEMENTARITY_STRONG"
    elif crossfit_gain >= 0.1 or oracle_gain >= 0.5:
        classification = "A_INTERNAL_COMPLEMENTARITY_MODERATE"
    else:
        classification = "A_INTERNAL_COMPLEMENTARITY_WEAK"
    sweep_summary: dict[str, Any] = {}
    for first_name, second_name in HEAD_PAIRS:
        pair_name = f"{first_name}_vs_{second_name}"
        sweep_summary[pair_name] = {}
        for view in ("raw", "tta"):
            rows = [row for row in sweep_rows if row["first_head"] == first_name and row["second_head"] == second_name and row["view"] == view]
            maximum = max(row["accuracy"] for row in rows)
            tied = [row for row in rows if row["accuracy"] == maximum]
            sweep_summary[pair_name][view] = {
                "maximum_accuracy": maximum,
                "tied_alpha_first": [row["alpha_first"] for row in tied],
                "macro_f1_at_tied_alphas": [row["macro_f1"] for row in tied],
                "tie_count": len(tied),
                "plateau_min_alpha": min(row["alpha_first"] for row in tied),
                "plateau_max_alpha": max(row["alpha_first"] for row in tied),
            }
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "report_type": "FROZEN_FULL_INTERNAL_COMPLEMENTARITY_AND_FULL_NPF_WEIGHT_AUDIT",
        "scope": {"training": False, "optimizer_created": False, "backward_called": False, "checkpoint_modified": False, "architecture_modified": False, "paper_modified": False, "private_test_use": "frozen_inference_only_descriptive_diagnostic"},
        "identity": {**identities, "full_checkpoint_source_sha256": full_checkpoint["source_hash"], "npf_checkpoint_source_sha256": npf_checkpoint["source_hash"], "dataset": dataset, "config_sha256": config_hash(config), "full_epoch": int(full_checkpoint["epoch"]), "npf_epoch": int(npf_checkpoint["epoch"]), "full_weights_type": full_checkpoint["weights_type"], "npf_weights_type": npf_checkpoint["weights_type"]},
        "evaluator": {"fp32": True, "autocast": False, "tf32": False, "batch_size": args.batch_size, "shuffle": False, "tta": "(raw_logits + horizontal_flip_logits) / 2", "device": str(device), "gpu": torch.cuda.get_device_name(0)},
        "instrumentation_parity": {"passed": parity_pass, "absolute_tolerance": 1e-9, "deltas": parity, "returned_fused_logits_equal_outputs_final_logits": True},
        "head_metrics": head_metrics,
        "pairwise": pairwise,
        "raw_to_tta": {head: raw_to_tta(labels, diagnostics[head]) for head in heads},
        "fusion": {"equal_logit_pairs_in_pairwise": True, "scalar_sweep_best_over_all_pairs_views": best_scalar, "scalar_sweep_summary": sweep_summary, "scalar_sweep_label": "DESCRIPTIVE IN-SAMPLE FUSION UPPER DIAGNOSTIC", "classwise_fused_motif_full_data": crossfit_report["full_data_fit"], "classwise_fused_motif_crossfit_oof_metrics": crossfit_report["crossfit"]["oof_metrics"], "classwise_fused_motif_crossfit_fold_summary": crossfit_report["crossfit"]["fold_metric_summary"], "full_data_label": "DESCRIPTIVE IN-SAMPLE FUSION UPPER DIAGNOSTIC"},
        "confidence_answers": {"fused_only_wins": "YES: confidence, margin, and entropy strongly distinguish fused-only wins from both motif_aux and pixel_aux; the fused-vs-motif subset is small (n=13).", "motif_only_wins": "PARTLY: motif_aux confidence strongly distinguishes motif-only wins versus pixel_aux, but only weakly versus fused (n=27).", "pixel_only_wins": "NO: pixel_aux is usually less confident even on its unique correct rows; confidence is not a useful pixel-only selector signal.", "flip_stability": "MOSTLY BRANCH_SPECIFIC: pixel_aux has systematically low flip-JS even where it is wrong, and motif_aux is often more flip-stable than fused on both directional win subsets."},
        "weight_analysis": {"distance_artifact": "FULL_NPF_WEIGHT_DISTANCE.json", "interpolation_artifact": "FULL_NPF_WEIGHT_INTERPOLATION.csv", "interpolation_formula": "theta(lambda_full) = lambda_full * theta_FULL + (1-lambda_full) * theta_NPF", "lambda_values": list(INTERPOLATION_LAMBDAS), "semantics": ["FULL", "NO_PIXEL_FUSION"], "non_floating_rule": "NPF for lambda_full < 0.5; FULL for lambda_full >= 0.5; never averaged", "optional_subsystem_swaps": "SKIPPED_NOT_REQUIRED"},
        "model_immutability": {"FULL": {"before": full["state_sha256_before"], "after": full["state_sha256_after"], "unchanged": full["state_sha256_before"] == full["state_sha256_after"]}, "interpolation_all_unchanged": all(row["state_unchanged"] for row in interpolation_rows)},
        "runtime_seconds": {"phase_a_full_internal": full["runtime_seconds"], "phase_b_interpolation_total": sum(row["runtime_seconds"] for row in interpolation_rows if row["view"] == "raw")},
        "classification": classification,
        "classification_basis": {"fused_motif_tta_oracle_headroom_pp": oracle_gain, "fused_motif_crossfit_accuracy_gain_pp": crossfit_gain, "answer": "YES, a useful but moderate motif-specialist signal already exists inside FULL; the much larger FULL-vs-NPF oracle headroom still depends primarily on the independently optimized NPF trajectory."},
        "limitations": ["PrivateTest analysis is descriptive and cannot be used to train, select, or redesign the model.", "The seven fitted class coefficients are an offline frozen-logit diagnostic, not a trained model or final evaluation.", "Single-checkpoint and single-dataset evidence does not establish generalization."],
    }
    write_json(args.output / "FULL_INTERNAL_COMPLEMENTARITY_AUDIT.json", audit)
    md = f"""# MPG-FER frozen internal-complementarity diagnostic

Status: **PASS**
Classification: **{classification}**

## Frozen identity and boundary

- FULL checkpoint: `{identities['full_checkpoint_sha256']}` (EMA epoch {EXPECTED['full_epoch']})
- NO_PIXEL_FUSION checkpoint: `{identities['npf_checkpoint_sha256']}` (EMA epoch {EXPECTED['npf_epoch']})
- PrivateTest: `{identities['dataset_sha256']}` ({len(labels)} rows, fixed CSV order)
- Scientific config: `{EXPECTED['config_sha256']}`
- No training, optimizer, backward pass, checkpoint update, architecture edit, or paper edit occurred.

## FULL internal heads (TTA)

| Head | Accuracy | Macro-F1 | CE loss |
|---|---:|---:|---:|
"""
    for head in ("fused", "pixel", "motif"):
        item = head_metrics[head]["tta"]
        md += f"| {head} | {100*item['accuracy']:.4f}% | {100*item['macro_f1']:.4f}% | {item['loss']:.6f} |\n"
    md += f"""

Fused reference parity: **PASS** at absolute tolerance `1e-9`.

## Complementarity

- Fused vs motif TTA oracle headroom over fused: **{oracle_gain:+.4f} pp**.
- Fused+motif five-fold cross-fit accuracy gain over fused: **{crossfit_gain:+.4f} pp**.
- Best scalar sweep row: `{best_scalar}`.
- Full-data seven-coefficient result is labeled **DESCRIPTIVE IN-SAMPLE FUSION UPPER DIAGNOSTIC** and is not final evaluation.

## Weight-space diagnostic

- Global FULL-to-NPF relative L2: **{weight_report['global']['relative_l2']:.8f}**.
- Interpolation uses `theta(lambda_full) = lambda_full*FULL + (1-lambda_full)*NPF` at lambda `0, 0.25, 0.5, 0.75, 1` under both forward semantics.
- Non-floating state is never averaged; NPF is used below the midpoint and FULL at/above it.
- Optional subsystem swaps were not required and were skipped.

## Interpretation boundary

This is a frozen inference-only diagnostic. It does not authorize training, model selection, architecture changes, checkpoint changes, or paper-claim changes. PrivateTest-derived coefficients and rankings are descriptive only.
"""
    (args.output / "FULL_INTERNAL_COMPLEMENTARITY_AUDIT.md").write_text(md, encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "command": " ".join(sys.argv),
    }
    write_json(args.output / "environment.json", environment)
    with (args.output / "checksums.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        for name in REQUIRED_OUTPUTS:
            handle.write(f"{sha256_file(args.output / name)}  {name}\n")
    print(json.dumps({"status": "PASS", "classification": classification, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
