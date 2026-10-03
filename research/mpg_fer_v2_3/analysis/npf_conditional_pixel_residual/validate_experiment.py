"""Independent validator for the NPF residual experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
import torch

from protocol import disagreement, metrics, sha256_file, state_sha256


REQUIRED = {
    "PUBLIC_FULL_NPF_AUDIT.json",
    "PUBLIC_FULL_NPF_AUDIT.md",
    "PUBLIC_FULL_NPF_OUTPUTS.npz",
    "PUBLIC_FULL_NPF_FUSION_SWEEP.csv",
    "TRAINING_CONFIG.json",
    "R1_HISTORY.csv",
    "R2_HISTORY.csv",
    "R1_best.pt",
    "R2_best.pt",
    "RESIDUAL_EXPERIMENT_SUMMARY.json",
    "RESIDUAL_EXPERIMENT_SUMMARY.md",
    "SELECTED_SAMPLE_DIAGNOSTICS.csv",
    "environment.json",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def close(actual: float, expected: float, tolerance: float = 2e-9) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise RuntimeError(f"numeric mismatch: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(r"D:\KaggleStaging\mpg-fer-ablation7-20261002\analysis\npf_conditional_pixel_residual"))
    args = parser.parse_args()
    manifest_path = args.output / "checksums.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    recorded = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    actual_files = {path.name for path in args.output.iterdir() if path.is_file() and path.name != "checksums.sha256"}
    if set(recorded) != actual_files or not REQUIRED.issubset(actual_files):
        raise RuntimeError("artifact/checksum file-set mismatch")
    for name, digest in recorded.items():
        if sha256_file(args.output / name) != digest:
            raise RuntimeError(f"checksum mismatch: {name}")

    audit = json.loads((args.output / "PUBLIC_FULL_NPF_AUDIT.json").read_text(encoding="utf-8"))
    summary = json.loads((args.output / "RESIDUAL_EXPERIMENT_SUMMARY.json").read_text(encoding="utf-8"))
    config = json.loads((args.output / "TRAINING_CONFIG.json").read_text(encoding="utf-8"))
    environment = json.loads((args.output / "environment.json").read_text(encoding="utf-8"))
    if not all(finite(item) for item in (audit, summary, config, environment)):
        raise RuntimeError("non-finite JSON")
    if audit["status"] != "PASS" or audit["scope"]["private_opened"] is not False:
        raise RuntimeError("Phase-0 scope violation")
    if config["variants"] != ["R1", "R2"] or config["epochs"] != 30 or config["early_stop_patience"] != 8:
        raise RuntimeError("training contract mismatch")

    with np.load(args.output / "PUBLIC_FULL_NPF_OUTPUTS.npz") as arrays:
        labels = arrays["true_label"]
        if labels.shape != (3589,) or not np.array_equal(arrays["row_index"], np.arange(3589)):
            raise RuntimeError("Public row alignment mismatch")
        for model in ("full", "npf"):
            raw = arrays[f"{model}_raw_logits"]
            flip = arrays[f"{model}_flip_logits"]
            tta = arrays[f"{model}_tta_logits"]
            if raw.shape != (3589, 7) or not np.isfinite(raw).all() or not np.array_equal(tta, (0.5 * (raw + flip)).astype(np.float32)):
                raise RuntimeError(f"malformed Phase-0 logits: {model}")
            for view, logits in (("raw", raw), ("tta", tta)):
                actual = metrics(labels, logits)
                expected = audit["metrics"]["FULL" if model == "full" else "NO_PIXEL_FUSION"][view]
                for key in ("loss", "accuracy", "macro_f1"):
                    close(actual[key], expected[key])
        for name, shape in (("npf_pixel_readout_raw", (3589, 128)), ("npf_pixel_readout_flip", (3589, 128)), ("npf_motif_readout_raw", (3589, 384)), ("npf_motif_readout_flip", (3589, 384))):
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise RuntimeError(f"readout mismatch: {name}")
        for view in ("raw", "tta"):
            full_logits = arrays[f"full_{view}_logits"]
            npf_logits = arrays[f"npf_{view}_logits"]
            actual_disagreement = disagreement(labels, full_logits.argmax(1), npf_logits.argmax(1))
            for key, value in actual_disagreement.items():
                close(value, audit["views"][view]["disagreement"][key], tolerance=1e-15)
            equal = metrics(labels, 0.5 * (full_logits + npf_logits))
            for key in ("loss", "accuracy", "macro_f1"):
                close(equal[key], audit["views"][view]["equal_logit_fusion"][key])
        sweep = read_csv(args.output / "PUBLIC_FULL_NPF_FUSION_SWEEP.csv")
        if len(sweep) != 202:
            raise RuntimeError("Phase-0 sweep row count mismatch")
        for row in sweep:
            view = row["view"]
            alpha = float(row["alpha_full"])
            fused = alpha * arrays[f"full_{view}_logits"] + (1.0 - alpha) * arrays[f"npf_{view}_logits"]
            actual = metrics(labels, fused)
            for key in ("loss", "accuracy", "macro_f1"):
                close(actual[key], float(row[key]))

    selected_by_history = {}
    for variant in ("R1", "R2"):
        history = read_csv(args.output / f"{variant}_HISTORY.csv")
        if not 1 <= len(history) <= 30 or [int(row["epoch"]) for row in history] != list(range(1, len(history) + 1)):
            raise RuntimeError(f"history epoch mismatch: {variant}")
        best = max(history, key=lambda row: (float(row["val_accuracy"]), float(row["val_macro_f1"])))
        selected_by_history[variant] = int(best["epoch"])
        checkpoint = torch.load(args.output / f"{variant}_best.pt", map_location="cpu", weights_only=False)
        if checkpoint["variant"] != variant or int(checkpoint["epoch"]) != selected_by_history[variant]:
            raise RuntimeError(f"selected checkpoint mismatch: {variant}")
        if checkpoint["training_config_sha256"] != sha256_file(args.output / "TRAINING_CONFIG.json"):
            raise RuntimeError("checkpoint/config identity mismatch")
        if state_sha256(checkpoint["backbone_state_dict"]) != checkpoint["frozen_backbone_state_sha256"]:
            raise RuntimeError("checkpoint backbone hash mismatch")
        keys = set(checkpoint["residual_state_dict"])
        if not keys or not all(key.startswith("pixel_delta.") or key.startswith("gate.") for key in keys):
            raise RuntimeError("unexpected trainable state")
        if variant == "R1" and any(key.startswith("gate.") for key in keys):
            raise RuntimeError("R1 contains gate state")
        if variant == "R2" and not any(key.startswith("gate.") for key in keys):
            raise RuntimeError("R2 gate state absent")
        if summary["variants"][variant]["selected_epoch"] != selected_by_history[variant]:
            raise RuntimeError("summary selected epoch mismatch")
    expected_selected = max(("R1", "R2"), key=lambda variant: (summary["variants"][variant]["selector"]["accuracy"], summary["variants"][variant]["selector"]["macro_f1"]))
    if summary["selected_variant_by_public"] != expected_selected:
        raise RuntimeError("Public selector winner mismatch")
    if not summary["strict_checks"]["frozen_backbone_bitwise_unchanged"] or not summary["strict_checks"]["private_opened_only_after_both_selectors_frozen"]:
        raise RuntimeError("strict boundary check failed")

    samples = read_csv(args.output / "SELECTED_SAMPLE_DIAGNOSTICS.csv")
    if len(samples) != 3589:
        raise RuntimeError("selected sample row count mismatch")
    labels = np.asarray([int(row["true_label"]) for row in samples])
    baseline = np.asarray([int(row["npf_tta_prediction"]) for row in samples])
    for variant in ("R1", "R2"):
        prediction = np.asarray([int(row[f"{variant.lower()}_tta_prediction"]) for row in samples])
        actual_accuracy = float(np.mean(prediction == labels))
        actual_f1 = float(f1_score(labels, prediction, average="macro", labels=np.arange(7), zero_division=0))
        close(actual_accuracy, summary["variants"][variant]["private"]["tta"]["accuracy"])
        close(actual_f1, summary["variants"][variant]["private"]["tta"]["macro_f1"])
        changes = summary["variants"][variant]["private_change_counts_tta"]
        fixed = int(np.sum((baseline != labels) & (prediction == labels)))
        broken = int(np.sum((baseline == labels) & (prediction != labels)))
        if fixed != changes["npf_wrong_to_residual_correct"] or broken != changes["npf_correct_to_residual_wrong"] or fixed - broken != changes["net_corrected_count"]:
            raise RuntimeError("fixed/broken identity mismatch")
    print(json.dumps({"status": "PASS", "files_verified": len(recorded), "public_samples": 3589, "private_samples": len(samples), "selected_epochs": selected_by_history, "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()
