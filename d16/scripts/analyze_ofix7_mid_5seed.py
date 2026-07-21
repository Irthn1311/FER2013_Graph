"""Two-stage analysis for the registered OFIX7-mid five-seed replication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from d16.scripts.prepare_ofix7_mid_final_replication import (
    REGISTERED_SEEDS,
    REGISTRATION_HASH_PATH,
    REGISTRATION_PATH,
    ROOT,
    load_json,
    sha256_file,
)


POLICIES = {
    "POLICY_MACRO_F1": "best_val_macro_f1",
    "POLICY_ACCURACY": "best_val_accuracy",
}


def require_policy_lock(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError("Test reveal requires checkpoint_policy_lock.json")
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists() or sha256_file(path) != sidecar.read_text(encoding="utf-8").strip():
        raise RuntimeError("Checkpoint policy lock is absent or modified")
    return load_json(path)


def assert_validation_artifact(path: Path) -> None:
    lowered = path.name.lower()
    if "test" in lowered:
        raise RuntimeError(f"Validation-lock test embargo rejected artifact: {path}")


def read_csv(path: Path) -> list[dict[str, str]]:
    assert_validation_artifact(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_revealed_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_validation_json(path: Path) -> dict[str, Any]:
    assert_validation_artifact(path)
    return load_json(path)


def run_dir_for_seed(root: Path, seed: int) -> Path:
    return root / f"ofix7_mid_seed{seed}"


def metric_value(row: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return float(row[name])
    raise KeyError(names)


def validation_row(run_dir: Path, policy_stem: str) -> dict[str, Any]:
    metrics_path = run_dir / "validation_snapshots" / f"{policy_stem}_metrics.json"
    predictions_path = run_dir / "validation_snapshots" / f"{policy_stem}_predictions.csv"
    per_class_path = run_dir / "validation_snapshots" / f"{policy_stem}_per_class.csv"
    checkpoint_path = run_dir / "checkpoints" / f"{policy_stem}.pt"
    for path in (metrics_path, predictions_path, per_class_path, checkpoint_path):
        assert_validation_artifact(path)
        if not path.exists():
            raise FileNotFoundError(path)
    metrics = read_validation_json(metrics_path)
    per_class = read_csv(per_class_path)
    train_log = read_csv(run_dir / "train_log.csv")
    epoch = int(metric_value(metrics, "epoch"))
    train_epoch = next((row for row in train_log if int(float(row["epoch"])) == epoch), None)
    if train_epoch is None:
        raise RuntimeError(f"Checkpoint epoch {epoch} absent from train_log.csv")
    train_macro = metric_value(train_epoch, "train_macro_f1")
    val_macro = metric_value(metrics, "macro_f1", "val_macro_f1")
    return {
        "epoch": epoch,
        "validation_accuracy": metric_value(metrics, "accuracy", "val_accuracy"),
        "validation_macro_f1": val_macro,
        "train_macro_f1": train_macro,
        "macro_gap_pp": (train_macro - val_macro) * 100.0,
        "per_class": {str(row.get("class_name", row.get("class", row.get("label")))): metric_value(row, "f1", "f1_score") for row in per_class},
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_prediction_count": len(read_csv(predictions_path)),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def validation_lock(runs_root: Path, output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for seed in REGISTERED_SEEDS:
        run_dir = run_dir_for_seed(runs_root, seed)
        if not (run_dir / "REPLICATION_COMPLETE.json").exists():
            raise RuntimeError(f"Replication run is incomplete: {run_dir}")
        details[str(seed)] = {}
        for policy, stem in POLICIES.items():
            item = validation_row(run_dir, stem)
            details[str(seed)][policy] = item
            rows.append({"seed": seed, "policy": policy, **{key: value for key, value in item.items() if key != "per_class"}})
        if not (run_dir / "checkpoints/last.pt").exists():
            raise FileNotFoundError(run_dir / "checkpoints/last.pt")

    macro = [details[str(seed)]["POLICY_MACRO_F1"] for seed in REGISTERED_SEEDS]
    acc = [details[str(seed)]["POLICY_ACCURACY"] for seed in REGISTERED_SEEDS]
    gain_pp = mean([(right["validation_accuracy"] - left["validation_accuracy"]) * 100 for left, right in zip(macro, acc)])
    macro_loss_pp = mean([(right["validation_macro_f1"] - left["validation_macro_f1"]) * 100 for left, right in zip(macro, acc)])
    seed_wins = sum(right["validation_accuracy"] > left["validation_accuracy"] for left, right in zip(macro, acc))
    classes = sorted(set().union(*(item["per_class"] for item in macro), *(item["per_class"] for item in acc)))
    class_losses = {
        name: mean([(right["per_class"].get(name, math.nan) - left["per_class"].get(name, math.nan)) * 100 for left, right in zip(macro, acc)])
        for name in classes
    }
    max_class_loss_pp = max([-value for value in class_losses.values() if math.isfinite(value)], default=0.0)
    gap_increase_pp = mean([right["macro_gap_pp"] - left["macro_gap_pp"] for left, right in zip(macro, acc)])
    sd_increase_pp = (sample_sd([item["validation_accuracy"] for item in acc]) - sample_sd([item["validation_accuracy"] for item in macro])) * 100
    gates = {
        "mean_validation_accuracy_gain_ge_0_50pp": gain_pp >= 0.50,
        "mean_validation_macro_f1_loss_ge_minus_0_50pp": macro_loss_pp >= -0.50,
        "accuracy_seed_wins_ge_3": seed_wins >= 3,
        "no_class_mean_f1_loss_gt_3pp": max_class_loss_pp <= 3.0,
        "macro_gap_increase_le_2pp": gap_increase_pp <= 2.0,
        "accuracy_sd_increase_le_0_50pp": sd_increase_pp <= 0.50,
    }
    selected = "POLICY_ACCURACY" if all(gates.values()) else "POLICY_MACRO_F1"
    lock = {
        "version": "ofix7-mid-policy-lock-v1",
        "registration_sha256": REGISTRATION_HASH_PATH.read_text(encoding="utf-8").strip(),
        "registered_seeds": REGISTERED_SEEDS,
        "stage": "VALIDATION_ONLY",
        "selected_policy": selected,
        "selected_checkpoint": POLICIES[selected] + ".pt",
        "rules_full_precision": {
            "mean_validation_accuracy_gain_pp": gain_pp,
            "mean_validation_macro_f1_change_pp": macro_loss_pp,
            "accuracy_seed_wins": seed_wins,
            "max_class_mean_f1_loss_pp": max_class_loss_pp,
            "macro_gap_increase_pp": gap_increase_pp,
            "accuracy_sd_increase_pp": sd_increase_pp,
        },
        "gates": gates,
        "per_seed": details,
        "test_artifacts_read": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "checkpoint_policy_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lock_path.with_suffix(".sha256").write_text(sha256_file(lock_path) + "\n", encoding="utf-8")
    with (output_root / "validation_checkpoint_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output_root / "validation_checkpoint_comparison.md").write_text(
        "# Validation-only checkpoint policy\n\n"
        f"Selected: **{selected}**. Test artifacts were not read.\n\n"
        "```json\n" + json.dumps(lock["rules_full_precision"], indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return lock


def test_reveal(args: argparse.Namespace) -> dict[str, Any]:
    lock = require_policy_lock(args.checkpoint_policy_lock)
    if not args.prior_dir or not args.prior_dir.exists():
        raise FileNotFoundError("--prior-dir is required for test reveal")
    selected = str(lock["selected_checkpoint"])
    output = args.output_root / "test_reveal"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in REGISTERED_SEEDS:
        run_dir = run_dir_for_seed(args.runs_root, seed)
        for label, checkpoint_name in (("locked", selected), ("last", "last.pt")):
            eval_dir = output / f"seed{seed}_{label}"
            command = [
                sys.executable, "-B", "d16/training/train_d16.py",
                "--config", str(run_dir / "resolved_config.yaml"),
                "--prior_dir", str(args.prior_dir),
                "--output_dir", str(eval_dir),
                "--device", args.device,
                "--eval_only", "--checkpoint", str(run_dir / "checkpoints" / checkpoint_name),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            metrics = read_revealed_csv(eval_dir / "test_metrics.csv")[-1]
            results.append({"seed": seed, "checkpoint_role": label, "checkpoint": checkpoint_name, **metrics})
    path = output / "test_reveal_results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    return {"stage": "TEST_REVEAL", "selected_policy": lock["selected_policy"], "rows": len(results), "output": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["validation-lock", "test-reveal"])
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-policy-lock", type=Path)
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if sha256_file(REGISTRATION_PATH) != REGISTRATION_HASH_PATH.read_text(encoding="utf-8").strip():
        raise RuntimeError("Replication registration hash mismatch")
    result = validation_lock(args.runs_root, args.output_root) if args.stage == "validation-lock" else test_reveal(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
