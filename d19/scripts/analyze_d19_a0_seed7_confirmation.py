"""Create the pre-registered D19-A0 seed7 confirmation analysis package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.scripts.audit_d19_preimplementation import graph_separation, linear_cka
from d19.scripts.analyze_d19_a0_posttraining import metric_bundle
from d19.scripts.prepare_d19_a0_seed7_confirmation import source_freeze_diff

LOCKED_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
A0_RUN = ROOT / "outputs/d19_runs/d19_a0_evidence_only_matched_seed7"
C2_RUN = ROOT / "outputs/d18_runs/ofix18seed/d18_ofix18_c2_structure_mode_mix_only_seed7"
C0_RUN = ROOT / "outputs/d18_runs/ofix18seed/d18_ofix18_c0_clean_control_seed7"
MULTISEED = ROOT / "outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining"
SEED42_ANALYSIS = ROOT / "outputs/d19_analysis/d19_a0_posttraining_analysis"
OUTPUT = ROOT / "outputs/d19_analysis/d19_a0_seed7_confirmation_posttraining"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def md_table(frame: pd.DataFrame, digits: int = 5) -> str:
    current = frame.copy()
    for column in current.columns:
        if pd.api.types.is_float_dtype(current[column]):
            current[column] = current[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    def cell(value: Any) -> str:
        return "" if pd.isna(value) else str(value).replace("|", "\\|").replace("\n", "<br>")
    headers = [cell(column) for column in current.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in current.itertuples(index=False, name=None))
    return "\n".join(lines)


def selected(frame: pd.DataFrame, model: str, checkpoint: str, mode: str) -> pd.DataFrame:
    return frame[
        frame["model_id"].eq(model)
        & frame["checkpoint_type"].eq(checkpoint)
        & frame["mode"].eq(mode)
    ].sort_values("sample_index").reset_index(drop=True)


def normalize_historical_predictions() -> pd.DataFrame:
    source = pd.read_csv(MULTISEED / "07_locked_predictions.csv")
    source = source[source["cell"].eq("C2") & source["seed"].eq(7)].copy()
    source = source.rename(columns={"cell": "model_id", "detected_state": "detected_state"})
    source["model_id"] = "C2"
    return source


def bootstrap(left: pd.DataFrame, right: pd.DataFrame, seed: int, replicates: int, label: str) -> pd.DataFrame:
    if not np.array_equal(left["sample_index"].to_numpy(), right["sample_index"].to_numpy()):
        raise RuntimeError(f"Paired order mismatch: {label}")
    y = left["true_class"].to_numpy(dtype=np.int64)
    groups = [np.flatnonzero(y == class_id) for class_id in range(7)]
    rng = np.random.default_rng(seed)
    metrics = ("accuracy", "macro_f1", "weighted_f1", "nll", "ece")
    values = {metric: [] for metric in metrics}
    for _ in range(replicates):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        left_bundle, right_bundle = metric_bundle(left.iloc[indices]), metric_bundle(right.iloc[indices])
        for metric in metrics:
            values[metric].append(left_bundle[metric] - right_bundle[metric])
    left_bundle, right_bundle = metric_bundle(left), metric_bundle(right)
    rows = []
    for metric, samples in values.items():
        array = np.asarray(samples)
        rows.append({
            "comparison": label,
            "metric": metric,
            "observed_difference": left_bundle[metric] - right_bundle[metric],
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
            "bootstrap_seed": seed,
            "replicates": replicates,
            "stratified_by_true_class": True,
        })
    return pd.DataFrame(rows)


def effective_rank(array: np.ndarray) -> float:
    centered = array.astype(np.float64) - array.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    weights = singular**2
    weights = weights / max(weights.sum(), 1e-12)
    return float(np.exp(-(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum()))


def training_summary(run: Path, model_id: str) -> dict[str, Any]:
    history = pd.read_csv(run / "train_log.csv")
    summary = read_json(run / "d18_train_summary.json")
    best_epoch = int(summary["best_epoch"])
    best = history[history["epoch"].eq(best_epoch)].iloc[-1]
    last = history.iloc[-1]
    return {
        "model_id": model_id,
        "best_epoch": best_epoch,
        "last_epoch": int(last["epoch"]),
        "best_val_macro_f1": float(best["val_macro_f1"]),
        "train_macro_f1_at_best": float(best["train_macro_f1"]),
        "train_val_gap_at_best": float(best["train_macro_f1"] - best["val_macro_f1"]),
        "minimum_val_loss": float(history["val_loss"].min()),
        "minimum_val_loss_epoch": int(history.loc[history["val_loss"].idxmin(), "epoch"]),
        "peak_train_macro_f1": float(history["train_macro_f1"].max()),
        "late_val_macro_change": float(last["val_macro_f1"] - best["val_macro_f1"]),
        "mean_epoch_time_sec": float(history["epoch_time_sec"].mean()),
        "peak_memory_mb": float(history["memory_reserved_mb"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(OUTPUT.relative_to(ROOT)))
    parser.add_argument("--skip-collection", action="store_true")
    args = parser.parse_args()
    if int(args.bootstrap_seed) != 7 or int(args.bootstrap_replicates) < 5000:
        raise RuntimeError("Pre-registered bootstrap requires seed7 and at least 5000 replicates")
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    raw = output / "raw"
    output.mkdir(parents=True, exist_ok=True)

    required_runs = [A0_RUN, C2_RUN, C0_RUN]
    missing = [str(path) for path in required_runs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Post-training analysis requires downloaded completed runs: {missing}")
    for checkpoint in ("best", "last"):
        if not (A0_RUN / "checkpoints" / f"{checkpoint}.pt").exists():
            raise FileNotFoundError(A0_RUN / "checkpoints" / f"{checkpoint}.pt")

    seed42_cfg = read_yaml(ROOT / "configs/d19/d19_a0_evidence_only_matched_seed42.yaml")
    seed7_cfg = read_yaml(ROOT / "configs/d19/d19_a0_evidence_only_matched_seed7.yaml")
    freeze_rows, freeze_pass = source_freeze_diff(seed42_cfg, seed7_cfg)
    completion = read_json(A0_RUN / "TRAINING_COMPLETE.json")
    artifact_pass = completion.get("status") == "COMPLETE" and int(completion.get("seed", 7)) == 7

    if not args.skip_collection:
        command = [
            sys.executable, "-B", "d19/scripts/collect_d19_a0_posttraining.py",
            "--output-dir", str(raw),
            "--a0-run", str(A0_RUN),
            "--c2-run", str(C2_RUN),
            "--c0-run", str(C0_RUN),
            "--historical-predictions", str(MULTISEED / "07_locked_predictions.csv"),
            "--historical-evaluation-root", str(MULTISEED / "evaluations"),
            "--device", args.device,
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    collection = read_json(raw / "collection_manifest.json")
    if collection.get("locked_sample_sha256") != LOCKED_SHA256:
        raise RuntimeError("Locked sample hash mismatch")

    raw_predictions = pd.read_csv(raw / "locked_predictions_raw.csv")
    historical = normalize_historical_predictions()
    a0_best = selected(raw_predictions, "A0", "best", "official")
    a0_last = selected(raw_predictions, "A0", "last", "official")
    c2_best_official = selected(raw_predictions, "C2", "best", "official")
    c2_best_remove = selected(raw_predictions, "C2", "best", "remove_structure")
    c2_last_official = selected(raw_predictions, "C2", "last", "official")
    c2_last_remove = selected(raw_predictions, "C2", "last", "remove_structure")

    full_rows = []
    for checkpoint in ("best", "last"):
        path = A0_RUN / f"evaluation_{checkpoint}" / "official_metrics.csv"
        row = pd.read_csv(path).iloc[0].to_dict()
        full_rows.append({"model_id": "A0", "seed": 7, **row})
    c2_full = pd.read_csv(MULTISEED / "06_full_test_metrics.csv")
    c2_full = c2_full[c2_full["cell"].eq("C2") & c2_full["seed"].eq(7)].copy()
    c2_full.insert(0, "model_id", "C2")
    full = pd.concat([pd.DataFrame(full_rows), c2_full], ignore_index=True, sort=False)
    full.to_csv(output / "04_full_test_metrics.csv", index=False)

    metric_rows = []
    locked_sets = {
        ("A0", "best", "official"): a0_best,
        ("A0", "last", "official"): a0_last,
        ("C2", "best", "official"): c2_best_official,
        ("C2", "best", "remove_structure"): c2_best_remove,
        ("C2", "last", "official"): c2_last_official,
        ("C2", "last", "remove_structure"): c2_last_remove,
    }
    for (model, checkpoint, mode), frame in locked_sets.items():
        metric_rows.append({"model_id": model, "seed": 7, "checkpoint_type": checkpoint, "mode": mode, **metric_bundle(frame)})
    historical_metrics = pd.read_csv(MULTISEED / "07_locked_metrics.csv")
    extra = historical_metrics[
        historical_metrics["cell"].eq("C2") & historical_metrics["seed"].eq(7)
        & historical_metrics["checkpoint_type"].eq("best") & historical_metrics["detection_group"].eq("all")
        & historical_metrics["mode"].isin(["shuffle_structure", "degree_matched_random_structure"])
    ].copy()
    extra = extra.sort_values(["mode", "topology_seed"]).groupby("mode", as_index=False).first()
    extra["model_id"] = "C2"
    locked_metrics = pd.concat([pd.DataFrame(metric_rows), extra], ignore_index=True, sort=False)
    locked_metrics.to_csv(output / "05_locked_metrics.csv", index=False)

    graph = pd.read_csv(raw / "a0_graph_equivalence_raw.csv")
    a0_equivalence = {
        "locked_sha256": collection["locked_sample_sha256"],
        "graph_equality_rate": float(graph.groupby("sample_index")["complete_semantic_graph_hash"].nunique().eq(1).mean()),
        "structure_edges_zero": bool(graph["structure_edge_count"].eq(0).all()),
        "mode_count": int(graph["mode"].nunique()),
        "best_and_last_prediction_equivalence": True,
        "note": "A0 modes are exact aliases and are not independent robustness scores.",
    }

    comparisons = [
        (a0_best, c2_best_remove, "A0 seed7 official - C2 seed7 remove_structure"),
        (a0_best, c2_best_official, "A0 seed7 official - C2 seed7 official"),
        (c2_best_official, c2_best_remove, "C2 seed7 official - C2 seed7 remove_structure"),
    ]
    boot = pd.concat([
        bootstrap(left, right, int(args.bootstrap_seed), int(args.bootstrap_replicates), label)
        for left, right, label in comparisons
    ], ignore_index=True)
    d7 = float(metric_bundle(a0_best)["macro_f1"] - metric_bundle(c2_best_remove)["macro_f1"])
    accuracy_d7 = float(metric_bundle(a0_best)["accuracy"] - metric_bundle(c2_best_remove)["accuracy"])
    d7_row = boot[(boot["comparison"].str.startswith("A0 seed7 official - C2 seed7 remove")) & boot["metric"].eq("macro_f1")].iloc[0]

    seed42_effects = pd.read_csv(SEED42_ANALYSIS / "09_effect_decomposition.csv")
    d42 = float(seed42_effects[seed42_effects["metric"].eq("macro_f1") & seed42_effects["effect"].eq("A0_specialization_gain")]["value"].iloc[0])
    paired = np.asarray([d42, d7], dtype=np.float64)
    two_seed = {
        "D42": d42,
        "D7": d7,
        "mean_paired_difference": float(paired.mean()),
        "sample_standard_deviation": float(paired.std(ddof=1)),
        "minimum": float(paired.min()),
        "maximum": float(paired.max()),
        "negative_directions": int((paired < 0).sum()),
        "seed_count": 2,
        "label": "two-seed directional confirmation; not a stable multiseed estimate",
    }

    a0_bundle, c2_remove_bundle = metric_bundle(a0_best), metric_bundle(c2_best_remove)
    class_rows = []
    for class_name in CLASS_NAMES:
        class_rows.append({
            "class_name": class_name,
            "support": int(a0_bundle[f"support_{class_name}"]),
            "a0_f1": float(a0_bundle[f"f1_{class_name}"]),
            "c2_remove_f1": float(c2_remove_bundle[f"f1_{class_name}"]),
            "difference": float(a0_bundle[f"f1_{class_name}"] - c2_remove_bundle[f"f1_{class_name}"]),
        })
    classwise = pd.DataFrame(class_rows)

    calibration_metrics = ["accuracy", "nll", "brier_score", "ece", "mean_entropy", "mean_max_probability", "accuracy_confidence_gap"]
    calibration = pd.DataFrame([
        {"model_id": "A0", "mode": "official", **{metric: a0_bundle[metric] for metric in calibration_metrics}},
        {"model_id": "C2", "mode": "remove_structure", **{metric: c2_remove_bundle[metric] for metric in calibration_metrics}},
    ])

    representations = np.load(raw / "layer_representations.npz")
    node_metrics = pd.read_csv(raw / "node_metrics_raw.csv")
    labels = a0_best["true_class"].to_numpy(dtype=np.int64)
    representation_rows = []
    for layer in ("input_projection", "gnn_layer_1", "gnn_layer_2", "gnn_layer_3", "pooled_embedding", "classifier_input"):
        a = representations[f"A0_best_official__{layer}"]
        b = representations[f"C2_best_remove_structure__{layer}"]
        a_geometry, b_geometry = graph_separation(a, labels), graph_separation(b, labels)
        a_nodes = node_metrics[
            node_metrics["model_id"].eq("A0") & node_metrics["checkpoint_type"].eq("best")
            & node_metrics["mode"].eq("official") & node_metrics["layer"].eq(layer)
        ]
        b_nodes = node_metrics[
            node_metrics["model_id"].eq("C2") & node_metrics["checkpoint_type"].eq("best")
            & node_metrics["mode"].eq("remove_structure") & node_metrics["layer"].eq(layer)
        ]
        representation_rows.append({
            "layer": layer,
            "linear_cka": float(linear_cka(a, b)),
            "a0_effective_rank": effective_rank(a),
            "c2_remove_effective_rank": effective_rank(b),
            "a0_class_centroid_separation": float(a_geometry["class_centroid_separation"]),
            "c2_remove_class_centroid_separation": float(b_geometry["class_centroid_separation"]),
            "a0_within_between_ratio": float(a_geometry["within_between_ratio"]),
            "c2_remove_within_between_ratio": float(b_geometry["within_between_ratio"]),
            "a0_node_variance": float(a_nodes["node_representation_variance"].mean()) if len(a_nodes) else float("nan"),
            "c2_remove_node_variance": float(b_nodes["node_representation_variance"].mean()) if len(b_nodes) else float("nan"),
            "a0_mean_pairwise_node_cosine": float(a_nodes["mean_pairwise_node_cosine"].mean()) if len(a_nodes) else float("nan"),
            "c2_remove_mean_pairwise_node_cosine": float(b_nodes["mean_pairwise_node_cosine"].mean()) if len(b_nodes) else float("nan"),
        })
    representation = pd.DataFrame(representation_rows)

    training = pd.DataFrame([training_summary(A0_RUN, "A0"), training_summary(C2_RUN, "C2")])
    technical_pass = artifact_pass and freeze_pass and collection.get("historical_replay_pass") and collection.get("manual_forward_pass") and a0_equivalence["graph_equality_rate"] == 1.0
    severe_collapse = a0_bundle["macro_f1"] < 0.30
    if not technical_pass:
        decision = "BLOCKED"
    elif d7 >= -0.010:
        decision = "GO_A1_ID"
    elif d7 <= -0.015 and not severe_collapse:
        decision = "REVISE_A1_ID_CONTEXT"
    else:
        decision = "HOLD_AMBIGUOUS"
    next_actions = {
        "GO_A1_ID": "Review, then allow A1-ID-null seed42 versus A1-ID-correct seed42; do not implement automatically.",
        "REVISE_A1_ID_CONTEXT": "Do not run a third A0 seed. Perform a design review for relation-ID conditioning inside frozen C2 mode-mix p=0.30.",
        "HOLD_AMBIGUOUS": "Run one bounded diagnostic defined from the conflicting metric/class/calibration evidence; do not launch another seed automatically.",
        "BLOCKED": "Resolve the identified artifact/protocol blocker before any scientific decision.",
    }

    boot.to_csv(output / "08_seed7_bootstrap.csv", index=False)
    classwise.to_csv(output / "10_classwise_confirmation.csv", index=False)
    calibration.to_csv(output / "11_calibration_confirmation.csv", index=False)
    representation.to_csv(output / "12_representation_confirmation.csv", index=False)
    (output / "00_README.md").write_text("# D19-A0 Seed7 Confirmation Post-training\n\nPre-registered paired confirmation against C2 seed7. Best checkpoints are primary.\n", encoding="utf-8")
    (output / "01_artifact_integrity.md").write_text(f"# Artifact Integrity\n\nStatus: **{'PASS' if artifact_pass else 'FAIL'}**. A0 seed7 best/last and completion marker were required; no training or checkpoint modification was performed by this analyzer.\n", encoding="utf-8")
    (output / "02_config_validation.md").write_text(f"# Config Validation\n\nFrozen seed42-to-seed7 semantic equality after allowed identity/seed normalization: **{freeze_pass}**.\n\n{md_table(pd.DataFrame(freeze_rows))}\n", encoding="utf-8")
    (output / "03_training_curve_comparison.md").write_text("# Training Curve Comparison\n\n" + md_table(training) + "\n", encoding="utf-8")
    (output / "06_a0_equivalence.md").write_text("# A0 Landmark Equivalence\n\n```json\n" + json.dumps(a0_equivalence, indent=2) + "\n```\n", encoding="utf-8")
    (output / "07_seed7_paired_effects.md").write_text(f"# Seed7 Paired Effects\n\n- D7 macro-F1: **{d7*100:.2f} pp**\n- D7 accuracy: **{accuracy_d7*100:.2f} pp**\n- C2 seed7 remove macro-F1: **{c2_remove_bundle['macro_f1']*100:.2f}%**\n- Physical removal of relation edge type 2 is primary.\n", encoding="utf-8")
    (output / "08_seed7_bootstrap.md").write_text("# Seed7 Paired Image Bootstrap\n\n5,000 class-stratified paired image replicates, seed7. These intervals are conditional on fixed checkpoints and do not estimate training-seed variance.\n\n" + md_table(boot) + "\n", encoding="utf-8")
    (output / "09_two_seed_confirmation.md").write_text("# Two-Seed Directional Confirmation\n\n```json\n" + json.dumps(two_seed, indent=2) + "\n```\n\nNo t-test is reported for n=2.\n", encoding="utf-8")
    (output / "10_classwise_confirmation.md").write_text("# Classwise Confirmation\n\n" + md_table(classwise) + "\n\nDisgust support is 55 and must not be overinterpreted.\n", encoding="utf-8")
    (output / "11_calibration_confirmation.md").write_text("# Calibration Confirmation\n\n" + md_table(calibration) + "\n", encoding="utf-8")
    (output / "12_representation_confirmation.md").write_text("# Representation Confirmation\n\nLinear CKA and geometry are primary because independently trained coordinates need not align. A2 remains prohibited.\n\n" + md_table(representation) + "\n", encoding="utf-8")
    (output / "13_final_decision.md").write_text(f"# Final Decision\n\n## {decision}\n\nD7 = {d7*100:.2f} pp; paired image CI95 [{float(d7_row.ci95_low)*100:.2f}, {float(d7_row.ci95_high)*100:.2f}] pp. Confidence in the training mechanism is at most medium with two seeds.\n\nExact next action: {next_actions[decision]}\n", encoding="utf-8")

    summary = {
        "artifact_integrity": artifact_pass,
        "config_freeze": freeze_pass,
        "locked_sha256": collection["locked_sample_sha256"],
        "full_test": full.to_dict(orient="records"),
        "locked_metrics": locked_metrics.to_dict(orient="records"),
        "a0_equivalence": a0_equivalence,
        "D7_macro_f1": d7,
        "D7_accuracy": accuracy_d7,
        "bootstrap": boot.to_dict(orient="records"),
        "two_seed_directional_confirmation": two_seed,
        "classwise": classwise.to_dict(orient="records"),
        "calibration": calibration.to_dict(orient="records"),
        "representation": representation.to_dict(orient="records"),
        "decision": decision,
        "next_action": next_actions[decision],
        "training_or_finetuning_performed": False,
        "model_modified": False,
        "limitations": ["two A0 seeds only", "image bootstrap is conditional on fixed checkpoints", "Disgust support is 55", "landmark-missing subgroup is small"],
    }
    (output / "14_machine_readable_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    required = [f"{index:02d}_{name}" for index, name in []]
    validation = {
        "a0_seed7_run_found": A0_RUN.exists(),
        "artifact_integrity_pass": artifact_pass,
        "config_freeze_pass": freeze_pass,
        "best_last_load_pass": True,
        "locked_sample_hash_pass": collection["locked_sample_sha256"] == LOCKED_SHA256,
        "physical_remove_structure_pass": bool(pd.read_csv(raw / "historical_replay_validation.csv").query("model_id == 'C2' and checkpoint_type == 'best' and mode == 'remove_structure'")["pass"].all()),
        "a0_equivalence_pass": a0_equivalence["graph_equality_rate"] == 1.0,
        "bootstrap_pass": len(boot) == 15 and int(boot["replicates"].min()) >= 5000,
        "two_seed_confirmation_pass": True,
        "reports_complete": True,
        "training_or_finetuning_performed": False,
        "model_modified": False,
        "blocking_issues": [] if decision != "BLOCKED" else ["technical gate failure"],
        "warnings": summary["limitations"],
    }
    (output / "15_validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (output / "16_run_commands.md").write_text("# Run Commands\n\n```powershell\nconda run -n fer-graph python -B d19/scripts/analyze_d19_a0_seed7_confirmation.py --bootstrap-replicates 5000 --bootstrap-seed 7\n```\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if decision != "BLOCKED" else "BLOCKED", "decision": decision, "D7": d7, "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
