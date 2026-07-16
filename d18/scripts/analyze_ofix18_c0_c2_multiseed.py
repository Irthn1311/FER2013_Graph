"""Aggregate OFIX18 paired C0/C2 multi-seed results without pooling image predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

TRAINING_SEEDS = (7, 21, 42, 84, 123)
TOPOLOGY_SEEDS = (11, 23, 37, 53, 71)
T_CRITICAL_DF4_975 = 2.7764451051977987


def run_name(cell: str, seed: int) -> str:
    stem = "c0_clean_control" if cell == "C0" else "c2_structure_mode_mix_only"
    return f"d18_ofix18_{stem}_seed{seed}"


def run_dir(cell: str, seed: int, new_root: Path, existing_root: Path) -> Path:
    name = run_name(cell, seed)
    return existing_root / name if seed == 42 else new_root / name


def metric_row(path: Path, mode: str) -> pd.Series:
    frame = pd.read_csv(path)
    rows = frame[frame["mode"].eq(mode)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one {mode} row in {path}, got {len(rows)}")
    return rows.iloc[0]


def selected_training_state(source: Path, checkpoint: str) -> dict[str, Any]:
    checkpoint_path = source / "checkpoints" / f"{checkpoint}.pt"
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    epoch = int(payload.get("best_epoch") if checkpoint == "best" else payload.get("epoch"))
    history = pd.read_csv(source / "train_log.csv")
    rows = history[history["epoch"].astype(int).eq(epoch)]
    if len(rows) != 1:
        raise RuntimeError(f"Epoch {epoch} not found uniquely in {source / 'train_log.csv'}")
    row = rows.iloc[0]
    train_macro = float(row["train_macro_f1"])
    val_macro = float(row["val_macro_f1"])
    return {
        "selected_epoch": epoch,
        "train_macro_f1": train_macro,
        "val_macro_f1": val_macro,
        "train_val_macro_gap": train_macro - val_macro,
    }


def expected_paths(
    evaluation_root: Path,
    new_run_root: Path,
    existing_run_root: Path,
) -> list[Path]:
    paths: list[Path] = []
    for cell in ("C0", "C2"):
        for seed in TRAINING_SEEDS:
            source = run_dir(cell, seed, new_run_root, existing_run_root)
            paths += [
                source / "checkpoints/best.pt",
                source / "checkpoints/last.pt",
                source / "train_log.csv",
            ]
            for checkpoint in ("best", "last"):
                base = evaluation_root / run_name(cell, seed) / checkpoint
                paths += [
                    base / "full_official/counterfactual_metrics.csv",
                    base / "locked_core/counterfactual_metrics.csv",
                ]
                paths += [
                    base / f"locked_topology_seed{topology_seed}/counterfactual_metrics.csv"
                    for topology_seed in TOPOLOGY_SEEDS
                ]
    return paths


def model_metrics(
    cell: str,
    seed: int,
    checkpoint: str,
    evaluation_root: Path,
    source: Path,
) -> dict[str, Any]:
    base = evaluation_root / run_name(cell, seed) / checkpoint
    full = metric_row(base / "full_official/counterfactual_metrics.csv", "official")
    core_path = base / "locked_core/counterfactual_metrics.csv"
    official = metric_row(core_path, "official")
    remove = metric_row(core_path, "remove_structure")
    shuffle = metric_row(core_path, "shuffle_structure")

    topology_rows: list[dict[str, Any]] = []
    for topology_seed in TOPOLOGY_SEEDS:
        path = base / f"locked_topology_seed{topology_seed}/counterfactual_metrics.csv"
        for mode in ("permute_structure_destinations", "degree_matched_random_structure"):
            row = metric_row(path, mode)
            topology_rows.append({
                "topology_seed": topology_seed,
                "mode": mode,
                "macro_f1": float(row["macro_f1"]),
                "accuracy": float(row["accuracy"]),
                "ece_15bin": float(row["ece_15bin"]),
            })
    topology = pd.DataFrame(topology_rows)
    permutation = topology[topology["mode"].eq("permute_structure_destinations")]
    random = topology[topology["mode"].eq("degree_matched_random_structure")]
    permute_mean = float(permutation["macro_f1"].mean())
    random_mean = float(random["macro_f1"].mean())
    locked_values = [
        float(official["macro_f1"]),
        float(remove["macro_f1"]),
        float(shuffle["macro_f1"]),
        permute_mean,
        random_mean,
    ]
    training = selected_training_state(source, checkpoint)
    result = {
        "cell": cell,
        "training_seed": seed,
        "checkpoint_type": checkpoint,
        "run_name": run_name(cell, seed),
        "full_test_official_accuracy": float(full["accuracy"]),
        "full_test_official_macro_f1": float(full["macro_f1"]),
        "full_test_official_weighted_f1": float(full["weighted_f1"]),
        "full_test_official_nll": float(full["nll"]),
        "full_test_official_brier": float(full["brier_score"]),
        "full_test_official_ece": float(full["ece_15bin"]),
        "full_test_official_entropy": float(full["mean_predictive_entropy"]),
        "locked_official_accuracy": float(official["accuracy"]),
        "locked_official_macro_f1": float(official["macro_f1"]),
        "locked_official_ece": float(official["ece_15bin"]),
        "locked_remove_macro_f1": float(remove["macro_f1"]),
        "locked_shuffle_macro_f1": float(shuffle["macro_f1"]),
        "locked_permute_macro_f1_mean": permute_mean,
        "locked_permute_macro_f1_std": float(permutation["macro_f1"].std(ddof=1)),
        "locked_permute_macro_f1_min": float(permutation["macro_f1"].min()),
        "locked_permute_macro_f1_max": float(permutation["macro_f1"].max()),
        "locked_random_macro_f1_mean": random_mean,
        "locked_random_macro_f1_std": float(random["macro_f1"].std(ddof=1)),
        "locked_random_macro_f1_min": float(random["macro_f1"].min()),
        "locked_random_macro_f1_max": float(random["macro_f1"].max()),
        "robust_min": float(min(locked_values)),
        "robust_avg": float(np.mean(locked_values)),
        "official_to_remove_drop": float(official["macro_f1"] - remove["macro_f1"]),
        "official_to_shuffle_drop": float(official["macro_f1"] - shuffle["macro_f1"]),
        "semantic_structure_advantage": float(official["macro_f1"] - random_mean),
        "residual_structure_contribution": float(official["macro_f1"] - remove["macro_f1"]),
        **training,
    }
    for column in full.index:
        if str(column).startswith(("precision_", "recall_", "f1_", "support_")):
            result[f"full_test_{column}"] = float(full[column])
    for column in official.index:
        if str(column).startswith(("precision_", "recall_", "f1_", "support_")):
            result[f"locked_official_{column}"] = float(official[column])
    for prefix, row in (("locked_remove", remove), ("locked_shuffle", shuffle)):
        for column in row.index:
            if str(column).startswith("f1_"):
                result[f"{prefix}_{column}"] = float(row[column])
    return result


def paired_rows(models: pd.DataFrame, checkpoint: str) -> pd.DataFrame:
    subset = models[models["checkpoint_type"].eq(checkpoint)]
    c0 = subset[subset["cell"].eq("C0")].set_index("training_seed")
    c2 = subset[subset["cell"].eq("C2")].set_index("training_seed")
    if list(c0.index) != list(c2.index) or set(c0.index) != set(TRAINING_SEEDS):
        raise RuntimeError("Paired seed alignment failed")
    metrics = {
        "official_full_test_macro_f1_diff": "full_test_official_macro_f1",
        "official_locked_macro_f1_diff": "locked_official_macro_f1",
        "remove_structure_macro_f1_diff": "locked_remove_macro_f1",
        "shuffle_structure_macro_f1_diff": "locked_shuffle_macro_f1",
        "permuted_structure_macro_f1_diff": "locked_permute_macro_f1_mean",
        "random_structure_macro_f1_diff": "locked_random_macro_f1_mean",
        "robust_min_diff": "robust_min",
        "robust_avg_diff": "robust_avg",
        "official_to_remove_drop_diff": "official_to_remove_drop",
        "official_to_shuffle_drop_diff": "official_to_shuffle_drop",
        "train_validation_gap_diff": "train_val_macro_gap",
        "selected_epoch_diff": "selected_epoch",
        "official_ece_diff": "full_test_official_ece",
    }
    rows = []
    for seed in TRAINING_SEEDS:
        row: dict[str, Any] = {"checkpoint_type": checkpoint, "training_seed": seed}
        for output_name, source_name in metrics.items():
            row[output_name] = float(c2.loc[seed, source_name] - c0.loc[seed, source_name])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_differences(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_columns = [
        column for column in paired.columns
        if column not in {"checkpoint_type", "training_seed"}
    ]
    for checkpoint, group in paired.groupby("checkpoint_type"):
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            half_width = T_CRITICAL_DF4_975 * std / math.sqrt(len(values))
            rows.append({
                "checkpoint_type": checkpoint,
                "metric": metric,
                "n_training_seeds": len(values),
                "mean": mean,
                "std": std,
                "median": float(np.median(values)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "positive_count": int((values > 0).sum()),
                "negative_count": int((values < 0).sum()),
                "zero_count": int((values == 0).sum()),
                "sign_consistency": int(max((values > 0).sum(), (values < 0).sum())),
                "t_ci95_low": mean - half_width,
                "t_ci95_high": mean + half_width,
            })
    return pd.DataFrame(rows)


def criterion(aggregate: pd.DataFrame, metric: str) -> pd.Series:
    rows = aggregate[
        aggregate["checkpoint_type"].eq("best")
        & aggregate["metric"].eq(metric)
    ]
    if len(rows) != 1:
        raise RuntimeError(metric)
    return rows.iloc[0]


def success_decision(paired: pd.DataFrame, aggregate: pd.DataFrame, models: pd.DataFrame) -> dict[str, Any]:
    best_pairs = paired[paired["checkpoint_type"].eq("best")].set_index("training_seed")
    official = criterion(aggregate, "official_locked_macro_f1_diff")
    remove = criterion(aggregate, "remove_structure_macro_f1_diff")
    shuffle = criterion(aggregate, "shuffle_structure_macro_f1_diff")
    robust = criterion(aggregate, "robust_min_diff")
    official_bad_count = int((best_pairs["official_locked_macro_f1_diff"] < -0.04).sum())
    gates = {
        "official_mean_at_least_minus_2_5pp": bool(official["mean"] >= -0.025),
        "at_most_one_official_loss_worse_than_minus_4pp": bool(official_bad_count <= 1),
        "remove_mean_gain_at_least_8pp": bool(remove["mean"] >= 0.08),
        "remove_positive_at_least_4_of_5": bool(remove["positive_count"] >= 4),
        "shuffle_mean_gain_at_least_6pp": bool(shuffle["mean"] >= 0.06),
        "shuffle_positive_at_least_4_of_5": bool(shuffle["positive_count"] >= 4),
        "robust_min_mean_gain_at_least_8pp": bool(robust["mean"] >= 0.08),
        "robust_min_positive_at_least_4_of_5": bool(robust["positive_count"] >= 4),
    }
    c2_best = models[
        models["cell"].eq("C2") & models["checkpoint_type"].eq("best")
    ]
    semantic = {
        "mean_semantic_structure_advantage": float(c2_best["semantic_structure_advantage"].mean()),
        "positive_semantic_seed_count": int((c2_best["semantic_structure_advantage"] > 0).sum()),
        "mean_residual_structure_contribution": float(c2_best["residual_structure_contribution"].mean()),
        "positive_residual_seed_count": int((c2_best["residual_structure_contribution"] > 0).sum()),
        "secondary_semantic_signal_pass": bool(
            c2_best["semantic_structure_advantage"].mean() > 0
            and (c2_best["semantic_structure_advantage"] > 0).sum() >= 3
        ),
    }
    paired_best = paired[paired["checkpoint_type"].eq("best")]
    mean_train_gap = float(paired_best["train_validation_gap_diff"].mean())
    mean_ece_diff = float(paired_best["official_ece_diff"].mean())
    class_gain_means = {}
    for class_name in ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"):
        c0_values = models[
            models["cell"].eq("C0") & models["checkpoint_type"].eq("best")
        ].set_index("training_seed")[f"locked_remove_f1_{class_name}"]
        c2_values = models[
            models["cell"].eq("C2") & models["checkpoint_type"].eq("best")
        ].set_index("training_seed")[f"locked_remove_f1_{class_name}"]
        class_gain_means[class_name] = float((c2_values - c0_values).mean())
    hidden_exchange_guards = {
        "mean_train_val_gap_increase_at_most_5pp": bool(mean_train_gap <= 0.05),
        "mean_official_ece_increase_at_most_0_05": bool(mean_ece_diff <= 0.05),
        "remove_gain_positive_in_at_least_4_classes": bool(
            sum(value > 0 for value in class_gain_means.values()) >= 4
        ),
    }
    primary_pass = all(gates.values()) and all(hidden_exchange_guards.values())
    if primary_pass:
        decision = "STABLE_C2_SUCCESS"
    elif (
        remove["positive_count"] >= 4
        and shuffle["positive_count"] >= 4
        and robust["positive_count"] >= 4
    ):
        decision = "MIXED_OFFICIAL_TRADEOFF"
    else:
        decision = "SEED_INSTABILITY"
    return {
        "decision": decision,
        "primary_success": primary_pass,
        "primary_gates": gates,
        "hidden_exchange_guards": hidden_exchange_guards,
        "hidden_exchange_diagnostics": {
            "mean_train_validation_gap_difference": mean_train_gap,
            "mean_official_ece_difference": mean_ece_diff,
            "mean_remove_f1_gain_by_class": class_gain_means,
        },
        "official_loss_worse_than_4pp_count": official_bad_count,
        "secondary_semantic_structure": semantic,
        "checkpoint_policy": "best primary; last sensitivity only",
        "no_image_pooling": True,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.fillna("").copy()
    for column in values.select_dtypes(include=[np.number]).columns:
        values[column] = values[column].map(lambda value: f"{value:.6f}")
    header = "| " + " | ".join(values.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in values.to_numpy()]
    return "\n".join([header, rule, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_run_root", default="outputs/d18_runs/ofix18_multiseed")
    parser.add_argument("--existing_run_root", default="outputs/d18_runs/ofix18")
    parser.add_argument(
        "--evaluation_root",
        default="outputs/d18_analysis/ofix18_c0_c2_multiseed_evaluation",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/d18_analysis/ofix18_c0_c2_multiseed_results",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    new_root = Path(args.new_run_root)
    existing_root = Path(args.existing_run_root)
    evaluation_root = Path(args.evaluation_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    missing = [
        str(path)
        for path in expected_paths(evaluation_root, new_root, existing_root)
        if not path.exists()
    ]
    if missing:
        payload = {
            "status": "WAITING_FOR_ARTIFACTS",
            "missing_count": len(missing),
            "missing_artifacts": missing,
        }
        (output / "missing_artifacts.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2))
        if args.strict:
            raise FileNotFoundError(f"{len(missing)} required artifacts are missing")
        return

    model_rows = []
    for cell in ("C0", "C2"):
        for seed in TRAINING_SEEDS:
            source = run_dir(cell, seed, new_root, existing_root)
            for checkpoint in ("best", "last"):
                model_rows.append(
                    model_metrics(
                        cell,
                        seed,
                        checkpoint,
                        evaluation_root,
                        source,
                    )
                )
    models = pd.DataFrame(model_rows)
    paired = pd.concat(
        [paired_rows(models, checkpoint) for checkpoint in ("best", "last")],
        ignore_index=True,
    )
    aggregate = aggregate_differences(paired)
    decision = success_decision(paired, aggregate, models)

    models.to_csv(output / "01_per_model_metrics.csv", index=False)
    paired.to_csv(output / "02_paired_seed_differences.csv", index=False)
    aggregate.to_csv(output / "03_paired_seed_summary.csv", index=False)
    report = [
        "# OFIX18 C0/C2 Paired Multi-Seed Analysis",
        "",
        "Training seeds are the independent units. Image predictions are not pooled across models.",
        "",
        "## Best-checkpoint paired differences",
        "",
        markdown_table(paired[paired["checkpoint_type"].eq("best")]),
        "",
        "## Seed-level uncertainty",
        "",
        markdown_table(aggregate[aggregate["checkpoint_type"].eq("best")]),
        "",
        "## Decision",
        "",
        f"- Decision: **{decision['decision']}**",
        f"- Primary success: **{decision['primary_success']}**",
        f"- Gates: {json.dumps(decision['primary_gates'], sort_keys=True)}",
        f"- Secondary semantic signal: {json.dumps(decision['secondary_semantic_structure'], sort_keys=True)}",
        "",
        "The 95% t intervals quantify training-seed uncertainty with n=5. They are separate from the prior conditional image-level bootstrap.",
        "",
    ]
    (output / "04_multiseed_analysis.md").write_text("\n".join(report), encoding="utf-8")
    summary = {
        "status": "COMPLETE",
        "training_seeds": list(TRAINING_SEEDS),
        "topology_seeds": list(TOPOLOGY_SEEDS),
        "model_metrics": json.loads(models.to_json(orient="records")),
        "paired_differences": json.loads(paired.to_json(orient="records")),
        "paired_summary": json.loads(aggregate.to_json(orient="records")),
        "decision": decision,
        "limitations": [
            "Five training seeds remain a small sample.",
            "All runs use the same FER2013 split.",
            "Topology seeds are not training seeds.",
            "Best is validation-selected and last is sensitivity only.",
            "Locked 715 metrics and full-test metrics are different populations.",
        ],
    }
    (output / "05_machine_readable_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "ANALYSIS_COMPLETE.json").write_text(
        json.dumps({"status": "COMPLETE", "decision": decision["decision"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "COMPLETE", "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
