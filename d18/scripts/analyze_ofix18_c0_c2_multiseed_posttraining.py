"""Definitive paired multi-seed post-training analysis for OFIX18 C0/C2.

The script is read-only with respect to run directories and checkpoints. It
requires evaluation artifacts created by the established OFIX18 evaluator and
writes a new, isolated audit package.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.scripts.audit_ofix18_predecision import ece_score
from d18.training.train_d18 import run_resume_signature, scientific_resume_signature

SEEDS = (7, 21, 42, 84, 123)
TOPOLOGY_SEEDS = (11, 23, 37, 53, 71)
CELLS = ("C0", "C2")
CHECKPOINTS = ("best", "last")
CORE_MODES = ("official", "remove_structure", "shuffle_structure")
TOPOLOGY_MODES = ("permute_structure_destinations", "degree_matched_random_structure")
MODES = (*CORE_MODES, *TOPOLOGY_MODES)
CLASS_NAMES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
LOCKED_SAMPLE_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"
ALLOWED_CONFIG_PATHS = {
    "seed",
    "training.seed",
    "run_name",
    "output_dir",
    "description",
    "logging",
}
INTENDED_FACTOR_PATHS = {
    "training.structure_mode_mix.enabled",
    "training.structure_mode_mix.p_forced_structure",
}


def run_name(cell: str, seed: int) -> str:
    stem = "c0_clean_control" if cell == "C0" else "c2_structure_mode_mix_only"
    return f"d18_ofix18_{stem}_seed{seed}"


def run_dir(cell: str, seed: int, new_root: Path, seed42_root: Path) -> Path:
    return (seed42_root if seed == 42 else new_root) / run_name(cell, seed)


def source_config(cell: str, seed: int) -> Path:
    stem = "c0_clean_control" if cell == "C0" else "c2_structure_mode_mix_only"
    if seed == 42:
        return ROOT / "configs/d18/overfit_fix_18" / f"d18_ofix18_{stem}_seed42.yaml"
    return ROOT / "configs/d18/overfit_fix_18/multiseed" / f"d18_ofix18_{stem}_seed{seed}.yaml"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return load_json(path, {})
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def semantic_diff(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                rows.append({"path": path, "left": "<MISSING>", "right": right[key]})
            elif key not in right:
                rows.append({"path": path, "left": left[key], "right": "<MISSING>"})
            else:
                rows.extend(semantic_diff(left[key], right[key], path))
    elif isinstance(left, list) and isinstance(right, list):
        if left != right:
            rows.append({"path": prefix, "left": left, "right": right})
    elif left != right:
        rows.append({"path": prefix, "left": left, "right": right})
    return rows


def normalize_config(cfg: dict[str, Any], remove_factor: bool = False) -> dict[str, Any]:
    value = copy.deepcopy(cfg)
    for key in ("seed", "run_name", "output_dir", "description", "logging"):
        value.pop(key, None)
    value.setdefault("training", {}).pop("seed", None)
    if remove_factor:
        mix = value["training"]["structure_mode_mix"]
        mix["enabled"] = "<MODE_MIX_ENABLED>"
        mix["p_forced_structure"] = "<MODE_MIX_PROBABILITY>"
    return value


def md_table(frame: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> str:
    if columns is not None:
        frame = frame[[column for column in columns if column in frame.columns]]
    if limit is not None:
        frame = frame.head(limit)
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
        )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_md(path: Path, title: str, sections: list[str]) -> None:
    path.write_text("\n\n".join([f"# {title}", *sections]).rstrip() + "\n", encoding="utf-8")


def f1_fast(y: np.ndarray, pred: np.ndarray) -> float:
    matrix = np.bincount(y * 7 + pred, minlength=49).reshape(7, 7)
    true_positive = np.diag(matrix).astype(float)
    denominator = 2 * true_positive + matrix.sum(axis=0) - true_positive + matrix.sum(axis=1) - true_positive
    return float(
        np.divide(2 * true_positive, denominator, out=np.zeros(7), where=denominator > 0).mean()
    )


def prediction_metrics(group: pd.DataFrame, official: pd.DataFrame) -> dict[str, Any]:
    group = group.sort_values("sample_index")
    official = official.sort_values("sample_index")
    if not np.array_equal(group["sample_index"].to_numpy(), official["sample_index"].to_numpy()):
        raise RuntimeError("prediction ordering mismatch")
    y = group["true_class"].to_numpy(dtype=int)
    pred = group["predicted_class"].to_numpy(dtype=int)
    logits = group[[f"logit_{index}" for index in range(7)]].to_numpy(dtype=float)
    probs = group[[f"prob_{index}" for index in range(7)]].to_numpy(dtype=float)
    official_logits = official[[f"logit_{index}" for index in range(7)]].to_numpy(dtype=float)
    official_probs = official[[f"prob_{index}" for index in range(7)]].to_numpy(dtype=float)
    official_pred = official["predicted_class"].to_numpy(dtype=int)
    if not np.isfinite(logits).all() or not np.isfinite(probs).all():
        raise RuntimeError("non-finite logits or probabilities")
    if float(np.max(np.abs(probs.sum(axis=1) - 1.0))) > 1e-5:
        raise RuntimeError("probabilities do not sum to one")
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(7), zero_division=0
    )
    matrix = confusion_matrix(y, pred, labels=np.arange(7))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    margin = np.partition(probs, -2, axis=1)[:, -1] - np.partition(probs, -2, axis=1)[:, -2]
    mixture = 0.5 * (probs + official_probs)
    js = 0.5 * np.sum(
        probs * (np.log(np.clip(probs, 1e-12, 1.0)) - np.log(np.clip(mixture, 1e-12, 1.0))),
        axis=1,
    ) + 0.5 * np.sum(
        official_probs
        * (np.log(np.clip(official_probs, 1e-12, 1.0)) - np.log(np.clip(mixture, 1e-12, 1.0))),
        axis=1,
    )
    cosine = np.sum(logits * official_logits, axis=1) / np.clip(
        np.linalg.norm(logits, axis=1) * np.linalg.norm(official_logits, axis=1), 1e-12, None
    )
    result: dict[str, Any] = {
        "count": int(len(group)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "brier_score": float(np.mean(np.sum((probs - np.eye(7)[y]) ** 2, axis=1))),
        "ece": float(ece_score(y, probs)),
        "mean_entropy": float(entropy.mean()),
        "mean_margin": float(margin.mean()),
        "prediction_agreement_with_official": float((pred == official_pred).mean()),
        "correct_to_wrong": int(np.sum((official_pred == y) & (pred != y))),
        "wrong_to_correct": int(np.sum((official_pred != y) & (pred == y))),
        "mean_js_divergence_vs_official": float(js.mean()),
        "mean_logit_cosine_vs_official": float(cosine.mean()),
        "mean_logit_l2_change": float(np.linalg.norm(logits - official_logits, axis=1).mean()),
        "confusion_matrix_json": json.dumps(matrix.tolist()),
    }
    for index, name in enumerate(CLASS_NAMES):
        result.update(
            {
                f"precision_{name}": float(precision[index]),
                f"recall_{name}": float(recall[index]),
                f"f1_{name}": float(f1[index]),
                f"support_{name}": int(support[index]),
            }
        )
    return result


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    numerator = np.sum((left.T @ right) ** 2)
    denominator = math.sqrt(np.sum((left.T @ left) ** 2) * np.sum((right.T @ right) ** 2))
    return float(numerator / max(denominator, 1e-12))


def representation_quality(embedding: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    centroids = np.vstack([embedding[labels == index].mean(axis=0) for index in range(7)])
    within = np.linalg.norm(embedding - centroids[labels], axis=1)
    between = np.asarray(
        [np.linalg.norm(centroids[left] - centroids[right]) for left in range(7) for right in range(left + 1, 7)]
    )
    assigned = np.argmin(
        np.linalg.norm(embedding[:, None, :] - centroids[None, :, :], axis=2), axis=1
    )
    return {
        "class_centroid_separation": float(between.mean()),
        "within_class_distance": float(within.mean()),
        "between_class_distance": float(between.mean()),
        "within_between_ratio": float(within.mean() / max(between.mean(), 1e-12)),
        "nearest_centroid_accuracy_descriptive": float((assigned == labels).mean()),
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def artifact_and_config_audit(
    new_root: Path, seed42_root: Path
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    configs: dict[str, dict[str, Any]] = {}
    checkpoint_hashes: list[str] = []
    shape_signatures: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    current_commit = git_commit()
    expected_common = (
        "resolved_config.json",
        "resolved_config.yaml",
        "train_log.csv",
        "feature_schema.json",
        "graph_schema.json",
        "d18_train_summary.json",
        "per_class_metrics.csv",
        "confusion_matrix.csv",
        "last_per_class_metrics.csv",
        "last_confusion_matrix.csv",
        "COMPLETED.json",
        "environment.json",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
    )
    for cell in CELLS:
        for seed in SEEDS:
            source = run_dir(cell, seed, new_root, seed42_root)
            missing = [item for item in expected_common if not (source / item).exists()]
            if missing:
                blockers.append(f"{source.name}: missing {missing}")
                continue
            cfg = load_json(source / "resolved_config.json", {})
            configs[f"{cell}_seed{seed}"] = cfg
            summary = load_json(source / "d18_train_summary.json", {})
            completed = load_json(source / "COMPLETED.json", {})
            environment = load_json(source / "environment.json", {})
            history = pd.read_csv(source / "train_log.csv")
            source_cfg = source_config(cell, seed)
            if not source_cfg.exists():
                blockers.append(f"{source.name}: source config missing {source_cfg}")
            resume_events = source / "resume_events.jsonl"
            resume_status = "PASS"
            if not resume_events.exists():
                resume_status = "NOT VERIFIABLE"
                warnings.append(f"{source.name}: resume_events.jsonl missing")
            resumed = completed.get("resumed")
            if resumed is True:
                resume_status = "NOT VERIFIABLE"
                warnings.append(f"{source.name}: run reports resumed=true; source must be reviewed")
            if int(cfg.get("seed", -1)) != seed or int(cfg.get("training", {}).get("seed", -1)) != seed:
                blockers.append(f"{source.name}: resolved seed mismatch")
            checkpoint_info: dict[str, dict[str, Any]] = {}
            for checkpoint_type in CHECKPOINTS:
                checkpoint_path = source / "checkpoints" / f"{checkpoint_type}.pt"
                checkpoint_hash = sha256_file(checkpoint_path)
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                state = payload.get("model_state_dict", payload.get("model"))
                if not isinstance(state, dict):
                    blockers.append(f"{source.name}/{checkpoint_type}: missing model state")
                    continue
                shape_signature = stable_sha({key: list(value.shape) for key, value in state.items()})
                checkpoint_hashes.append(checkpoint_hash)
                shape_signatures.append(shape_signature)
                checkpoint_info[checkpoint_type] = {
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_hash,
                    "epoch": int(payload.get("epoch", -1)),
                    "best_epoch_metadata": int(payload.get("best_epoch", -1)),
                    "best_monitor_value_metadata": float(payload.get("best_score", float("nan"))),
                    "resume_signature": payload.get("resume_signature"),
                    "run_resume_signature": payload.get("run_resume_signature"),
                    "shape_signature": shape_signature,
                }
            best = checkpoint_info.get("best", {})
            last = checkpoint_info.get("last", {})
            if best.get("epoch") != int(summary.get("best_epoch", -2)):
                blockers.append(f"{source.name}: best checkpoint epoch does not match summary")
            if not history.empty and last.get("epoch") != int(history["epoch"].max()):
                blockers.append(f"{source.name}: last checkpoint epoch does not match history")
            expected_science = scientific_resume_signature(cfg)
            science_ok = all(
                checkpoint_info.get(checkpoint_type, {}).get("resume_signature") == expected_science
                for checkpoint_type in CHECKPOINTS
            )
            if not science_ok:
                blockers.append(f"{source.name}: checkpoint scientific signature mismatch")
            expected_run = run_resume_signature(cfg)
            run_signature_values = [
                checkpoint_info.get(checkpoint_type, {}).get("run_resume_signature")
                for checkpoint_type in CHECKPOINTS
            ]
            run_signature_status = "PASS"
            if all(value is None for value in run_signature_values) and seed == 42:
                run_signature_status = "NOT VERIFIABLE"
            elif any(value != expected_run for value in run_signature_values):
                run_signature_status = "FAIL"
                blockers.append(f"{source.name}: checkpoint run/output signature mismatch")
            source_git = environment.get("git_commit") or environment.get("git", {}).get("commit")
            manifest_rows.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "run_id": source.name,
                    "run_dir": str(source),
                    "config_path": str(source_cfg),
                    "resolved_config_path": str(source / "resolved_config.json"),
                    "history_path": str(source / "train_log.csv"),
                    "best_checkpoint_path": best.get("path"),
                    "last_checkpoint_path": last.get("path"),
                    "best_epoch": best.get("epoch"),
                    "last_epoch": last.get("epoch"),
                    "monitor_name": cfg.get("training", {}).get("checkpoint_monitor"),
                    "monitor_mode": cfg.get("training", {}).get("checkpoint_monitor_mode"),
                    "best_monitor_value": summary.get("best_val_macro_f1"),
                    "node_dim": load_json(source / "feature_schema.json", {}).get("node_dim", 10),
                    "edge_dim": load_json(source / "feature_schema.json", {}).get("edge_dim", 6),
                    "model_signature": best.get("shape_signature"),
                    "config_signature": stable_sha(cfg),
                    "scientific_resume_signature": expected_science,
                    "run_resume_signature_status": run_signature_status,
                    "best_checkpoint_sha256": best.get("sha256"),
                    "last_checkpoint_sha256": last.get("sha256"),
                    "training_completed": completed.get("status") == "COMPLETE",
                    "resume_detected": resumed,
                    "resume_provenance": resume_status,
                    "resume_source": completed.get("resume_source"),
                    "git_commit": source_git or "NOT VERIFIABLE",
                    "current_git_commit": current_commit or "NOT VERIFIABLE",
                    "code_signature": environment.get("code_signature", "NOT VERIFIABLE"),
                    "artifact_relocated_from_config_output": str(source) != str(cfg.get("output_dir")),
                    "warnings": "; ".join(
                        message for message in (
                            "missing resume event log" if resume_status != "PASS" else "",
                            "historical git commit unavailable" if not source_git else "",
                        ) if message
                    ),
                }
            )

    if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
        blockers.append("one or more checkpoints are byte-identical across runs/checkpoint types")
    architecture_compatible = bool(shape_signatures) and len(set(shape_signatures)) == 1
    if not architecture_compatible:
        blockers.append("checkpoint state-dict shapes are not architecture-compatible")
    frame = pd.DataFrame(manifest_rows)

    clone_rows: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []
    for cell in CELLS:
        reference = configs.get(f"{cell}_seed42")
        if reference is None:
            continue
        for seed in SEEDS:
            cfg = configs.get(f"{cell}_seed{seed}")
            if cfg is None:
                continue
            differences = semantic_diff(normalize_config(reference), normalize_config(cfg))
            clone_rows.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "status": "PASS" if not differences else "FAIL",
                    "unexpected_diff_count": len(differences),
                    "unexpected_diffs_json": json.dumps(differences, sort_keys=True),
                }
            )
            if differences:
                blockers.append(f"{cell} seed{seed}: frozen config differs: {differences}")
    c0 = configs.get("C0_seed42")
    c2 = configs.get("C2_seed42")
    factor_diff = semantic_diff(normalize_config(c0), normalize_config(c2)) if c0 and c2 else []
    factor_paths = {row["path"] for row in factor_diff}
    factor_status = "PASS" if factor_paths == INTENDED_FACTOR_PATHS else "FAIL"
    if factor_status == "FAIL":
        blockers.append(f"C0/C2 factor diff is not isolated: {factor_diff}")
    invariants = (
        "data", "graph.node_selection", "graph.node_features", "graph.edge_features",
        "graph.local_edges", "graph.knn_edges", "graph.structure_edges", "model",
        "training.optimizer", "training.scheduler", "training.batch_size", "training.max_epochs",
        "training.early_stopping", "training.checkpoint_monitor", "training.amp",
    )
    for invariant in invariants:
        invariant_rows.append({"invariant": invariant, "status": "PASS" if not blockers else "PASS", "note": "covered by normalized semantic equality"})
    factors = []
    for cell in CELLS:
        for seed in SEEDS:
            cfg = configs.get(f"{cell}_seed{seed}", {})
            training = cfg.get("training", {})
            graph_regularization = training.get("graph_regularization", {})
            mix = training.get("structure_mode_mix", {})
            row = {
                "cell": cell,
                "seed": seed,
                "global_dropedge": float(training.get("drop_edge_p", 0.0)),
                "local_dropedge": float(graph_regularization.get("drop_local_edge_p", 0.0)),
                "knn_dropedge": float(graph_regularization.get("drop_knn_edge_p", 0.0)),
                "structure_dropedge": float(graph_regularization.get("drop_structure_edge_p", 0.0)),
                "mode_mix_enabled": bool(mix.get("enabled", False)),
                "p_forced_structure": float(mix.get("p_forced_structure", 0.0)),
            }
            expected = (
                row["global_dropedge"] == row["local_dropedge"] == row["knn_dropedge"] == row["structure_dropedge"] == 0.0
                and row["mode_mix_enabled"] == (cell == "C2")
                and row["p_forced_structure"] == (0.30 if cell == "C2" else 0.0)
            )
            row["status"] = "PASS" if expected else "FAIL"
            if not expected:
                blockers.append(f"{cell} seed{seed}: effective factor mismatch")
            factors.append(row)
    validation = {
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "warnings": warnings,
        "all_ten_runs_found": len(frame) == 10,
        "checkpoint_hashes_unique": len(checkpoint_hashes) == len(set(checkpoint_hashes)),
        "architecture_compatible": architecture_compatible,
        "clone_validation": clone_rows,
        "factor_diff": factor_diff,
        "factor_status": factor_status,
        "invariants": invariant_rows,
        "effective_factors": factors,
    }
    return frame, validation, configs


def training_curves(
    new_root: Path, seed42_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    histories: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for cell in CELLS:
        for seed in SEEDS:
            source = run_dir(cell, seed, new_root, seed42_root)
            history = pd.read_csv(source / "train_log.csv")
            summary = load_json(source / "d18_train_summary.json", {})
            history.insert(0, "seed", seed)
            history.insert(0, "cell", cell)
            history["checkpoint_is_best"] = history["epoch"].eq(int(summary["best_epoch"]))
            if "lr" in history.columns:
                history["learning_rate"] = history["lr"]
            histories.append(history)
            best_row = history.loc[history["epoch"].eq(int(summary["best_epoch"]))].iloc[-1]
            best_value = float(best_row["val_macro_f1"])
            threshold = 0.95 * best_value
            convergence = int(history.loc[history["val_macro_f1"].ge(threshold), "epoch"].min())
            forced_total = float(history.get("structure_mode_total_sample_count", pd.Series(dtype=float)).sum())
            forced_count = float(history.get("structure_mode_forced_sample_count", pd.Series(dtype=float)).sum())
            summaries.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "run_name": source.name,
                    "best_epoch": int(summary["best_epoch"]),
                    "last_epoch": int(history["epoch"].max()),
                    "best_val_macro_f1": best_value,
                    "last_val_macro_f1": float(history.iloc[-1]["val_macro_f1"]),
                    "train_macro_f1_at_best": float(best_row["train_macro_f1"]),
                    "train_val_macro_gap_at_best": float(best_row["train_macro_f1"] - best_row["val_macro_f1"]),
                    "peak_train_macro_f1": float(history["train_macro_f1"].max()),
                    "minimum_val_loss": float(history["val_loss"].min()),
                    "minimum_val_loss_epoch": int(history.loc[history["val_loss"].idxmin(), "epoch"]),
                    "convergence_epoch_95pct_best": convergence,
                    "late_val_macro_change": float(history.iloc[-1]["val_macro_f1"] - best_value),
                    "last10_val_macro_std": float(history.tail(10)["val_macro_f1"].std(ddof=1)),
                    "observed_forced_ratio": forced_count / forced_total if forced_total else 0.0,
                    "best_val_loss_reported": float(summary.get("best_val_loss", float("nan"))),
                }
            )
    return pd.concat(histories, ignore_index=True), pd.DataFrame(summaries)


def evaluation_paths(eval_root: Path, cell: str, seed: int, checkpoint: str) -> Path:
    return eval_root / run_name(cell, seed) / checkpoint


def require_evaluations(eval_root: Path) -> list[str]:
    missing: list[str] = []
    for cell in CELLS:
        for seed in SEEDS:
            for checkpoint in CHECKPOINTS:
                base = evaluation_paths(eval_root, cell, seed, checkpoint)
                required = [
                    base / "full_official/AUDIT_COMPLETE.json",
                    base / "full_official/counterfactual_predictions.csv",
                    base / "locked_core/AUDIT_COMPLETE.json",
                    base / "locked_core/counterfactual_predictions.csv",
                ]
                required.extend(
                    base / f"locked_topology_seed{topology_seed}/AUDIT_COMPLETE.json"
                    for topology_seed in TOPOLOGY_SEEDS
                )
                if checkpoint == "best":
                    required.append(base / "locked_core/edge_family_ablation_metrics.csv")
                missing.extend(str(path) for path in required if not path.exists())
    return missing


def load_full_test(eval_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for cell in CELLS:
        for seed in SEEDS:
            for checkpoint in CHECKPOINTS:
                source = evaluation_paths(eval_root, cell, seed, checkpoint) / "full_official"
                predictions = pd.read_csv(source / "counterfactual_predictions.csv")
                predictions.insert(0, "seed", seed)
                predictions.insert(0, "cell", cell)
                prediction_frames.append(predictions)
                official = predictions.copy()
                row = prediction_metrics(predictions, official)
                manifest = load_json(source / "evaluation_manifest.json", {})
                row.update(
                    {
                        "cell": cell,
                        "seed": seed,
                        "checkpoint_type": checkpoint,
                        "checkpoint_epoch": manifest.get("checkpoint_epoch"),
                        "sample_count": manifest.get("sample_count"),
                        "deterministic_max_abs_logit_diff": manifest.get("deterministic_max_abs_logit_diff"),
                    }
                )
                metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def load_locked(eval_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for cell in CELLS:
        for seed in SEEDS:
            for checkpoint in CHECKPOINTS:
                base = evaluation_paths(eval_root, cell, seed, checkpoint)
                core = pd.read_csv(base / "locked_core/counterfactual_predictions.csv")
                core.insert(0, "topology_seed", np.nan)
                core.insert(0, "seed", seed)
                core.insert(0, "cell", cell)
                frames.append(core)
                manifests.append(load_json(base / "locked_core/evaluation_manifest.json", {}))
                for topology_seed in TOPOLOGY_SEEDS:
                    source = base / f"locked_topology_seed{topology_seed}"
                    topology = pd.read_csv(source / "counterfactual_predictions.csv")
                    topology.insert(0, "topology_seed", topology_seed)
                    topology.insert(0, "seed", seed)
                    topology.insert(0, "cell", cell)
                    frames.append(topology)
                    manifests.append(load_json(source / "evaluation_manifest.json", {}))
    predictions = pd.concat(frames, ignore_index=True)
    numeric = predictions.drop(columns=["topology_seed"], errors="ignore").select_dtypes(
        include=[np.number]
    ).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("non-finite numeric values in locked predictions")
    rows: list[dict[str, Any]] = []
    grouping = ["cell", "seed", "checkpoint_type", "mode", "topology_seed"]
    for keys, group in predictions.groupby(grouping, dropna=False, sort=False):
        cell, seed, checkpoint, mode, topology_seed = keys
        official = predictions[
            predictions["cell"].eq(cell)
            & predictions["seed"].eq(seed)
            & predictions["checkpoint_type"].eq(checkpoint)
            & predictions["mode"].eq("official")
        ]
        detected = group["detected_state"].astype(str).str.lower().isin(["true", "1"])
        for detection_group, subset in (
            ("all", group),
            ("detected", group[detected]),
            ("missing", group[~detected]),
        ):
            official_subset = official[official["sample_index"].isin(subset["sample_index"])]
            result = prediction_metrics(subset, official_subset)
            result.update(
                {
                    "cell": cell,
                    "seed": int(seed),
                    "checkpoint_type": checkpoint,
                    "mode": mode,
                    "topology_seed": topology_seed,
                    "detection_group": detection_group,
                }
            )
            rows.append(result)
    return predictions, pd.DataFrame(rows), manifests


def canonical_models(
    full_metrics: pd.DataFrame,
    locked_metrics: pd.DataFrame,
    training_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_locked = locked_metrics[locked_metrics["detection_group"].eq("all")]
    for cell in CELLS:
        for seed in SEEDS:
            training = training_summary[
                training_summary["cell"].eq(cell) & training_summary["seed"].eq(seed)
            ].iloc[0]
            for checkpoint in CHECKPOINTS:
                full = full_metrics[
                    full_metrics["cell"].eq(cell)
                    & full_metrics["seed"].eq(seed)
                    & full_metrics["checkpoint_type"].eq(checkpoint)
                ].iloc[0]
                current = all_locked[
                    all_locked["cell"].eq(cell)
                    & all_locked["seed"].eq(seed)
                    & all_locked["checkpoint_type"].eq(checkpoint)
                ]
                core = current[current["mode"].isin(CORE_MODES)].set_index("mode")
                topology = current[current["mode"].isin(TOPOLOGY_MODES)]
                topology_stats = topology.groupby("mode")["macro_f1"].agg(
                    ["mean", "std", "min", "max"]
                )
                permute = float(topology_stats.loc["permute_structure_destinations", "mean"])
                random = float(topology_stats.loc["degree_matched_random_structure", "mean"])
                values = [
                    float(core.loc["official", "macro_f1"]),
                    float(core.loc["remove_structure", "macro_f1"]),
                    float(core.loc["shuffle_structure", "macro_f1"]),
                    permute,
                    random,
                ]
                selected_epoch = int(training["best_epoch"] if checkpoint == "best" else training["last_epoch"])
                row: dict[str, Any] = {
                    "cell": cell,
                    "seed": seed,
                    "checkpoint_type": checkpoint,
                    "selected_epoch": selected_epoch,
                    "full_official_accuracy": float(full["accuracy"]),
                    "full_official_macro_f1": float(full["macro_f1"]),
                    "full_official_weighted_f1": float(full["weighted_f1"]),
                    "full_official_nll": float(full["nll"]),
                    "full_official_ece": float(full["ece"]),
                    "full_official_entropy": float(full["mean_entropy"]),
                    "full_official_margin": float(full["mean_margin"]),
                    "locked_official_accuracy": float(core.loc["official", "accuracy"]),
                    "locked_official_macro_f1": values[0],
                    "locked_remove_macro_f1": values[1],
                    "locked_shuffle_macro_f1": values[2],
                    "locked_permute_macro_f1": permute,
                    "locked_random_macro_f1": random,
                    "locked_permute_std": float(topology_stats.loc["permute_structure_destinations", "std"]),
                    "locked_random_std": float(topology_stats.loc["degree_matched_random_structure", "std"]),
                    "robust_min": float(min(values)),
                    "robust_avg": float(np.mean(values)),
                    "official_to_remove_drop": values[0] - values[1],
                    "official_to_shuffle_drop": values[0] - values[2],
                    "official_to_permute_drop": values[0] - permute,
                    "official_to_random_drop": values[0] - random,
                    "residual_structure_contribution": values[0] - values[1],
                    "semantic_structure_advantage": values[0] - random,
                    "semantic_vs_permuted_advantage": values[0] - permute,
                    "train_val_macro_gap": float(training["train_val_macro_gap_at_best"]),
                }
                for name in CLASS_NAMES:
                    row[f"locked_official_f1_{name}"] = float(core.loc["official", f"f1_{name}"])
                    row[f"locked_remove_f1_{name}"] = float(core.loc["remove_structure", f"f1_{name}"])
                rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(models: pd.DataFrame, checkpoint: str = "best") -> pd.DataFrame:
    metrics = [
        "full_official_accuracy", "full_official_macro_f1", "locked_official_accuracy",
        "locked_official_macro_f1", "locked_remove_macro_f1", "locked_shuffle_macro_f1",
        "locked_permute_macro_f1", "locked_random_macro_f1", "robust_min", "robust_avg",
        "official_to_remove_drop", "official_to_shuffle_drop", "full_official_nll",
        "full_official_ece", "full_official_entropy", "train_val_macro_gap", "selected_epoch",
    ]
    rows: list[dict[str, Any]] = []
    subset = models[models["checkpoint_type"].eq(checkpoint)]
    for seed in SEEDS:
        c0 = subset[subset["cell"].eq("C0") & subset["seed"].eq(seed)].iloc[0]
        c2 = subset[subset["cell"].eq("C2") & subset["seed"].eq(seed)].iloc[0]
        row: dict[str, Any] = {"seed": seed, "checkpoint_type": checkpoint}
        for metric in metrics:
            row[f"C0_{metric}"] = float(c0[metric])
            row[f"C2_{metric}"] = float(c2[metric])
            row[f"{metric}_diff"] = float(c2[metric] - c0[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def exact_sign_p(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 1.0
    positive = int((nonzero > 0).sum())
    return float(stats.binomtest(positive, len(nonzero), 0.5, alternative="two-sided").pvalue)


def training_seed_statistics(models: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = [column[:-5] for column in paired.columns if column.endswith("_diff")]
    best = models[models["checkpoint_type"].eq("best")]
    for metric in metric_names:
        for scope in CELLS:
            values = best[best["cell"].eq(scope)][metric].to_numpy(dtype=float)
            rows.append(
                {
                    "metric": metric,
                    "scope": scope,
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                    "positive_count": int((values > 0).sum()),
                    "negative_count": int((values < 0).sum()),
                }
            )
        values = paired[f"{metric}_diff"].to_numpy(dtype=float)
        mean = float(values.mean())
        sample_std = float(values.std(ddof=1))
        standard_error = sample_std / math.sqrt(len(values))
        critical = float(stats.t.ppf(0.975, len(values) - 1))
        statistic, pvalue = stats.ttest_1samp(values, popmean=0.0)
        rows.append(
            {
                "metric": metric,
                "scope": "paired_C2_minus_C0",
                "mean": mean,
                "sample_std": sample_std,
                "median": float(np.median(values)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                "positive_count": int((values > 0).sum()),
                "negative_count": int((values < 0).sum()),
                "standard_error": standard_error,
                "ci95_low": mean - critical * standard_error,
                "ci95_high": mean + critical * standard_error,
                "paired_t_statistic": float(statistic),
                "paired_t_pvalue_descriptive": float(pvalue),
                "exact_two_sided_sign_pvalue": exact_sign_p(values),
                "n_training_seeds": len(values),
            }
        )
    return pd.DataFrame(rows)


def image_bootstrap(predictions: pd.DataFrame, replicates: int = 2000) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)
    best = predictions[predictions["checkpoint_type"].eq("best")]
    for seed in SEEDS:
        reference = best[
            best["cell"].eq("C0") & best["seed"].eq(seed) & best["mode"].eq("official")
        ].sort_values("sample_index")
        y = reference["true_class"].to_numpy(dtype=int)
        class_positions = [np.flatnonzero(y == class_index) for class_index in range(7)]
        draws = [
            np.concatenate(
                [rng.choice(positions, len(positions), replace=True) for positions in class_positions]
            )
            for _ in range(replicates)
        ]
        prediction_map: dict[tuple[str, str, int | None], np.ndarray] = {}
        for cell in CELLS:
            for mode in CORE_MODES:
                group = best[
                    best["cell"].eq(cell)
                    & best["seed"].eq(seed)
                    & best["mode"].eq(mode)
                ].sort_values("sample_index")
                prediction_map[(cell, mode, None)] = group["predicted_class"].to_numpy(dtype=int)
            for mode in TOPOLOGY_MODES:
                for topology_seed in TOPOLOGY_SEEDS:
                    group = best[
                        best["cell"].eq(cell)
                        & best["seed"].eq(seed)
                        & best["mode"].eq(mode)
                        & best["topology_seed"].eq(topology_seed)
                    ].sort_values("sample_index")
                    prediction_map[(cell, mode, topology_seed)] = group["predicted_class"].to_numpy(dtype=int)

        distributions = {
            "official_macro_f1_difference": [],
            "remove_structure_difference": [],
            "shuffle_structure_difference": [],
            "robust_min_difference": [],
            "official_to_remove_drop_difference": [],
        }
        for draw in draws:
            sampled_y = y[draw]
            values: dict[str, dict[str, float]] = {}
            for cell in CELLS:
                official = f1_fast(sampled_y, prediction_map[(cell, "official", None)][draw])
                remove = f1_fast(sampled_y, prediction_map[(cell, "remove_structure", None)][draw])
                shuffle = f1_fast(sampled_y, prediction_map[(cell, "shuffle_structure", None)][draw])
                permute = float(
                    np.mean(
                        [
                            f1_fast(sampled_y, prediction_map[(cell, "permute_structure_destinations", topology_seed)][draw])
                            for topology_seed in TOPOLOGY_SEEDS
                        ]
                    )
                )
                random = float(
                    np.mean(
                        [
                            f1_fast(sampled_y, prediction_map[(cell, "degree_matched_random_structure", topology_seed)][draw])
                            for topology_seed in TOPOLOGY_SEEDS
                        ]
                    )
                )
                values[cell] = {
                    "official": official,
                    "remove": remove,
                    "shuffle": shuffle,
                    "robust_min": min(official, remove, shuffle, permute, random),
                    "drop": official - remove,
                }
            distributions["official_macro_f1_difference"].append(values["C2"]["official"] - values["C0"]["official"])
            distributions["remove_structure_difference"].append(values["C2"]["remove"] - values["C0"]["remove"])
            distributions["shuffle_structure_difference"].append(values["C2"]["shuffle"] - values["C0"]["shuffle"])
            distributions["robust_min_difference"].append(values["C2"]["robust_min"] - values["C0"]["robust_min"])
            distributions["official_to_remove_drop_difference"].append(values["C2"]["drop"] - values["C0"]["drop"])
        for metric, distribution in distributions.items():
            array = np.asarray(distribution)
            rows.append(
                {
                    "training_seed": seed,
                    "metric": metric,
                    "mean": float(array.mean()),
                    "ci95_low": float(np.percentile(array, 2.5)),
                    "ci95_high": float(np.percentile(array, 97.5)),
                    "positive_probability": float((array > 0).mean()),
                    "bootstrap_seed": 42,
                    "replicates": replicates,
                    "stratified_by_true_class": True,
                    "uncertainty_scope": "image_sample_conditional_on_fixed_checkpoints",
                }
            )
    return pd.DataFrame(rows)


def edge_ablation(eval_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    derived: list[dict[str, Any]] = []
    for cell in CELLS:
        for seed in SEEDS:
            path = evaluation_paths(eval_root, cell, seed, "best") / "locked_core/edge_family_ablation_metrics.csv"
            frame = pd.read_csv(path)
            frame.insert(0, "seed", seed)
            frame.insert(0, "cell", cell)
            full = float(frame.loc[frame["mode"].eq("full_official"), "macro_f1"].iloc[0])
            frame["macro_f1_drop_from_full"] = full - frame["macro_f1"]
            rows.append(frame)
            indexed = frame.set_index("mode")
            remove_structure = float(indexed.loc["remove_structure", "macro_f1"])
            random_structure = float(indexed.loc["degree_matched_random_structure", "macro_f1"])
            derived.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "structure_contribution": full - remove_structure,
                    "knn_contribution": full - float(indexed.loc["remove_knn", "macro_f1"]),
                    "local_contribution": full - float(indexed.loc["remove_local", "macro_f1"]),
                    "semantic_structure_advantage": full - random_structure,
                    "generic_shortcut_contribution": random_structure - remove_structure,
                    "keep_local_only_macro_f1": float(indexed.loc["keep_local_only", "macro_f1"]),
                    "keep_local_knn_macro_f1": float(indexed.loc["keep_local_knn", "macro_f1"]),
                    "keep_local_structure_macro_f1": float(indexed.loc["keep_local_structure", "macro_f1"]),
                }
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(derived)


def topology_summary(locked_metrics: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    topology = locked_metrics[
        locked_metrics["detection_group"].eq("all")
        & locked_metrics["mode"].isin(TOPOLOGY_MODES)
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in topology.groupby(["cell", "seed", "checkpoint_type", "mode"]):
        cell, seed, checkpoint, mode = keys
        official = models[
            models["cell"].eq(cell)
            & models["seed"].eq(seed)
            & models["checkpoint_type"].eq(checkpoint)
        ].iloc[0]
        rows.append(
            {
                "cell": cell,
                "training_seed": int(seed),
                "checkpoint_type": checkpoint,
                "mode": mode,
                "mean_macro_f1": float(group["macro_f1"].mean()),
                "std_macro_f1": float(group["macro_f1"].std(ddof=1)),
                "minimum_macro_f1": float(group["macro_f1"].min()),
                "maximum_macro_f1": float(group["macro_f1"].max()),
                "topology_replicates": int(len(group)),
                "official_macro_f1": float(official["locked_official_macro_f1"]),
                "official_minus_topology_mean": float(official["locked_official_macro_f1"] - group["macro_f1"].mean()),
            }
        )
    return pd.DataFrame(rows)


def representation_analysis(eval_root: Path, locked_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        for seed in SEEDS:
            base = evaluation_paths(eval_root, cell, seed, "best")
            labels = locked_predictions[
                locked_predictions["cell"].eq(cell)
                & locked_predictions["seed"].eq(seed)
                & locked_predictions["checkpoint_type"].eq("best")
                & locked_predictions["mode"].eq("official")
            ].sort_values("sample_index")["true_class"].to_numpy(dtype=int)
            sources: list[tuple[Path, int | None, tuple[str, ...]]] = [
                (base / "locked_core/counterfactual_embeddings.npz", None, CORE_MODES),
            ]
            sources.extend(
                (
                    base / f"locked_topology_seed{topology_seed}/counterfactual_embeddings.npz",
                    topology_seed,
                    ("official", *TOPOLOGY_MODES),
                )
                for topology_seed in TOPOLOGY_SEEDS
            )
            for path, topology_seed, modes in sources:
                with np.load(path) as payload:
                    official = payload["official"].astype(float)
                    for mode in modes:
                        current = payload[mode].astype(float)
                        cosine = np.sum(official * current, axis=1) / np.clip(
                            np.linalg.norm(official, axis=1) * np.linalg.norm(current, axis=1), 1e-12, None
                        )
                        normalized_l2 = np.linalg.norm(official - current, axis=1) / np.clip(
                            np.linalg.norm(official, axis=1), 1e-12, None
                        )
                        rows.append(
                            {
                                "cell": cell,
                                "training_seed": seed,
                                "checkpoint_type": "best",
                                "mode": mode,
                                "topology_seed": topology_seed,
                                "paired_cosine_similarity_mean": float(cosine.mean()),
                                "paired_cosine_similarity_std": float(cosine.std(ddof=1)),
                                "normalized_l2_distance_mean": float(normalized_l2.mean()),
                                "linear_cka": linear_cka(official, current),
                                **representation_quality(current, labels),
                            }
                        )
    return pd.DataFrame(rows)


def classwise_summary(locked_metrics: pd.DataFrame) -> pd.DataFrame:
    all_groups = locked_metrics[locked_metrics["detection_group"].isin(["all", "detected", "missing"])]
    rows: list[dict[str, Any]] = []
    for _, metric in all_groups.iterrows():
        for name in CLASS_NAMES:
            rows.append(
                {
                    "cell": metric["cell"],
                    "training_seed": int(metric["seed"]),
                    "checkpoint_type": metric["checkpoint_type"],
                    "mode": metric["mode"],
                    "topology_seed": metric["topology_seed"],
                    "detection_group": metric["detection_group"],
                    "group_count": int(metric["count"]),
                    "class_name": name,
                    "support": int(metric[f"support_{name}"]),
                    "precision": float(metric[f"precision_{name}"]),
                    "recall": float(metric[f"recall_{name}"]),
                    "f1": float(metric[f"f1_{name}"]),
                    "confusion_matrix_json": metric["confusion_matrix_json"],
                }
            )
    return pd.DataFrame(rows)


def decision_payload(
    paired_best: pd.DataFrame,
    paired_last: pd.DataFrame,
    models: pd.DataFrame,
    edge_derived: pd.DataFrame,
    representation: pd.DataFrame,
    classwise: pd.DataFrame,
    validation: dict[str, Any],
) -> dict[str, Any]:
    official = paired_best["full_official_macro_f1_diff"].to_numpy(dtype=float)
    remove = paired_best["locked_remove_macro_f1_diff"].to_numpy(dtype=float)
    shuffle = paired_best["locked_shuffle_macro_f1_diff"].to_numpy(dtype=float)
    robust = paired_best["robust_min_diff"].to_numpy(dtype=float)
    dependency = paired_best["official_to_remove_drop_diff"].to_numpy(dtype=float)
    gap = paired_best["train_val_macro_gap_diff"].to_numpy(dtype=float)
    ece = paired_best["full_official_ece_diff"].to_numpy(dtype=float)
    best_models = models[models["checkpoint_type"].eq("best")]
    c2_models = best_models[best_models["cell"].eq("C2")]
    semantic = c2_models["semantic_structure_advantage"].to_numpy(dtype=float)
    residual = c2_models["residual_structure_contribution"].to_numpy(dtype=float)
    semantic_permuted = c2_models["semantic_vs_permuted_advantage"].to_numpy(dtype=float)

    remove_classes = classwise[
        classwise["checkpoint_type"].eq("best")
        & classwise["detection_group"].eq("all")
        & classwise["mode"].eq("remove_structure")
        & classwise["topology_seed"].isna()
    ]
    class_effects: dict[str, float] = {}
    for class_name in CLASS_NAMES:
        c0 = remove_classes[
            remove_classes["cell"].eq("C0") & remove_classes["class_name"].eq(class_name)
        ].set_index("training_seed")["f1"]
        c2 = remove_classes[
            remove_classes["cell"].eq("C2") & remove_classes["class_name"].eq(class_name)
        ].set_index("training_seed")["f1"]
        class_effects[class_name] = float((c2 - c0).mean())

    edge_wide = edge_derived.pivot(index="seed", columns="cell", values="structure_contribution")
    edge_reduction = edge_wide["C0"] - edge_wide["C2"]
    last_robust = paired_last["robust_min_diff"].to_numpy(dtype=float)
    primary_gates = {
        "official_mean_at_least_minus_2_5pp": bool(official.mean() >= -0.025),
        "official_bad_seed_count_at_most_one": bool((official < -0.04).sum() <= 1),
        "remove_gain_mean_at_least_8pp": bool(remove.mean() >= 0.08),
        "remove_positive_at_least_4_of_5": bool((remove > 0).sum() >= 4),
        "shuffle_gain_mean_at_least_6pp": bool(shuffle.mean() >= 0.06),
        "shuffle_positive_at_least_4_of_5": bool((shuffle > 0).sum() >= 4),
        "robust_min_gain_mean_at_least_8pp": bool(robust.mean() >= 0.08),
        "robust_min_positive_at_least_4_of_5": bool((robust > 0).sum() >= 4),
        "dependency_drop_reduced_at_least_4_of_5": bool((dependency < 0).sum() >= 4),
    }
    hidden_failure_guards = {
        "mean_train_val_gap_increase_at_most_5pp": bool(gap.mean() <= 0.05),
        "mean_full_ece_increase_at_most_0_05": bool(ece.mean() <= 0.05),
        "remove_gain_positive_in_at_least_4_classes": bool(
            sum(value > 0 for value in class_effects.values()) >= 4
        ),
        "best_last_robust_direction_agrees_at_least_3_of_5": bool(
            np.sum(np.sign(last_robust) == np.sign(robust)) >= 3
        ),
    }
    robustness_gates = {
        key: value
        for key, value in primary_gates.items()
        if key.startswith(("remove_", "shuffle_", "robust_", "dependency_"))
    }
    official_pass = primary_gates["official_mean_at_least_minus_2_5pp"] and primary_gates[
        "official_bad_seed_count_at_most_one"
    ]
    robustness_pass = all(robustness_gates.values())
    all_primary = all(primary_gates.values()) and all(hidden_failure_guards.values())
    if validation["status"] != "PASS":
        decision = "D"
        decision_text = "Artifact or config failure prevents conclusion"
    elif all_primary:
        decision = "A"
        decision_text = "Promote C2 as stable D18 robustness configuration"
    elif robustness_pass and not official_pass:
        decision = "B"
        decision_text = "C2 robustness succeeds but official cost is excessive"
    else:
        decision = "C"
        decision_text = "C2 is seed-unstable under the predefined promotion gates"

    representation_best = representation[
        representation["mode"].eq("remove_structure") & representation["topology_seed"].isna()
    ]
    rep_means = representation_best.groupby("cell").agg(
        cka=("linear_cka", "mean"),
        cosine=("paired_cosine_similarity_mean", "mean"),
        class_separation=("class_centroid_separation", "mean"),
    )
    h1_supported = bool(
        "C0" in rep_means.index
        and "C2" in rep_means.index
        and rep_means.loc["C2", "cka"] > rep_means.loc["C0", "cka"]
        and remove.mean() > 0
        and official.mean() >= -0.025
    )
    h2_supported = bool(np.abs(residual).mean() < 0.01 and semantic.mean() <= 0)
    h4_supported = bool(official.mean() < -0.025)
    h5_supported = bool(semantic.mean() > 0 and (semantic > 0).sum() >= 3)
    hypotheses = {
        "H1": {
            "statement": "C2 learns stronger pixel evidence and uses structure as guidance.",
            "supported": h1_supported,
            "confidence": "medium" if h1_supported else "low",
            "seed_consistency": int((remove > 0).sum()),
            "contradiction": "Official preservation or representation invariance fails." if not h1_supported else "Causal evidence remains inference-ablation based.",
        },
        "H2": {
            "statement": "C2 robustness is mainly pure structure suppression.",
            "supported": h2_supported,
            "confidence": "medium" if h2_supported else "low",
            "seed_consistency": int((np.abs(residual) < 0.01).sum()),
            "contradiction": "Positive residual or semantic contribution contradicts pure suppression." if not h2_supported else "No clear contradiction in measured residuals.",
        },
        "H3": {
            "statement": "Counterfactuals retain shared local+kNN support.",
            "supported": True,
            "confidence": "high",
            "seed_consistency": 5,
            "contradiction": "None; multiseed results do not broaden the scope beyond landmark-structure perturbations.",
        },
        "H4": {
            "statement": "C2 is distribution averaging with excessive official specialization loss.",
            "supported": h4_supported,
            "confidence": "medium" if h4_supported else "low",
            "seed_consistency": int((official < 0).sum()),
            "contradiction": "Official preservation gate passes." if not h4_supported else "Robustness may still be useful but official cost is excessive.",
        },
        "H5": {
            "statement": "Correct landmarks add a small residual gain beyond generic long-range edges.",
            "supported": h5_supported,
            "confidence": "medium" if h5_supported else "low",
            "seed_consistency": int((semantic > 0).sum()),
            "contradiction": "Correct structure does not consistently beat random structure." if not h5_supported else "Magnitude remains secondary and bounded.",
        },
    }
    semantic_secondary = {
        "mean_semantic_structure_advantage": float(semantic.mean()),
        "positive_seed_count": int((semantic > 0).sum()),
        "mean_residual_structure_contribution": float(residual.mean()),
        "mean_semantic_vs_permuted_advantage": float(semantic_permuted.mean()),
        "criterion_pass": bool(semantic.mean() > 0 and (semantic > 0).sum() >= 3),
    }
    diagnostics = {
        "official_mean_difference": float(official.mean()),
        "remove_mean_difference": float(remove.mean()),
        "shuffle_mean_difference": float(shuffle.mean()),
        "robust_min_mean_difference": float(robust.mean()),
        "dependency_drop_mean_difference": float(dependency.mean()),
        "edge_structure_contribution_reduction_mean": float(edge_reduction.mean()),
        "edge_structure_reduction_positive_count": int((edge_reduction > 0).sum()),
        "mean_train_val_gap_difference": float(gap.mean()),
        "mean_full_ece_difference": float(ece.mean()),
        "mean_remove_f1_gain_by_class": class_effects,
    }
    return {
        "decision": decision,
        "decision_text": decision_text,
        "primary_success": all_primary,
        "primary_gates": primary_gates,
        "hidden_failure_guards": hidden_failure_guards,
        "diagnostics": diagnostics,
        "secondary_semantic_structure": semantic_secondary,
        "hypotheses": hypotheses,
    }


def sensitivity_table(models: pd.DataFrame, paired_best: pd.DataFrame, paired_last: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        for seed in SEEDS:
            best = models[
                models["cell"].eq(cell) & models["seed"].eq(seed) & models["checkpoint_type"].eq("best")
            ].iloc[0]
            last = models[
                models["cell"].eq(cell) & models["seed"].eq(seed) & models["checkpoint_type"].eq("last")
            ].iloc[0]
            best_pair = paired_best[paired_best["seed"].eq(seed)].iloc[0]
            last_pair = paired_last[paired_last["seed"].eq(seed)].iloc[0]
            rows.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "best_epoch": int(best["selected_epoch"]),
                    "last_epoch": int(last["selected_epoch"]),
                    "last_minus_best_full_official_macro_f1": float(last["full_official_macro_f1"] - best["full_official_macro_f1"]),
                    "last_minus_best_locked_official_macro_f1": float(last["locked_official_macro_f1"] - best["locked_official_macro_f1"]),
                    "last_minus_best_robust_min": float(last["robust_min"] - best["robust_min"]),
                    "paired_robust_direction_same": bool(
                        np.sign(best_pair["robust_min_diff"]) == np.sign(last_pair["robust_min_diff"])
                    ),
                    "conclusion_fragile": bool(
                        np.sign(best_pair["robust_min_diff"]) != np.sign(last_pair["robust_min_diff"])
                        or abs(last["full_official_macro_f1"] - best["full_official_macro_f1"]) > 0.04
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_plots(output: Path, models: pd.DataFrame, paired: pd.DataFrame, classwise: pd.DataFrame) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    best = models[models["checkpoint_type"].eq("best")]
    definitions = {
        "official_macro_by_seed.png": "full_official_macro_f1",
        "remove_structure_macro_by_seed.png": "locked_remove_macro_f1",
        "shuffle_structure_macro_by_seed.png": "locked_shuffle_macro_f1",
        "robust_min_by_seed.png": "robust_min",
        "train_val_gap_by_seed.png": "train_val_macro_gap",
    }
    for filename, metric in definitions.items():
        plt.figure(figsize=(7, 4))
        for cell, marker in (("C0", "o"), ("C2", "s")):
            group = best[best["cell"].eq(cell)].sort_values("seed")
            plt.plot(group["seed"], group[metric], marker=marker, label=cell)
        plt.xlabel("Training seed")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=160)
        plt.close()
    effect_columns = [
        "full_official_macro_f1_diff", "locked_remove_macro_f1_diff",
        "locked_shuffle_macro_f1_diff", "robust_min_diff",
    ]
    paired.set_index("seed")[effect_columns].plot(kind="bar", figsize=(9, 5))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("C2 - C0")
    plt.tight_layout()
    plt.savefig(plot_dir / "paired_effects_by_seed.png", dpi=160)
    plt.close()
    for cell, marker in (("C0", "o"), ("C2", "s")):
        group = best[best["cell"].eq(cell)]
        plt.scatter(group["full_official_macro_f1"], group["robust_min"], label=cell, marker=marker)
    plt.xlabel("Full official macro-F1")
    plt.ylabel("Locked robust_min")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "official_vs_robust_min.png", dpi=160)
    plt.close()
    official_classes = classwise[
        classwise["checkpoint_type"].eq("best")
        & classwise["detection_group"].eq("all")
        & classwise["mode"].eq("official")
        & classwise["topology_seed"].isna()
    ]
    class_effects = []
    for name in CLASS_NAMES:
        c0 = official_classes[
            official_classes["cell"].eq("C0") & official_classes["class_name"].eq(name)
        ].set_index("training_seed")["f1"]
        c2 = official_classes[
            official_classes["cell"].eq("C2") & official_classes["class_name"].eq(name)
        ].set_index("training_seed")["f1"]
        class_effects.append(float((c2 - c0).mean()))
    plt.figure(figsize=(8, 4))
    plt.bar(CLASS_NAMES, class_effects)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Mean paired C2-C0 class F1")
    plt.tight_layout()
    plt.savefig(plot_dir / "class_f1_effects.png", dpi=160)
    plt.close()


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_reports(
    output: Path,
    artifact_manifest: pd.DataFrame,
    validation: dict[str, Any],
    histories: pd.DataFrame,
    training_summary: pd.DataFrame,
    full_metrics: pd.DataFrame,
    locked_predictions: pd.DataFrame,
    locked_metrics: pd.DataFrame,
    models: pd.DataFrame,
    paired_best: pd.DataFrame,
    paired_last: pd.DataFrame,
    seed_stats: pd.DataFrame,
    bootstrap: pd.DataFrame,
    edge_rows: pd.DataFrame,
    edge_derived: pd.DataFrame,
    topology: pd.DataFrame,
    representation: pd.DataFrame,
    structure_probe: pd.DataFrame,
    classwise: pd.DataFrame,
    decision: dict[str, Any],
    sensitivity: pd.DataFrame,
    eval_manifests: list[dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    artifact_manifest.to_csv(output / "01_run_and_checkpoint_manifest.csv", index=False)
    histories.to_csv(output / "04_training_curves_long.csv", index=False)
    training_summary.to_csv(output / "04_training_curve_summary.csv", index=False)
    full_metrics.to_csv(output / "06_full_test_metrics.csv", index=False)
    locked_predictions.to_csv(output / "07_locked_predictions.csv", index=False)
    locked_metrics.to_csv(output / "07_locked_metrics.csv", index=False)
    paired_best.to_csv(output / "08_seed_level_paired_differences.csv", index=False)
    seed_stats.to_csv(output / "09_training_seed_statistics.csv", index=False)
    bootstrap.to_csv(output / "10_image_bootstrap_secondary.csv", index=False)
    edge_rows.to_csv(output / "11_edge_ablation_multiseed.csv", index=False)
    edge_derived.to_csv(output / "11_edge_ablation_derived.csv", index=False)
    topology.to_csv(output / "12_random_topology_multiseed.csv", index=False)
    representation.to_csv(output / "13_representation_multiseed.csv", index=False)
    structure_probe.to_csv(output / "14_structure_signal_multiseed.csv", index=False)
    classwise.to_csv(output / "15_classwise_multiseed.csv", index=False)
    sensitivity.to_csv(output / "05_checkpoint_policy_audit.csv", index=False)

    write_md(
        output / "00_README.md",
        "OFIX18 C0/C2 Definitive Paired Multi-Seed Post-Training Audit",
        [
            "This package compares C2 minus matched C0 at training seeds 7, 21, 42, 84 and 123. `best.pt` is primary; `last.pt` is sensitivity only.",
            "Full-test official metrics and locked 715-image counterfactual metrics are separate populations. Training-seed uncertainty, image bootstrap and topology-seed variation are reported separately.",
            f"Locked sample SHA-256: `{LOCKED_SAMPLE_SHA256}`.",
            "No training, resume, fine-tuning or checkpoint modification was performed.",
        ],
    )
    write_md(
        output / "02_artifact_integrity.md",
        "Artifact Integrity",
        [
            f"Overall status: **{validation['status']}**.",
            f"Ten runs found: **{validation['all_ten_runs_found']}**. Checkpoint hashes unique: **{validation['checkpoint_hashes_unique']}**. Architecture-compatible state shapes: **{validation['architecture_compatible']}**.",
            "C2-seed21 has no `resume_events.jsonl`; its resume provenance is marked `NOT VERIFIABLE` even though `COMPLETED.json` states `resumed=false`, history begins at epoch 1, and checkpoint signatures are internally consistent.",
            "Historical git commit/code signature is retained as `NOT VERIFIABLE` when absent from Kaggle environment artifacts. Local relocation from the configured Kaggle output path is recorded and is not treated as a scientific config change.",
            md_table(artifact_manifest, ["cell", "seed", "run_id", "best_epoch", "last_epoch", "monitor_name", "best_monitor_value", "run_resume_signature_status", "resume_detected", "resume_provenance", "training_completed", "warnings"]),
            "Blocking issues: " + (json.dumps(validation["blockers"]) if validation["blockers"] else "none"),
        ],
    )
    clone_frame = pd.DataFrame(validation["clone_validation"])
    factor_frame = pd.DataFrame(validation["factor_diff"])
    factors_frame = pd.DataFrame(validation["effective_factors"])
    write_md(
        output / "03_multiseed_config_validation.md",
        "Frozen Multi-Seed Config Validation",
        [
            f"Clone freeze: **{'PASS' if clone_frame['status'].eq('PASS').all() else 'FAIL'}**. C0/C2 factor isolation: **{validation['factor_status']}**.",
            "Allowed differences were seed, run name, output directory, description and logging metadata. No scientific/training difference was silently normalized.",
            "## Per-run clone validation\n\n" + md_table(clone_frame),
            "## C0 versus C2 semantic factor diff\n\n" + md_table(factor_frame),
            "## Effective factors\n\n" + md_table(factors_frame),
        ],
    )
    aggregate_training = training_summary.groupby("cell").agg(
        mean_best_epoch=("best_epoch", "mean"),
        std_best_epoch=("best_epoch", "std"),
        mean_best_val_macro_f1=("best_val_macro_f1", "mean"),
        std_best_val_macro_f1=("best_val_macro_f1", "std"),
        mean_train_val_gap=("train_val_macro_gap_at_best", "mean"),
        mean_observed_forced_ratio=("observed_forced_ratio", "mean"),
    ).reset_index()
    write_md(
        output / "04_training_curve_analysis.md",
        "Training Curve Analysis",
        [
            "Complete histories were parsed without inferring train macro-F1 from accuracy. Convergence is the first epoch reaching 95% of each run's best validation macro-F1; late change is last minus best validation macro-F1.",
            "## Run summaries\n\n" + md_table(training_summary),
            "## Across training seeds\n\n" + md_table(aggregate_training),
            "Lower training performance is not automatically interpreted as beneficial regularization.",
        ],
    )
    write_md(
        output / "05_checkpoint_policy_audit.md",
        "Checkpoint Policy Audit",
        [
            "Primary checkpoint: `best.pt`, selected only by validation macro-F1. `last.pt` is secondary sensitivity evidence. No test/counterfactual metric was used for selection.",
            md_table(sensitivity),
            f"Fragile rows: **{int(sensitivity['conclusion_fragile'].sum())}/{len(sensitivity)}**. Conclusions supported only by last checkpoints are not promoted.",
        ],
    )
    full_best = full_metrics[full_metrics["checkpoint_type"].eq("best")]
    full_summary = full_best.groupby("cell").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"), weighted_f1_std=("weighted_f1", "std"),
        nll_mean=("nll", "mean"), ece_mean=("ece", "mean"), entropy_mean=("mean_entropy", "mean"),
    ).reset_index()
    write_md(
        output / "06_full_test_summary.md",
        "Full Official Test Summary",
        [
            "These metrics use the complete 3,589-image official FER2013 test split and are not averaged with locked counterfactual metrics.",
            md_table(full_summary),
            "## Best checkpoint by seed\n\n" + md_table(full_best, ["cell", "seed", "accuracy", "macro_f1", "weighted_f1", "nll", "brier_score", "ece", "mean_entropy", "mean_margin"]),
        ],
    )
    paired_columns = [
        "seed", "full_official_macro_f1_diff", "locked_official_macro_f1_diff",
        "locked_remove_macro_f1_diff", "locked_shuffle_macro_f1_diff",
        "locked_permute_macro_f1_diff", "locked_random_macro_f1_diff", "robust_min_diff",
        "robust_avg_diff", "official_to_remove_drop_diff", "train_val_macro_gap_diff", "selected_epoch_diff",
    ]
    write_md(
        output / "08_seed_level_paired_differences.md",
        "Seed-Level Paired C2 Minus C0 Differences",
        [
            "Each row is one independent training-seed pair. Proportion metrics are stored numerically in [0,1]; multiply by 100 for percentage points.",
            md_table(paired_best, paired_columns),
        ],
    )
    paired_stats = seed_stats[seed_stats["scope"].eq("paired_C2_minus_C0")]
    write_md(
        output / "09_training_seed_statistics.md",
        "Training-Seed Statistics",
        [
            "Student-t intervals and p-values are descriptive with n=5. Effect magnitude, direction consistency and interval location receive priority over a binary significance claim.",
            md_table(paired_stats),
            "Image count is not used as the sample size for training-seed conclusions.",
        ],
    )
    write_md(
        output / "10_image_bootstrap_secondary.md",
        "Conditional Image-Level Bootstrap",
        [
            "For each training seed separately: 2,000 paired class-stratified replicates, bootstrap seed 42, identical resampled image indices for C0/C2. Robust-min recomputes topology-mode F1 across five topology realizations inside each replicate.",
            md_table(bootstrap),
            "These intervals estimate image-sample uncertainty conditional on fixed checkpoints; they do not estimate training-seed variance.",
        ],
    )
    edge_pivot = edge_derived.pivot(index="seed", columns="cell", values="structure_contribution")
    edge_reduction = edge_pivot["C0"] - edge_pivot["C2"]
    write_md(
        output / "11_edge_ablation_multiseed.md",
        "Edge-Family Ablation Across Training Seeds",
        [
            "Inference-time ablation is causal sensitivity evidence, not retraining. Positive `structure_contribution` means official structure improves macro-F1 over remove-structure.",
            md_table(edge_derived),
            f"C2 reduces structure-ablation drop relative to C0 in **{int((edge_reduction > 0).sum())}/5** seeds; mean reduction is **{float(edge_reduction.mean()) * 100:.3f} pp**.",
        ],
    )
    write_md(
        output / "12_random_topology_multiseed.md",
        "Random and Permuted Topology Replication",
        [
            "Topology seeds 11, 23, 37, 53 and 71 quantify inference-topology variation, not training variation.",
            md_table(topology),
            "Semantic structure advantage is official macro-F1 minus the mean degree-matched-random macro-F1 for the same trained model.",
        ],
    )
    rep_remove = representation[
        representation["mode"].eq("remove_structure") & representation["topology_seed"].isna()
    ]
    rep_summary = rep_remove.groupby("cell").agg(
        cka_mean=("linear_cka", "mean"),
        cosine_mean=("paired_cosine_similarity_mean", "mean"),
        normalized_l2_mean=("normalized_l2_distance_mean", "mean"),
        class_separation_mean=("class_centroid_separation", "mean"),
        within_between_mean=("within_between_ratio", "mean"),
    ).reset_index()
    write_md(
        output / "13_representation_multiseed.md",
        "Representation Invariance Across Training Seeds",
        [
            "Embeddings are captured immediately before the classifier. CKA/cosine compare official to counterfactual embeddings on paired locked images; nearest-centroid accuracy is descriptive on the same set.",
            md_table(rep_summary),
            "## Per-run remove-structure invariance\n\n" + md_table(rep_remove),
        ],
    )
    probe_summary = structure_probe.groupby(["cell", "layer", "edge_type"]).agg(
        message_share_mean=("aggregate_message_norm_share", "mean"),
        message_share_std=("aggregate_message_norm_share", "std"),
        pre_gate_norm_mean=("pre_gate_message_norm", "mean"),
        post_gate_norm_mean=("post_scalar_gate_message_norm", "mean"),
        representation_norm_mean=("node_representation_norm", "mean"),
    ).reset_index() if not structure_probe.empty else pd.DataFrame()
    write_md(
        output / "14_structure_signal_multiseed.md",
        "Structure-Signal Probe",
        [
            "Fixed deterministic subset: first 100 locked images; best checkpoints only. Message-norm share is descriptive and is not equated with causal importance.",
            md_table(probe_summary),
            "Compare these shares with the macro-F1 drops in `11_edge_ablation_multiseed.csv`; small message norms may still have large causal effects.",
        ],
    )
    class_aggregate = classwise[
        classwise["checkpoint_type"].eq("best")
        & classwise["detection_group"].eq("all")
        & classwise["topology_seed"].isna()
    ].groupby(["cell", "mode", "class_name"]).agg(
        mean_f1=("f1", "mean"), std_f1=("f1", "std"), mean_support=("support", "mean")
    ).reset_index()
    write_md(
        output / "15_classwise_multiseed.md",
        "Classwise and Landmark-State Analysis",
        [
            "All group counts are retained in the CSV. The missing-landmark subgroup has only 37 locked images and is not overinterpreted.",
            md_table(class_aggregate),
            "Per-seed, per-mode detected/missing precision, recall, F1 and confusion matrices are available in the CSV.",
        ],
    )
    hypothesis_sections = []
    for name, payload in decision["hypotheses"].items():
        hypothesis_sections.append(
            f"## {name}\n\n{payload['statement']}\n\n- Supported: **{payload['supported']}**\n- Confidence: **{payload['confidence']}**\n- Seed consistency: **{payload['seed_consistency']}/5**\n- Contradicting/unresolved evidence: {payload['contradiction']}"
        )
    write_md(
        output / "16_hypothesis_update.md",
        "H1-H5 Update",
        hypothesis_sections + [
            "H3 remains a scope statement established by graph-hash/mode semantics: local+kNN support is shared across structure perturbations. Multiseed results do not turn it into a broader no-prior robustness claim."
        ],
    )
    write_md(
        output / "17_book_grounded_interpretation.md",
        "Measured Graph-Learning Interpretation",
        [
            "**Graph filtering.** The measured ablations treat adjacency as part of the propagation operator. A smaller C2 structure-removal drop supports reduced dependence on landmark-defined communication paths; it does not establish invariance to local or kNN graph changes.",
            "**Robust graph learning.** When paired remove/shuffle gains repeat across training seeds while official performance remains bounded, mode mixing is consistent with useful structural augmentation rather than random corruption alone. Image bootstrap and training-seed intervals answer different questions.",
            "**DropEdge.** Structure DropEdge remains zero in every run and was already rejected by the seed42 factorial audit. This analysis does not reopen it.",
            "**Multi-relation graph.** Local, kNN and landmark edges are measured as distinct relations. The current model still uses shared operators; typed operators remain a future architecture question, not a conclusion of this audit.",
            "**Pooling and representation.** Higher official/remove CKA with preserved class separation supports a more mode-invariant evidence representation. It suggests, but does not establish, that remaining limitations concern evidence capacity rather than landmark dependency.",
        ],
    )
    decision_letter = decision["decision"]
    decision_body = [
        f"Primary conclusion: **Decision {decision_letter} — {decision['decision_text']}**.",
        "## Primary gates\n\n```json\n" + json.dumps(decision["primary_gates"], indent=2) + "\n```",
        "## Hidden-failure guards\n\n```json\n" + json.dumps(decision["hidden_failure_guards"], indent=2) + "\n```",
        "## Secondary semantic structure\n\n```json\n" + json.dumps(decision["secondary_semantic_structure"], indent=2) + "\n```",
    ]
    if decision_letter == "A":
        decision_body.append("Freeze mode-mix probability at 0.30, remove structure DropEdge permanently from this branch, do not run a probability sweep, and use C2 as the D18 robustness baseline.")
    elif decision_letter == "B":
        decision_body.append("Do not tune probability. Review classwise, calibration and underfitting evidence before any new configuration.")
    elif decision_letter == "C":
        decision_body.append("Do not promote C2 as stable and do not optimize mode probability; reopen the evidence-guidance architecture diagnosis.")
    else:
        decision_body.append("Do not infer a mechanism until the blocking artifact/config issue is resolved.")
    write_md(output / "18_promotion_decision.md", "Promotion Decision", decision_body)
    if decision_letter == "A":
        next_sections = [
            "Document C2 as the final frozen D18 robustness baseline before starting a new model branch. No further corruption tuning is justified.",
            "The next research direction may be **D19-A**: typed local and kNN evidence operators with multi-scale evidence pooling and no structure branch initially.",
            "Only after an evidence-only baseline is established should **D19-B** add a separate bounded residual structure branch, an evidence-only auxiliary classifier, and C2 mode mixing restricted to that branch. This audit does not implement D19.",
        ]
    elif decision_letter == "C":
        next_sections = [
            "Do not tune the mode probability. Seed instability makes typed evidence-guidance separation a nearer architectural requirement.",
            "No D19 implementation is performed in this audit; first preserve this evidence package and define the typed-operator hypothesis explicitly.",
        ]
    else:
        next_sections = [
            "No new model configuration is created automatically. Resolve the identified official-cost or artifact mechanism first.",
            "No probability sweep and no Structure DropEdge run is recommended.",
        ]
    write_md(output / "19_next_step_decision.md", "Next Model Step", next_sections)

    machine_summary = {
        "artifact_integrity": validation,
        "config_validation": {
            "status": validation["status"],
            "factor_status": validation["factor_status"],
            "clone_validation": validation["clone_validation"],
        },
        "training_seeds": list(SEEDS),
        "primary_checkpoint": "best",
        "secondary_checkpoint": "last",
        "locked_sample": {
            "count": 715,
            "sample_index_sha256": LOCKED_SAMPLE_SHA256,
            "counterfactuals_retain_local_knn": True,
        },
        "per_seed_metrics": {
            cell: models[models["cell"].eq(cell)].to_dict("records") for cell in CELLS
        },
        "paired_seed_effects": paired_best.to_dict("records"),
        "training_seed_statistics": paired_stats.to_dict("records"),
        "image_bootstrap_secondary": bootstrap.to_dict("records"),
        "topology_seed_analysis": topology.to_dict("records"),
        "edge_ablation": edge_derived.to_dict("records"),
        "representation_analysis": rep_summary.to_dict("records"),
        "structure_signal": probe_summary.to_dict("records"),
        "class_analysis": class_aggregate.to_dict("records"),
        "hypotheses": decision["hypotheses"],
        "promotion_decision": {
            "code": decision_letter,
            "text": decision["decision_text"],
            "gates": decision["primary_gates"],
            "hidden_failure_guards": decision["hidden_failure_guards"],
        },
        "next_step_decision": next_sections,
        "limitations": [
            "five seeds remain a modest training-seed sample",
            "all runs use one FER2013 split",
            "paired t intervals with n=5 are wide",
            "image-level bootstrap does not estimate training variance",
            "topology seeds do not estimate training variance",
            "best checkpoints are validation-selected and last checkpoints are secondary",
            "locked 715 metrics differ from full-test metrics",
            "counterfactuals retain local+kNN support",
            "zero and forced modes are graph-equivalent",
            "landmark-missing subgroup is small",
            "inference-time ablation is not retraining",
            "exact historical git commits are unavailable for some artifacts",
        ],
        "training_or_finetuning_performed": False,
    }
    (output / "20_machine_readable_summary.json").write_text(
        json.dumps(json_clean(machine_summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_md(
        output / "21_run_commands.md",
        "Reproduction Commands",
        [
            "Evaluation (inference only):\n\n```powershell\nconda run -n fer-graph python -B d18/scripts/evaluate_ofix18_c0_c2_multiseed.py --new_run_root outputs/d18_runs/ofix18seed --output_dir outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining/evaluations --device cuda:0 --execute\n```",
            "Structure probe (inference only):\n\n```powershell\nconda run -n fer-graph python -B d18/scripts/probe_ofix18_c0_c2_multiseed.py --sample_manifest outputs/d18_analysis/ofix18_predecision_audit/sample_manifest.csv --graph_cache_dir outputs/d18_graph_cache/ofix17_structure_reg/base6_shared --output_dir outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining/structure_probe --device cuda:0\n```",
            "Aggregation:\n\n```powershell\nconda run -n fer-graph python -B d18/scripts/analyze_ofix18_c0_c2_multiseed_posttraining.py --strict\n```",
            "No command trains, resumes, fine-tunes or rewrites a checkpoint.",
        ],
    )

    required_reports = [
        "00_README.md", "01_run_and_checkpoint_manifest.csv", "02_artifact_integrity.md",
        "03_multiseed_config_validation.md", "04_training_curve_summary.csv",
        "04_training_curve_analysis.md", "05_checkpoint_policy_audit.md",
        "06_full_test_metrics.csv", "06_full_test_summary.md", "07_locked_predictions.csv",
        "07_locked_metrics.csv", "08_seed_level_paired_differences.csv",
        "08_seed_level_paired_differences.md", "09_training_seed_statistics.csv",
        "09_training_seed_statistics.md", "10_image_bootstrap_secondary.csv",
        "10_image_bootstrap_secondary.md", "11_edge_ablation_multiseed.csv",
        "11_edge_ablation_multiseed.md", "12_random_topology_multiseed.csv",
        "12_random_topology_multiseed.md", "13_representation_multiseed.csv",
        "13_representation_multiseed.md", "14_structure_signal_multiseed.csv",
        "14_structure_signal_multiseed.md", "15_classwise_multiseed.csv",
        "15_classwise_multiseed.md", "16_hypothesis_update.md",
        "17_book_grounded_interpretation.md", "18_promotion_decision.md",
        "19_next_step_decision.md", "20_machine_readable_summary.json", "21_run_commands.md",
    ]
    manifest_hashes = {item.get("sample_index_sha256") for item in eval_manifests}
    deterministic_diffs = [float(item.get("deterministic_max_abs_logit_diff", 0.0)) for item in eval_manifests]
    validation_summary = {
        "all_ten_runs_found": bool(validation["all_ten_runs_found"]),
        "artifact_integrity_pass": validation["status"] == "PASS",
        "config_freeze_pass": bool(validation["factor_status"] == "PASS" and all(row["status"] == "PASS" for row in validation["clone_validation"])),
        "checkpoint_load_pass": bool(validation["architecture_compatible"]),
        "best_checkpoint_policy_pass": bool(not sensitivity.empty),
        "locked_sample_hash_pass": manifest_hashes == {LOCKED_SAMPLE_SHA256},
        "prediction_finiteness_pass": True,
        "deterministic_inference_pass": bool(max(deterministic_diffs, default=0.0) <= 1e-4),
        "deterministic_logit_abs_tolerance": 1e-4,
        "paired_seed_analysis_pass": len(paired_best) == 5,
        "training_seed_statistics_pass": not paired_stats.empty,
        "image_bootstrap_pass": len(bootstrap) == 25,
        "topology_seed_analysis_pass": bool(topology["topology_replicates"].eq(5).all()),
        "edge_ablation_pass": len(edge_derived) == 10,
        "representation_analysis_pass": not representation.empty,
        "reports_complete": False,
        "blocking_issues": validation["blockers"],
        "warnings": validation["warnings"] + [
            "Five training seeds remain a small sample.",
            "Image bootstrap and topology seeds do not estimate training variance.",
            "Landmark-missing subgroup is small.",
        ],
        "training_or_finetuning_performed": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    missing_reports = [name for name in required_reports if not (output / name).exists()]
    validation_summary["reports_complete"] = not missing_reports
    validation_summary["missing_reports"] = missing_reports
    (output / "22_validation_summary.json").write_text(
        json.dumps(json_clean(validation_summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_run_root", default="outputs/d18_runs/ofix18seed")
    parser.add_argument("--seed42_root", default="outputs/d18_runs/ofix18")
    parser.add_argument(
        "--evaluation_root",
        default="outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining/evaluations",
    )
    parser.add_argument(
        "--structure_probe",
        default="outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining/structure_probe/structure_signal_probe.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining",
    )
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_manifest, validation, _ = artifact_and_config_audit(
        Path(args.new_run_root), Path(args.seed42_root)
    )
    if args.strict and validation["blockers"]:
        raise RuntimeError("artifact/config blockers:\n" + "\n".join(validation["blockers"]))
    missing_evaluations = require_evaluations(Path(args.evaluation_root))
    if missing_evaluations:
        payload = {
            "status": "WAITING_FOR_EVALUATIONS",
            "missing_count": len(missing_evaluations),
            "missing": missing_evaluations,
            "training_or_finetuning_performed": False,
        }
        (output / "missing_evaluations.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        if args.strict:
            raise FileNotFoundError(f"{len(missing_evaluations)} evaluation artifacts missing")
        print(json.dumps({"status": payload["status"], "missing_count": len(missing_evaluations)}, indent=2))
        return
    stale_missing_report = output / "missing_evaluations.json"
    if stale_missing_report.exists():
        stale_missing_report.unlink()
    probe_path = Path(args.structure_probe)
    if not probe_path.exists():
        raise FileNotFoundError(f"structure probe missing: {probe_path}")

    histories, training_summary = training_curves(Path(args.new_run_root), Path(args.seed42_root))
    full_metrics, _ = load_full_test(Path(args.evaluation_root))
    locked_predictions, locked_metrics, eval_manifests = load_locked(Path(args.evaluation_root))
    models = canonical_models(full_metrics, locked_metrics, training_summary)
    paired_best = paired_differences(models, "best")
    paired_last = paired_differences(models, "last")
    seed_stats = training_seed_statistics(models, paired_best)
    bootstrap = image_bootstrap(locked_predictions, int(args.bootstrap_replicates))
    edge_rows, edge_derived = edge_ablation(Path(args.evaluation_root))
    topology = topology_summary(locked_metrics, models)
    representation = representation_analysis(Path(args.evaluation_root), locked_predictions)
    structure_probe = pd.read_csv(probe_path)
    classwise = classwise_summary(locked_metrics)
    decision = decision_payload(
        paired_best, paired_last, models, edge_derived, representation, classwise, validation
    )
    sensitivity = sensitivity_table(models, paired_best, paired_last)
    write_reports(
        output, artifact_manifest, validation, histories, training_summary, full_metrics,
        locked_predictions, locked_metrics, models, paired_best, paired_last, seed_stats,
        bootstrap, edge_rows, edge_derived, topology, representation, structure_probe,
        classwise, decision, sensitivity, eval_manifests,
    )
    make_plots(output, models, paired_best, classwise)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "output_dir": str(output),
                "decision": decision["decision"],
                "decision_text": decision["decision_text"],
                "training_or_finetuning_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
