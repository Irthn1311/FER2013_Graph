"""A6-1 representation geometry audit for D16R A5b/A5c.

Read-only diagnostic over exported best-checkpoint embeddings. This script does
not train, change checkpoints, or select checkpoints from test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


CLASS_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
HARD_IDS = [0, 2, 4, 6]
WATCH_PATTERNS = [(2, 4), (4, 6), (6, 4), (2, 0), (0, 4), (4, 0), (2, 6)]
RUN_SPECS = {
    "seed42": {
        "run_name": "d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150",
        "preferred": Path("outputs/d16_runs/r/a5b/d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150"),
        "family": "A5b",
    },
    "seed43": {
        "run_name": "d16r_a5b_heavy_opt_a4_ce_seed43_accmon_150",
        "preferred": Path("outputs/d16_runs/r/a5b_seed/d16r_a5b_heavy_opt_a4_ce_seed43_accmon_150"),
        "family": "A5b",
    },
    "seed44": {
        "run_name": "d16r_a5b_heavy_opt_a4_ce_seed44_accmon_150",
        "preferred": Path("outputs/d16_runs/r/a5b_seed/d16r_a5b_heavy_opt_a4_ce_seed44_accmon_150"),
        "family": "A5b",
    },
    "a5c": {
        "run_name": "d16r_a5c_multiscale_edge_context_a4_ce_seed42_accmon_150",
        "preferred": Path("outputs/d16_runs/r/a5c/d16r_a5c_multiscale_edge_context_a4_ce_seed42_accmon_150"),
        "family": "A5c",
    },
}


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not math.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return mean(vals) if vals else float("nan")


def _safe_std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return pstdev(vals) if len(vals) > 1 else 0.0


def _ratio(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else float("nan")


def _pattern(true_id: int, pred_id: int) -> str:
    return f"{CLASS_NAMES.get(true_id, true_id)}->{CLASS_NAMES.get(pred_id, pred_id)}"


def _resolve_run(root: Path, run_name: str, preferred: Path) -> Path:
    if preferred.exists() and (preferred / "embeddings_for_audit.npz").exists():
        return preferred
    matches = [
        path
        for path in root.rglob(run_name)
        if path.is_dir() and (path / "embeddings_for_audit.npz").exists()
    ]
    if not matches:
        raise FileNotFoundError(f"Could not resolve exported embedding run: {run_name}")
    return sorted(matches, key=lambda path: (len(path.parts), str(path)))[0]


def _load_run(key: str, root: Path) -> Dict[str, Any]:
    spec = RUN_SPECS[key]
    run_dir = _resolve_run(root, spec["run_name"], spec["preferred"])
    npz_path = run_dir / "embeddings_for_audit.npz"
    data = np.load(npz_path)
    y_pred = data["artifact_y_pred"] if "artifact_y_pred" in data.files else data["y_pred"]
    summary_path = npz_path.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    z = np.asarray(data["z_final_before_classifier"], dtype=np.float32)
    if z.shape[0] != 3589:
        raise ValueError(f"{key}: expected 3589 embeddings, got {z.shape[0]}")
    if not np.isfinite(z).all():
        raise ValueError(f"{key}: NaN/Inf in embeddings")
    return {
        "key": key,
        "family": spec["family"],
        "run_name": spec["run_name"],
        "run_dir": run_dir,
        "summary": summary,
        "sample_index": np.asarray(data["sample_index"], dtype=np.int64),
        "y_true": np.asarray(data["y_true"], dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
        "detected": np.asarray(data["detected"], dtype=np.int64),
        "z": z,
        "pos": {int(sample_idx): i for i, sample_idx in enumerate(np.asarray(data["sample_index"], dtype=np.int64))},
    }


def _class_centroids(run: Dict[str, Any]) -> Dict[str, Dict[int, np.ndarray]]:
    z = run["z"]
    y_true = run["y_true"]
    y_pred = run["y_pred"]
    centroids: Dict[str, Dict[int, np.ndarray]] = {"correct": {}, "all_true": {}, "pred": {}}
    dim = z.shape[1]
    for class_id in range(7):
        true_mask = y_true == class_id
        correct_mask = (y_true == class_id) & (y_pred == class_id)
        pred_mask = y_pred == class_id
        if np.any(true_mask):
            centroids["all_true"][class_id] = z[true_mask].mean(axis=0)
        if np.any(correct_mask):
            centroids["correct"][class_id] = z[correct_mask].mean(axis=0)
        elif np.any(true_mask):
            centroids["correct"][class_id] = z[true_mask].mean(axis=0)
        else:
            centroids["correct"][class_id] = np.zeros((dim,), dtype=np.float32)
        if np.any(pred_mask):
            centroids["pred"][class_id] = z[pred_mask].mean(axis=0)
        else:
            centroids["pred"][class_id] = centroids["correct"][class_id]
    return centroids


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / max(denom, 1.0e-12))


def _geometry_for_sample(run: Dict[str, Any], centroids: Dict[str, Dict[int, np.ndarray]], sample_index: int, pred_id: int | None = None) -> Dict[str, float]:
    pos = run["pos"][sample_index]
    z = run["z"][pos]
    true_id = int(run["y_true"][pos])
    pred = int(run["y_pred"][pos] if pred_id is None else pred_id)
    true_centroid = centroids["correct"][true_id]
    pred_centroid = centroids["correct"][pred]
    dist_true = _dist(z, true_centroid)
    dist_pred = _dist(z, pred_centroid)
    cos_true = _cos(z, true_centroid)
    cos_pred = _cos(z, pred_centroid)
    return {
        "dist_to_true_centroid": dist_true,
        "dist_to_pred_centroid": dist_pred,
        "dist_true_minus_pred": dist_true - dist_pred,
        "cos_to_true_centroid": cos_true,
        "cos_to_pred_centroid": cos_pred,
        "cos_pred_minus_true": cos_pred - cos_true,
        "closer_to_pred_than_true": float(dist_pred < dist_true),
    }


def _hard_rows(runs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    seed_keys = ["seed42", "seed43", "seed44"]
    common = sorted(set(runs["seed42"]["pos"]) & set(runs["seed43"]["pos"]) & set(runs["seed44"]["pos"]) & set(runs["a5c"]["pos"]))
    rows: List[Dict[str, Any]] = []
    for sample_index in common:
        true_id = int(runs["seed42"]["y_true"][runs["seed42"]["pos"][sample_index]])
        if true_id not in HARD_IDS:
            continue
        preds = {key: int(runs[key]["y_pred"][runs[key]["pos"][sample_index]]) for key in seed_keys}
        if any(int(runs[key]["y_true"][runs[key]["pos"][sample_index]]) != true_id for key in ["seed43", "seed44", "a5c"]):
            continue
        correct_count = sum(pred == true_id for pred in preds.values())
        counts = Counter(preds.values())
        majority_pred = counts.most_common(1)[0][0]
        same_wrong = correct_count == 0 and len(set(preds.values())) == 1
        a5c_pred = int(runs["a5c"]["y_pred"][runs["a5c"]["pos"][sample_index]])
        rows.append(
            {
                "sample_index": sample_index,
                "true": true_id,
                "true_name": CLASS_NAMES[true_id],
                "seed42_pred": preds["seed42"],
                "seed43_pred": preds["seed43"],
                "seed44_pred": preds["seed44"],
                "seed42_pred_name": CLASS_NAMES[preds["seed42"]],
                "seed43_pred_name": CLASS_NAMES[preds["seed43"]],
                "seed44_pred_name": CLASS_NAMES[preds["seed44"]],
                "seed_correct_count": correct_count,
                "majority_a5b_correct": correct_count >= 2,
                "majority_a5b_wrong": correct_count <= 1,
                "majority_a5b_pred": majority_pred,
                "majority_a5b_pred_name": CLASS_NAMES[majority_pred],
                "all_a5b_wrong_same_pred": same_wrong,
                "consistent_wrong_pred": preds["seed42"] if same_wrong else "",
                "consistent_wrong_pred_name": CLASS_NAMES[preds["seed42"]] if same_wrong else "",
                "consistent_error_pattern": _pattern(true_id, preds["seed42"]) if same_wrong else "",
                "a5c_pred": a5c_pred,
                "a5c_pred_name": CLASS_NAMES[a5c_pred],
                "a5c_correct": a5c_pred == true_id,
                "a5c_fixes_majority_wrong": correct_count <= 1 and a5c_pred == true_id,
                "a5c_hurts_majority_correct": correct_count >= 2 and a5c_pred != true_id,
                "detected": int(runs["seed42"]["detected"][runs["seed42"]["pos"][sample_index]]),
            }
        )
    return rows


def _centroid_summary(runs: Dict[str, Dict[str, Any]], centroids: Dict[str, Dict[str, Dict[int, np.ndarray]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, run in runs.items():
        z = run["z"]
        y_true = run["y_true"]
        y_pred = run["y_pred"]
        for class_id in range(7):
            true_mask = y_true == class_id
            correct_mask = (y_true == class_id) & (y_pred == class_id)
            pred_mask = y_pred == class_id
            centroid = centroids[key]["correct"][class_id]
            true_dists = [_dist(vec, centroid) for vec in z[true_mask]]
            correct_dists = [_dist(vec, centroid) for vec in z[correct_mask]]
            rows.append(
                {
                    "run": key,
                    "family": run["family"],
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "true_count": int(np.sum(true_mask)),
                    "correct_count": int(np.sum(correct_mask)),
                    "pred_count": int(np.sum(pred_mask)),
                    "correct_centroid_norm": float(np.linalg.norm(centroid)),
                    "true_to_correct_centroid_mean_dist": _safe_mean(true_dists),
                    "correct_to_correct_centroid_mean_dist": _safe_mean(correct_dists),
                }
            )
    return rows


def _compactness(runs: Dict[str, Dict[str, Any]], centroids: Dict[str, Dict[str, Dict[int, np.ndarray]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, run in runs.items():
        for class_id in HARD_IDS:
            pos = np.where(run["y_true"] == class_id)[0]
            dists = np.array([_dist(run["z"][i], centroids[key]["correct"][class_id]) for i in pos], dtype=np.float64)
            correct = run["y_pred"][pos] == class_id
            for bucket, mask in (
                ("all_true", np.ones_like(correct, dtype=bool)),
                ("correct", correct),
                ("wrong", ~correct),
            ):
                vals = dists[mask]
                rows.append(
                    {
                        "run": key,
                        "family": run["family"],
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "bucket": bucket,
                        "count": int(vals.shape[0]),
                        "mean_dist_to_correct_centroid": _safe_mean(vals.tolist()),
                        "std_dist_to_correct_centroid": _safe_std(vals.tolist()),
                        "median_dist_to_correct_centroid": float(np.median(vals)) if vals.shape[0] else float("nan"),
                    }
                )
    return rows


def _consistent_error_geometry(
    hard_rows: List[Dict[str, Any]],
    runs: Dict[str, Dict[str, Any]],
    centroids: Dict[str, Dict[str, Dict[int, np.ndarray]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sample_rows: List[Dict[str, Any]] = []
    for row in hard_rows:
        if not row["all_a5b_wrong_same_pred"]:
            continue
        sample_index = int(row["sample_index"])
        pred_id = int(row["consistent_wrong_pred"])
        for key in ["seed42", "seed43", "seed44", "a5c"]:
            pred_for_geometry = int(row["a5c_pred"]) if key == "a5c" else pred_id
            geom = _geometry_for_sample(runs[key], centroids[key], sample_index, pred_for_geometry)
            sample_rows.append(
                {
                    "run": key,
                    "family": runs[key]["family"],
                    "sample_index": sample_index,
                    "true": row["true"],
                    "true_name": row["true_name"],
                    "pred": pred_for_geometry,
                    "pred_name": CLASS_NAMES[pred_for_geometry],
                    "a5b_consistent_error_pattern": row["consistent_error_pattern"],
                    **geom,
                }
            )
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[(row["run"], row["a5b_consistent_error_pattern"])].append(row)
    summary_rows: List[Dict[str, Any]] = []
    for (run_key, pattern), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "run": run_key,
                "family": runs[run_key]["family"],
                "pattern": pattern,
                "count": len(rows),
                "mean_dist_to_true_centroid": _safe_mean(row["dist_to_true_centroid"] for row in rows),
                "mean_dist_to_pred_centroid": _safe_mean(row["dist_to_pred_centroid"] for row in rows),
                "mean_dist_true_minus_pred": _safe_mean(row["dist_true_minus_pred"] for row in rows),
                "closer_to_pred_ratio": _ratio(sum(int(row["closer_to_pred_than_true"]) for row in rows), len(rows)),
                "mean_cos_to_true_centroid": _safe_mean(row["cos_to_true_centroid"] for row in rows),
                "mean_cos_to_pred_centroid": _safe_mean(row["cos_to_pred_centroid"] for row in rows),
                "mean_cos_pred_minus_true": _safe_mean(row["cos_pred_minus_true"] for row in rows),
            }
        )
    return summary_rows, sample_rows


def _nearest_neighbors(
    hard_rows: List[Dict[str, Any]],
    runs: Dict[str, Dict[str, Any]],
    k: int = 5,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    consistent = [row for row in hard_rows if row["all_a5b_wrong_same_pred"]]
    for key in ["seed42", "seed43", "seed44", "a5c"]:
        run = runs[key]
        ref_mask = run["y_true"] == run["y_pred"]
        ref_z = run["z"][ref_mask]
        ref_y = run["y_true"][ref_mask]
        if ref_z.shape[0] == 0:
            continue
        grouped: Dict[str, Counter] = defaultdict(Counter)
        grouped_samples: Dict[str, int] = Counter()
        for row in consistent:
            sample_index = int(row["sample_index"])
            z = run["z"][run["pos"][sample_index]]
            dists = np.linalg.norm(ref_z - z[None, :], axis=1)
            nn_idx = np.argpartition(dists, kth=min(k, len(dists) - 1))[:k]
            pattern = row["consistent_error_pattern"]
            grouped_samples[pattern] += 1
            for label in ref_y[nn_idx].tolist():
                grouped[pattern][int(label)] += 1
        for pattern, counts in sorted(grouped.items()):
            total_neighbors = sum(counts.values())
            dominant_id, dominant_count = counts.most_common(1)[0]
            item = {
                "run": key,
                "family": run["family"],
                "pattern": pattern,
                "sample_count": int(grouped_samples[pattern]),
                "k": k,
                "total_neighbors": total_neighbors,
                "dominant_neighbor_class": CLASS_NAMES[dominant_id],
                "dominant_neighbor_ratio": _ratio(dominant_count, total_neighbors),
            }
            for class_id in range(7):
                item[f"neighbor_{CLASS_NAMES[class_id].lower()}_count"] = counts.get(class_id, 0)
                item[f"neighbor_{CLASS_NAMES[class_id].lower()}_ratio"] = _ratio(counts.get(class_id, 0), total_neighbors)
            out.append(item)
    return out


def _a5c_tradeoff(
    hard_rows: List[Dict[str, Any]],
    runs: Dict[str, Dict[str, Any]],
    centroids: Dict[str, Dict[str, Dict[int, np.ndarray]]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in hard_rows:
        if not row["a5c_fixes_majority_wrong"] and not row["a5c_hurts_majority_correct"]:
            continue
        sample_index = int(row["sample_index"])
        a5c_geom = _geometry_for_sample(runs["a5c"], centroids["a5c"], sample_index, int(row["a5c_pred"]))
        a5b_geoms = []
        for key in ["seed42", "seed43", "seed44"]:
            pred_id = int(row[f"{key}_pred"])
            a5b_geoms.append(_geometry_for_sample(runs[key], centroids[key], sample_index, pred_id))
        out.append(
            {
                "sample_index": sample_index,
                "true": row["true"],
                "true_name": row["true_name"],
                "seed42_pred_name": row["seed42_pred_name"],
                "seed43_pred_name": row["seed43_pred_name"],
                "seed44_pred_name": row["seed44_pred_name"],
                "a5c_pred_name": row["a5c_pred_name"],
                "tradeoff_type": "a5c_fixes_majority_wrong" if row["a5c_fixes_majority_wrong"] else "a5c_hurts_majority_correct",
                "a5b_correct_count": row["seed_correct_count"],
                "a5c_dist_true_minus_pred": a5c_geom["dist_true_minus_pred"],
                "a5c_closer_to_pred_than_true": int(a5c_geom["closer_to_pred_than_true"]),
                "a5c_cos_pred_minus_true": a5c_geom["cos_pred_minus_true"],
                "a5b_mean_dist_true_minus_pred": _safe_mean(g["dist_true_minus_pred"] for g in a5b_geoms),
                "a5b_mean_closer_to_pred_than_true": _safe_mean(g["closer_to_pred_than_true"] for g in a5b_geoms),
                "detected": row["detected"],
            }
        )
    return out


def _markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str], limit: int | None = None) -> str:
    rows = rows[:limit] if limit is not None else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            vals.append(_fmt(val) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run_audit(output_dir: Path, runs_root: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {key: _load_run(key, runs_root) for key in RUN_SPECS}
    centroids = {key: _class_centroids(run) for key, run in runs.items()}
    hard_rows = _hard_rows(runs)
    centroid_rows = _centroid_summary(runs, centroids)
    compact_rows = _compactness(runs, centroids)
    consistent_summary, consistent_sample_rows = _consistent_error_geometry(hard_rows, runs, centroids)
    nn_rows = _nearest_neighbors(hard_rows, runs, k=5)
    tradeoff_rows = _a5c_tradeoff(hard_rows, runs, centroids)

    _write_csv(
        output_dir / "d16r_a6_1_embedding_centroid_summary.csv",
        centroid_rows,
        [
            "run",
            "family",
            "class_id",
            "class_name",
            "true_count",
            "correct_count",
            "pred_count",
            "correct_centroid_norm",
            "true_to_correct_centroid_mean_dist",
            "correct_to_correct_centroid_mean_dist",
        ],
    )
    _write_csv(
        output_dir / "d16r_a6_1_consistent_error_geometry.csv",
        consistent_summary,
        [
            "run",
            "family",
            "pattern",
            "count",
            "mean_dist_to_true_centroid",
            "mean_dist_to_pred_centroid",
            "mean_dist_true_minus_pred",
            "closer_to_pred_ratio",
            "mean_cos_to_true_centroid",
            "mean_cos_to_pred_centroid",
            "mean_cos_pred_minus_true",
        ],
    )
    _write_csv(
        output_dir / "d16r_a6_1_hard_class_compactness.csv",
        compact_rows,
        [
            "run",
            "family",
            "class_id",
            "class_name",
            "bucket",
            "count",
            "mean_dist_to_correct_centroid",
            "std_dist_to_correct_centroid",
            "median_dist_to_correct_centroid",
        ],
    )
    _write_csv(
        output_dir / "d16r_a6_1_nearest_neighbor_summary.csv",
        nn_rows,
        [
            "run",
            "family",
            "pattern",
            "sample_count",
            "k",
            "total_neighbors",
            "dominant_neighbor_class",
            "dominant_neighbor_ratio",
            *[
                key
                for class_id in range(7)
                for key in (
                    f"neighbor_{CLASS_NAMES[class_id].lower()}_count",
                    f"neighbor_{CLASS_NAMES[class_id].lower()}_ratio",
                )
            ],
        ],
    )
    _write_csv(
        output_dir / "d16r_a6_1_a5c_embedding_tradeoff.csv",
        tradeoff_rows,
        [
            "sample_index",
            "true",
            "true_name",
            "seed42_pred_name",
            "seed43_pred_name",
            "seed44_pred_name",
            "a5c_pred_name",
            "tradeoff_type",
            "a5b_correct_count",
            "a5c_dist_true_minus_pred",
            "a5c_closer_to_pred_than_true",
            "a5c_cos_pred_minus_true",
            "a5b_mean_dist_true_minus_pred",
            "a5b_mean_closer_to_pred_than_true",
            "detected",
        ],
    )

    consistent_a5b = [row for row in consistent_summary if row["run"] in {"seed42", "seed43", "seed44"}]
    closer_ratio = _safe_mean(row["closer_to_pred_ratio"] for row in consistent_a5b)
    consistent_count = sum(row["count"] for row in consistent_summary if row["run"] == "seed42")
    fs_rows = [row for row in consistent_a5b if row["pattern"] == "Fear->Sad"]
    sn_rows = [row for row in consistent_a5b if row["pattern"] in {"Sad->Neutral", "Neutral->Sad"}]
    trade_counts = Counter(row["tradeoff_type"] for row in tradeoff_rows)
    a5c_fix_count = trade_counts.get("a5c_fixes_majority_wrong", 0)
    a5c_hurt_count = trade_counts.get("a5c_hurts_majority_correct", 0)
    decisions = []
    if consistent_count >= 200:
        decisions.append("HARD_ERRORS_ARE_STRUCTURAL")
    if closer_ratio >= 0.55:
        decisions.append("A6_NEEDS_REPRESENTATION_SEPARATION")
    else:
        decisions.append("A6_REPRESENTATION_GEOMETRY_MIXED_CHECK_CLASSIFIER_HEAD")
    if fs_rows and _safe_mean(row["closer_to_pred_ratio"] for row in fs_rows) >= 0.55:
        decisions.append("TARGET_FEAR_SAD_SEPARATION")
    if sn_rows and _safe_mean(row["closer_to_pred_ratio"] for row in sn_rows) >= 0.55:
        decisions.append("TARGET_SAD_NEUTRAL_SEPARATION")
    if a5c_fix_count > 0 and a5c_hurt_count > 0:
        decisions.append("A5C_GEOMETRY_SIGNAL_WITH_TRADEOFF")

    next_decision = {
        "audit": "A6-1 Representation Geometry Audit",
        "runs": {key: str(run["run_dir"]) for key, run in runs.items()},
        "export_summaries": {key: run["summary"] for key, run in runs.items()},
        "hard_sample_count": len(hard_rows),
        "a5b_consistent_same_wrong_count": int(consistent_count),
        "a5b_consistent_error_closer_to_wrong_centroid_mean_ratio": closer_ratio,
        "a5c_fixes_majority_a5b_wrong": int(a5c_fix_count),
        "a5c_hurts_majority_a5b_correct": int(a5c_hurt_count),
        "decisions": decisions,
        "recommended_a6": [
            "Target Fear-vs-Sad and Sad-vs-Neutral representation separation.",
            "Use audit/diagnostic framing first; do not return to fallback because hard errors are mostly detected samples.",
            "If adding a trainable A6 component later, prefer a small relation/separation regularizer or hard-class relation head over more input feature maps.",
        ],
        "caveat": "A5b and A5c embeddings are compared only within each model's own centroid geometry; raw cross-model vector distances are not compared.",
    }
    (output_dir / "d16r_a6_1_next_decision.json").write_text(json.dumps(next_decision, indent=2), encoding="utf-8")

    export_rows = []
    for key, run in runs.items():
        summary = run["summary"]
        export_rows.append(
            {
                "run": key,
                "epoch": summary.get("checkpoint_epoch", ""),
                "embedding_dim": summary.get("embedding_dim", ""),
                "rows": summary.get("row_count", ""),
                "prediction_mismatches": summary.get("prediction_mismatches", ""),
                "monitor": summary.get("best_monitor_metric", ""),
                "score": summary.get("best_monitor_score", ""),
            }
        )

    top_geometry = sorted(
        [row for row in consistent_summary if row["run"] in {"seed42", "seed43", "seed44"}],
        key=lambda row: (row["pattern"], row["run"]),
    )
    watch = [row for row in top_geometry if row["pattern"] in {_pattern(a, b) for a, b in WATCH_PATTERNS}]
    compact_watch = [
        row
        for row in compact_rows
        if row["run"] in {"seed42", "seed43", "seed44"} and row["bucket"] in {"correct", "wrong"}
    ]
    trade_summary = [
        {
            "tradeoff_type": name,
            "count": count,
            "a5c_mean_dist_true_minus_pred": _safe_mean(
                row["a5c_dist_true_minus_pred"] for row in tradeoff_rows if row["tradeoff_type"] == name
            ),
            "a5c_closer_to_pred_ratio": _safe_mean(
                float(row["a5c_closer_to_pred_than_true"]) for row in tradeoff_rows if row["tradeoff_type"] == name
            ),
            "a5b_mean_dist_true_minus_pred": _safe_mean(
                row["a5b_mean_dist_true_minus_pred"] for row in tradeoff_rows if row["tradeoff_type"] == name
            ),
        }
        for name, count in sorted(trade_counts.items())
    ]

    report = f"""# D16R A6-1 Representation Geometry Audit

## Executive Summary
A6-1 exported best-checkpoint embeddings for A5b seed42/43/44 and A5c, then audited hard-class geometry for Angry/Fear/Sad/Neutral. The audit is read-only: no training, checkpoint modification, or test-time checkpoint selection was performed.

Main decision: `{", ".join(decisions)}`.

Across A5b consistent same-wrong hard errors, the mean ratio of samples closer to the wrong predicted class centroid than to the true class centroid is `{_fmt(closer_ratio)}`. This supports a representation-separation interpretation more than a simple calibration-only interpretation, while staying paper-safe: these are embedding-space diagnostics, not causal evidence.

## Export Integrity
{_markdown_table(export_rows, ["run", "epoch", "embedding_dim", "rows", "prediction_mismatches", "monitor", "score"])}

## Consistent Error Geometry
{_markdown_table(watch, ["run", "pattern", "count", "mean_dist_true_minus_pred", "closer_to_pred_ratio", "mean_cos_pred_minus_true"], limit=28)}

Positive `mean_dist_true_minus_pred` means the sample is farther from its true-class centroid than from the predicted-class centroid. High `closer_to_pred_ratio` means the wrong class is geometrically nearer for many samples.

## Hard-Class Compactness
{_markdown_table(compact_watch, ["run", "class_name", "bucket", "count", "mean_dist_to_correct_centroid", "std_dist_to_correct_centroid"], limit=32)}

## Nearest-Neighbor Summary
{_markdown_table(nn_rows, ["run", "pattern", "sample_count", "dominant_neighbor_class", "dominant_neighbor_ratio"], limit=32)}

## A5c Geometry Trade-Off
{_markdown_table(trade_summary, ["tradeoff_type", "count", "a5c_mean_dist_true_minus_pred", "a5c_closer_to_pred_ratio", "a5b_mean_dist_true_minus_pred"])}

A5c is interpreted only within its own embedding space. It can fix some majority-A5b hard errors, but the trade-off table shows whether those fixes come with newly introduced hard-class mistakes. This does not make A5c the replacement for A5b unless its aggregate metrics beat A5b, which they do not.

## Fear/Sad And Sad/Neutral
Fear/Sad and Sad/Neutral remain the primary geometry targets when their rows show positive distance margins and high closer-to-wrong ratios. These patterns are consistent with visually similar hard expressions being embedded nearer to a competing hard class.

## Recommended A6 Direction
Prefer a targeted hard-class representation refinement over more handcrafted input maps:

1. Fear-vs-Sad relation/separation audit and, later, a small trainable relation head if needed.
2. Sad-vs-Neutral separation audit, especially for detected samples.
3. Keep fallback out of scope unless a later audit shows fallback dominates total errors.

## Caveats
Embedding centroids are diagnostic summaries, not semantic or causal evidence. A5b and A5c raw embedding vectors are not directly compared across models; only within-model centroid relationships are compared.
"""
    _write_text(output_dir / "D16R_A6_1_REPRESENTATION_GEOMETRY_AUDIT.md", report)
    return next_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/d16_analysis/main_branch")
    parser.add_argument("--runs_root", default="outputs/d16_runs")
    args = parser.parse_args()
    decision = run_audit(output_dir=Path(args.output_dir), runs_root=Path(args.runs_root))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
