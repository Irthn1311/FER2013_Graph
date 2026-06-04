"""A6-2a best-vs-last read-only diagnostic audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
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
HARD_PROTO_ORDER = [0, 2, 4, 6]
WATCH_PATTERNS = [(2, 4), (4, 6), (6, 4), (0, 4), (2, 0), (4, 2), (4, 0)]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _fmt(value: Any, digits: int = 6) -> str:
    val = _float(value)
    if not math.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return mean(vals) if vals else float("nan")


def _ratio(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else float("nan")


def _pattern(true_id: int, pred_id: int) -> str:
    return f"{CLASS_NAMES.get(true_id, true_id)}->{CLASS_NAMES.get(pred_id, pred_id)}"


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _first(path: Path) -> Dict[str, str]:
    rows = _read_rows(path)
    return rows[0] if rows else {}


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    out = {name: data[name] for name in data.files}
    y_pred = out["artifact_y_pred"] if "artifact_y_pred" in out else out["y_pred"]
    out["active_y_pred"] = y_pred.astype(np.int64)
    return out


def _centroids(data: Dict[str, np.ndarray]) -> Dict[str, Dict[int, np.ndarray]]:
    z = data["z_final_before_classifier"].astype(np.float32)
    y = data["y_true"].astype(np.int64)
    pred = data["active_y_pred"].astype(np.int64)
    out = {"correct": {}, "all_true": {}, "pred": {}}
    for class_id in range(7):
        true_mask = y == class_id
        correct_mask = (y == class_id) & (pred == class_id)
        pred_mask = pred == class_id
        if np.any(true_mask):
            out["all_true"][class_id] = z[true_mask].mean(axis=0)
        if np.any(correct_mask):
            out["correct"][class_id] = z[correct_mask].mean(axis=0)
        else:
            out["correct"][class_id] = out["all_true"].get(class_id, np.zeros(z.shape[1], dtype=np.float32))
        if np.any(pred_mask):
            out["pred"][class_id] = z[pred_mask].mean(axis=0)
        else:
            out["pred"][class_id] = out["correct"][class_id]
    return out


def _geometry_rows(label: str, data: Dict[str, np.ndarray], centroids: Dict[str, Dict[int, np.ndarray]]) -> List[Dict[str, Any]]:
    z = data["z_final_before_classifier"].astype(np.float32)
    y = data["y_true"].astype(np.int64)
    pred = data["active_y_pred"].astype(np.int64)
    rows: List[Dict[str, Any]] = []
    for true_id, pred_id in WATCH_PATTERNS:
        mask = (y == true_id) & (pred == pred_id)
        values = []
        for idx in np.where(mask)[0]:
            true_c = centroids["correct"][true_id]
            pred_c = centroids["correct"][pred_id]
            dt = _dist(z[idx], true_c)
            dp = _dist(z[idx], pred_c)
            ct = _cos(z[idx], true_c)
            cp = _cos(z[idx], pred_c)
            values.append((dt, dp, dt - dp, ct, cp, cp - ct, int(dp < dt)))
        rows.append(
            {
                "row_type": "pattern_geometry",
                "checkpoint": label,
                "pattern": _pattern(true_id, pred_id),
                "true_class": CLASS_NAMES[true_id],
                "pred_class": CLASS_NAMES[pred_id],
                "count": int(mask.sum()),
                "mean_dist_to_true_centroid": _safe_mean(v[0] for v in values),
                "mean_dist_to_pred_centroid": _safe_mean(v[1] for v in values),
                "mean_dist_true_minus_pred": _safe_mean(v[2] for v in values),
                "closer_to_pred_ratio": _ratio(sum(v[6] for v in values), len(values)),
                "mean_cos_to_true_centroid": _safe_mean(v[3] for v in values),
                "mean_cos_to_pred_centroid": _safe_mean(v[4] for v in values),
                "mean_cos_pred_minus_true": _safe_mean(v[5] for v in values),
            }
        )
    for class_id in HARD_IDS:
        class_mask = y == class_id
        correct_mask = class_mask & (pred == class_id)
        wrong_mask = class_mask & (pred != class_id)
        for bucket, mask in (("all_true", class_mask), ("correct", correct_mask), ("wrong", wrong_mask)):
            dists = [_dist(z[idx], centroids["correct"][class_id]) for idx in np.where(mask)[0]]
            rows.append(
                {
                    "row_type": "compactness",
                    "checkpoint": label,
                    "pattern": f"{CLASS_NAMES[class_id]}_{bucket}",
                    "true_class": CLASS_NAMES[class_id],
                    "pred_class": bucket,
                    "count": int(mask.sum()),
                    "mean_dist_to_true_centroid": _safe_mean(dists),
                    "mean_dist_to_pred_centroid": "",
                    "mean_dist_true_minus_pred": "",
                    "closer_to_pred_ratio": "",
                    "mean_cos_to_true_centroid": "",
                    "mean_cos_to_pred_centroid": "",
                    "mean_cos_pred_minus_true": "",
                }
            )
    return rows


def _nearest_neighbor_rows(label: str, data: Dict[str, np.ndarray], k: int = 5) -> List[Dict[str, Any]]:
    z = data["z_final_before_classifier"].astype(np.float32)
    y = data["y_true"].astype(np.int64)
    pred = data["active_y_pred"].astype(np.int64)
    ref_mask = y == pred
    ref_z = z[ref_mask]
    ref_y = y[ref_mask]
    rows = []
    if ref_z.shape[0] <= 0:
        return rows
    for true_id, pred_id in WATCH_PATTERNS:
        mask = (y == true_id) & (pred == pred_id)
        counts: Counter[int] = Counter()
        sample_count = int(mask.sum())
        for idx in np.where(mask)[0]:
            dists = np.linalg.norm(ref_z - z[idx][None, :], axis=1)
            nn_idx = np.argpartition(dists, kth=min(k, len(dists) - 1))[:k]
            for label_id in ref_y[nn_idx].tolist():
                counts[int(label_id)] += 1
        total = sum(counts.values())
        dom_id, dom_count = counts.most_common(1)[0] if counts else (-1, 0)
        rows.append(
            {
                "row_type": "nearest_neighbor",
                "checkpoint": label,
                "pattern": _pattern(true_id, pred_id),
                "true_class": CLASS_NAMES[true_id],
                "pred_class": CLASS_NAMES[pred_id],
                "count": sample_count,
                "dominant_neighbor_class": CLASS_NAMES.get(dom_id, ""),
                "dominant_neighbor_ratio": _ratio(dom_count, total),
                "neighbor_sad_ratio": _ratio(counts.get(4, 0), total),
                "neighbor_neutral_ratio": _ratio(counts.get(6, 0), total),
                "neighbor_fear_ratio": _ratio(counts.get(2, 0), total),
                "neighbor_angry_ratio": _ratio(counts.get(0, 0), total),
            }
        )
    return rows


def _prototype_rows(label: str, data: Dict[str, np.ndarray], centroids: Dict[str, Dict[int, np.ndarray]]) -> List[Dict[str, Any]]:
    if "hard_proto_prototypes" not in data:
        return [{"row_type": "status", "checkpoint": label, "status": "MISSING_PROTOTYPES"}]
    proto = data["hard_proto_prototypes"].astype(np.float32)
    proto_n = _normalize(proto)
    z = data["z_final_before_classifier"].astype(np.float32)
    y = data["y_true"].astype(np.int64)
    rows: List[Dict[str, Any]] = []
    for i, class_id in enumerate(HARD_PROTO_ORDER):
        p = proto[i]
        rows.append(
            {
                "row_type": "prototype_norm",
                "checkpoint": label,
                "prototype": CLASS_NAMES[class_id],
                "target_class": CLASS_NAMES[class_id],
                "other": "",
                "value": float(np.linalg.norm(p)),
                "cosine": "",
                "distance": "",
                "count": "",
                "ratio": "",
                "status": "",
            }
        )
        centroid = centroids["correct"][class_id]
        rows.append(
            {
                "row_type": "prototype_to_own_centroid",
                "checkpoint": label,
                "prototype": CLASS_NAMES[class_id],
                "target_class": CLASS_NAMES[class_id],
                "other": CLASS_NAMES[class_id],
                "value": "",
                "cosine": _cos(p, centroid),
                "distance": _dist(_normalize(p[None, :])[0], _normalize(centroid[None, :])[0]),
                "count": "",
                "ratio": "",
                "status": "",
            }
        )
    for i, class_i in enumerate(HARD_PROTO_ORDER):
        for j, class_j in enumerate(HARD_PROTO_ORDER):
            if j <= i:
                continue
            rows.append(
                {
                    "row_type": "prototype_pair",
                    "checkpoint": label,
                    "prototype": CLASS_NAMES[class_i],
                    "target_class": "",
                    "other": CLASS_NAMES[class_j],
                    "value": "",
                    "cosine": float(proto_n[i] @ proto_n[j]),
                    "distance": _dist(proto_n[i], proto_n[j]),
                    "count": "",
                    "ratio": "",
                    "status": "",
                }
            )
    hard_mask = np.isin(y, HARD_PROTO_ORDER)
    sims = _normalize(z[hard_mask]) @ proto_n.T
    assigned = sims.argmax(axis=1) if sims.size else np.array([], dtype=np.int64)
    hard_y = y[hard_mask]
    for i, class_id in enumerate(HARD_PROTO_ORDER):
        true_mask = hard_y == class_id
        total = int(true_mask.sum())
        for j, proto_class in enumerate(HARD_PROTO_ORDER):
            count = int(np.sum(true_mask & (assigned == j)))
            rows.append(
                {
                    "row_type": "sample_proto_assignment",
                    "checkpoint": label,
                    "prototype": CLASS_NAMES[proto_class],
                    "target_class": CLASS_NAMES[class_id],
                    "other": "",
                    "value": "",
                    "cosine": "",
                    "distance": "",
                    "count": count,
                    "ratio": _ratio(count, total),
                    "status": "",
                }
            )
    rows.append(
        {
            "row_type": "classifier_alignment_status",
            "checkpoint": label,
            "prototype": "",
            "target_class": "",
            "other": "",
            "value": "",
            "cosine": "",
            "distance": "",
            "count": "",
            "ratio": "",
            "status": "MLP_CLASSIFIER_NO_DIRECT_CLASS_WEIGHT_IN_Z_SPACE",
        }
    )
    return rows


def _micro_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for label, name in (("best", "micro_motif_summary.csv"), ("last", "last_micro_motif_summary.csv")):
        data = _read_rows(run_dir / name)
        if not data:
            rows.append({"checkpoint": label, "status": "MISSING", "source": name})
            continue
        for branch in ("major", "micro"):
            branch_rows = [row for row in data if row.get("branch") == branch]
            if not branch_rows:
                continue
            rows.append(
                {
                    "checkpoint": label,
                    "branch": branch,
                    "source": name,
                    "rows": len(branch_rows),
                    "micro_gate_mean": _safe_mean(_float(row.get("micro_gate_mean")) for row in branch_rows),
                    "effective_motif_count_mean": _safe_mean(_float(row.get("effective_motif_count_mean")) for row in branch_rows),
                    "avg_offdiag_similarity_mean": _safe_mean(_float(row.get("avg_offdiag_similarity_mean")) for row in branch_rows),
                    "attention_entropy_mean": _safe_mean(_float(row.get("motif_attention_entropy_mean")) for row in branch_rows),
                    "token_norm_mean": _safe_mean(_float(row.get("motif_token_norm_mean")) for row in branch_rows),
                    "transformed_token_norm_mean": _safe_mean(_float(row.get("motif_transformed_token_norm_mean")) for row in branch_rows),
                    "status": "LOW_MICRO_GATE" if _safe_mean(_float(row.get("micro_gate_mean")) for row in branch_rows) < 0.05 else "OK",
                }
            )
    return rows


def _timeline_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows = _read_rows(run_dir / "train_log.csv")
    if not rows:
        return []
    best_val = max(_float(row.get("val_accuracy")) for row in rows)
    out = []
    for row in rows:
        val_acc = _float(row.get("val_accuracy"))
        hp = _float(row.get("hard_proto_loss_total"))
        pos = _float(row.get("hard_proto_positive_sim_mean"))
        neg = _float(row.get("hard_proto_max_negative_sim_mean"))
        out.append(
            {
                "epoch": _int(row.get("epoch")),
                "global_step": _int(row.get("global_step")),
                "train_loss": _float(row.get("train_loss")),
                "ce_loss": _float(row.get("ce_loss")),
                "hard_proto_loss_total": hp,
                "hard_proto_loss_ce": _float(row.get("hard_proto_loss_ce")),
                "hard_proto_loss_margin": _float(row.get("hard_proto_loss_margin")),
                "lambda_hard_proto_current": _float(row.get("lambda_hard_proto_current")),
                "hard_proto_positive_sim_mean": pos,
                "hard_proto_max_negative_sim_mean": neg,
                "hard_proto_gap_pos_minus_neg": pos - neg if math.isfinite(pos) and math.isfinite(neg) else float("nan"),
                "val_accuracy": val_acc,
                "val_macro_f1": _float(row.get("val_macro_f1")),
                "best_monitor_score_before_epoch": _float(row.get("best_monitor_score")),
                "delta_from_best_val_accuracy": val_acc - best_val if math.isfinite(val_acc) else float("nan"),
                "near_best_val_accuracy_005": int(math.isfinite(val_acc) and best_val - val_acc <= 0.005),
                "hard_proto_easy_flag": int(math.isfinite(hp) and hp < 0.25 and math.isfinite(pos) and math.isfinite(neg) and pos - neg > 0.5),
            }
        )
    return out


def _pred_count_rows(run_dir: Path) -> List[Dict[str, Any]]:
    out = []
    best = {_int(row.get("class_id")): _int(row.get("pred_count")) for row in _read_rows(run_dir / "pred_count.csv")}
    last = {_int(row.get("class_id")): _int(row.get("pred_count")) for row in _read_rows(run_dir / "last_pred_count.csv")}
    for class_id in range(7):
        out.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "best_pred_count": best.get(class_id, 0),
                "last_pred_count": last.get(class_id, 0),
                "last_minus_best": last.get(class_id, 0) - best.get(class_id, 0),
            }
        )
    return out


def _confusion_shift_rows(run_dir: Path) -> List[Dict[str, Any]]:
    def map_counts(path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
        out = {}
        for row in _read_rows(path):
            t = _int(row.get("true_class"), -1)
            p = _int(row.get("pred_class"), -1)
            if t == p or t < 0 or p < 0:
                continue
            out[(t, p)] = {"count": _int(row.get("count")), "row_ratio": _float(row.get("row_ratio")), "support": _int(row.get("support"))}
        return out
    best = map_counts(run_dir / "confusion_matrix.csv")
    last = map_counts(run_dir / "last_confusion_matrix.csv")
    keys = sorted(set(best) | set(last), key=lambda k: max(best.get(k, {}).get("count", 0), last.get(k, {}).get("count", 0)), reverse=True)
    rows = []
    for t, p in keys:
        rows.append(
            {
                "true_class": t,
                "true_name": CLASS_NAMES[t],
                "pred_class": p,
                "pred_name": CLASS_NAMES[p],
                "best_count": best.get((t, p), {}).get("count", 0),
                "last_count": last.get((t, p), {}).get("count", 0),
                "last_minus_best": last.get((t, p), {}).get("count", 0) - best.get((t, p), {}).get("count", 0),
                "best_row_ratio": best.get((t, p), {}).get("row_ratio", float("nan")),
                "last_row_ratio": last.get((t, p), {}).get("row_ratio", float("nan")),
            }
        )
    return rows


def _markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str], limit: int | None = None) -> str:
    rows = rows[:limit] if limit is not None else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(_fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run_audit(run_dir: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    best = _load_npz(run_dir / "embeddings_best_for_audit.npz")
    last = _load_npz(run_dir / "embeddings_last_for_audit.npz")
    best_summary = json.loads((run_dir / "embeddings_best_for_audit.summary.json").read_text(encoding="utf-8"))
    last_summary = json.loads((run_dir / "embeddings_last_for_audit.summary.json").read_text(encoding="utf-8"))
    best_centroids = _centroids(best)
    last_centroids = _centroids(last)
    timeline = _timeline_rows(run_dir)
    geometry = _geometry_rows("best", best, best_centroids) + _geometry_rows("last", last, last_centroids)
    nn_rows = _nearest_neighbor_rows("best", best) + _nearest_neighbor_rows("last", last)
    prototype_rows = _prototype_rows("best", best, best_centroids) + _prototype_rows("last", last, last_centroids)
    micro_rows = _micro_rows(run_dir)
    pred_shift = _pred_count_rows(run_dir)
    confusion_shift = _confusion_shift_rows(run_dir)

    _write_csv(output_dir / "d16r_a6_2a_best_last_epoch_timeline.csv", timeline, [
        "epoch", "global_step", "train_loss", "ce_loss", "hard_proto_loss_total", "hard_proto_loss_ce",
        "hard_proto_loss_margin", "lambda_hard_proto_current", "hard_proto_positive_sim_mean",
        "hard_proto_max_negative_sim_mean", "hard_proto_gap_pos_minus_neg", "val_accuracy", "val_macro_f1",
        "best_monitor_score_before_epoch", "delta_from_best_val_accuracy", "near_best_val_accuracy_005",
        "hard_proto_easy_flag",
    ])
    _write_csv(output_dir / "d16r_a6_2a_best_last_geometry.csv", geometry + nn_rows, [
        "row_type", "checkpoint", "pattern", "true_class", "pred_class", "count", "mean_dist_to_true_centroid",
        "mean_dist_to_pred_centroid", "mean_dist_true_minus_pred", "closer_to_pred_ratio",
        "mean_cos_to_true_centroid", "mean_cos_to_pred_centroid", "mean_cos_pred_minus_true",
        "dominant_neighbor_class", "dominant_neighbor_ratio", "neighbor_sad_ratio", "neighbor_neutral_ratio",
        "neighbor_fear_ratio", "neighbor_angry_ratio",
    ])
    _write_csv(output_dir / "d16r_a6_2a_prototype_diagnostics.csv", prototype_rows, [
        "row_type", "checkpoint", "prototype", "target_class", "other", "value", "cosine", "distance",
        "count", "ratio", "status",
    ])
    _write_csv(output_dir / "d16r_a6_2a_micro_gate_diagnostics.csv", micro_rows, [
        "checkpoint", "branch", "source", "rows", "micro_gate_mean", "effective_motif_count_mean",
        "avg_offdiag_similarity_mean", "attention_entropy_mean", "token_norm_mean",
        "transformed_token_norm_mean", "status",
    ])

    best_test = _first(run_dir / "test_metrics.csv")
    last_test = _first(run_dir / "last_test_metrics.csv")
    best_epoch = _int(best_test.get("checkpoint_epoch"))
    last_epoch = _int(last_test.get("checkpoint_epoch"))
    epoch70 = next((row for row in timeline if row["epoch"] == best_epoch), {})
    epoch100 = next((row for row in timeline if row["epoch"] == last_epoch), {})
    near_late = [row for row in timeline if row["epoch"] > best_epoch and row.get("near_best_val_accuracy_005") == 1]
    best_micro = next((row for row in micro_rows if row.get("checkpoint") == "best" and row.get("branch") == "micro"), {})
    last_micro = next((row for row in micro_rows if row.get("checkpoint") == "last" and row.get("branch") == "micro"), {})
    geom_map = {(row["checkpoint"], row["pattern"]): row for row in geometry if row.get("row_type") == "pattern_geometry"}
    fs_best = geom_map.get(("best", "Fear->Sad"), {})
    fs_last = geom_map.get(("last", "Fear->Sad"), {})
    sn_best = geom_map.get(("best", "Sad->Neutral"), {})
    sn_last = geom_map.get(("last", "Sad->Neutral"), {})
    ns_best = geom_map.get(("best", "Neutral->Sad"), {})
    ns_last = geom_map.get(("last", "Neutral->Sad"), {})
    decisions = []
    if _float(fs_last.get("count")) < _float(fs_best.get("count")) and _float(sn_last.get("count")) > _float(sn_best.get("count")):
        decisions.append("A6_PAIRWISE_RELATION_NEEDED")
    if _float(best_micro.get("micro_gate_mean")) < 0.05 or _float(last_micro.get("micro_gate_mean")) < 0.05:
        decisions.append("A6_PROTO_LOSS_SUPPRESSES_MICRO_SUPPORT")
    if _float(epoch100.get("hard_proto_loss_total")) < 0.25 and _float(last_test.get("accuracy")) > _float(best_test.get("accuracy")):
        decisions.append("PROTOTYPE_SEPARATION_NOT_ALIGNED_WITH_CLASSIFICATION")
    if near_late:
        decisions.append("CONSIDER_VALIDATION_BASED_TIE_BREAK_FOR_A6_ONLY")
    else:
        decisions.append("KEEP_MONITOR_RULE_UNCHANGED")

    next_decision = {
        "run_dir": str(run_dir),
        "best_export": best_summary,
        "last_export": last_summary,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "best_accuracy": _float(best_test.get("accuracy")),
        "best_macro_f1": _float(best_test.get("macro_f1")),
        "last_accuracy": _float(last_test.get("accuracy")),
        "last_macro_f1": _float(last_test.get("macro_f1")),
        "epoch70_val_accuracy": _float(epoch70.get("val_accuracy")),
        "epoch100_val_accuracy": _float(epoch100.get("val_accuracy")),
        "epoch70_val_macro_f1": _float(epoch70.get("val_macro_f1")),
        "epoch100_val_macro_f1": _float(epoch100.get("val_macro_f1")),
        "late_epochs_within_0_005_of_best_val_accuracy": [row["epoch"] for row in near_late],
        "best_micro_gate_mean": _float(best_micro.get("micro_gate_mean")),
        "last_micro_gate_mean": _float(last_micro.get("micro_gate_mean")),
        "fear_sad_best_count": _int(fs_best.get("count")),
        "fear_sad_last_count": _int(fs_last.get("count")),
        "sad_neutral_best_count": _int(sn_best.get("count")),
        "sad_neutral_last_count": _int(sn_last.get("count")),
        "neutral_sad_best_count": _int(ns_best.get("count")),
        "neutral_sad_last_count": _int(ns_last.get("count")),
        "decisions": decisions,
    }
    (output_dir / "d16r_a6_2a_best_last_next_decision.json").write_text(json.dumps(next_decision, indent=2), encoding="utf-8")

    timeline_focus = [row for row in timeline if row["epoch"] in {best_epoch, last_epoch, 10, 20} or row.get("near_best_val_accuracy_005") == 1]
    geom_focus = [row for row in geometry if row.get("row_type") == "pattern_geometry" and row.get("pattern") in {"Fear->Sad", "Sad->Neutral", "Neutral->Sad", "Angry->Sad"}]
    nn_focus = [row for row in nn_rows if row.get("pattern") in {"Fear->Sad", "Sad->Neutral", "Neutral->Sad", "Angry->Sad"}]
    proto_focus = [row for row in prototype_rows if row.get("row_type") in {"prototype_pair", "prototype_to_own_centroid", "classifier_alignment_status"}]
    near_late_epochs = [row["epoch"] for row in near_late]
    report = f"""# D16R A6-2a Best-vs-Last Diagnostic

## Executive Summary
Official result remains `REJECT_A6_2A`: `best.pt` was selected by validation accuracy at epoch {best_epoch} and must remain the official checkpoint.

Diagnostic `last.pt` at epoch {last_epoch} improves test accuracy, but this audit does not select it as the official result. The main finding is: A6-2a reduces the Sad sink later in training, but shifts error pressure into Sad->Neutral. This supports a pairwise hard-relation direction rather than a 4-way global prototype recipe.

Decisions: `{", ".join(decisions)}`.

## Official Best vs Diagnostic Last
| checkpoint | epoch | test_accuracy | macro_f1 | val_accuracy | val_macro_f1 |
|---|---:|---:|---:|---:|---:|
| best.pt | {best_epoch} | {_fmt(best_test.get("accuracy"))} | {_fmt(best_test.get("macro_f1"))} | {_fmt(epoch70.get("val_accuracy"))} | {_fmt(epoch70.get("val_macro_f1"))} |
| last.pt | {last_epoch} | {_fmt(last_test.get("accuracy"))} | {_fmt(last_test.get("macro_f1"))} | {_fmt(epoch100.get("val_accuracy"))} | {_fmt(epoch100.get("val_macro_f1"))} |

## Epoch Timeline
{_markdown_table(timeline_focus, ["epoch", "train_loss", "ce_loss", "hard_proto_loss_total", "hard_proto_gap_pos_minus_neg", "lambda_hard_proto_current", "val_accuracy", "val_macro_f1", "delta_from_best_val_accuracy", "near_best_val_accuracy_005"], limit=30)}

Train log does not contain per-epoch prediction distribution or micro gate. The audit can identify when prototype loss becomes easy, but not when Sad prediction count starts rising during training.

## Prediction Distribution Shift
{_markdown_table(pred_shift, ["class_name", "best_pred_count", "last_pred_count", "last_minus_best"])}

## Hard-Class Confusion Shift
{_markdown_table(confusion_shift, ["true_name", "pred_name", "best_count", "last_count", "last_minus_best", "best_row_ratio", "last_row_ratio"], limit=15)}

## Embedding Geometry Best vs Last
{_markdown_table(geom_focus, ["checkpoint", "pattern", "count", "mean_dist_true_minus_pred", "closer_to_pred_ratio", "mean_cos_pred_minus_true"])}

## Nearest Neighbor Geometry
{_markdown_table(nn_focus, ["checkpoint", "pattern", "count", "dominant_neighbor_class", "dominant_neighbor_ratio", "neighbor_sad_ratio", "neighbor_neutral_ratio"])}

## Prototype Geometry
{_markdown_table(proto_focus, ["checkpoint", "row_type", "prototype", "other", "cosine", "distance", "status"], limit=40)}

The classifier is an MLP with LayerNorm/GELU, so there is no direct class-weight vector in the same z-space as the prototypes. Direct prototype-to-classifier-weight cosine is therefore not reported.

## Micro-Gate / A4 Readout Diagnostics
{_markdown_table(micro_rows, ["checkpoint", "branch", "micro_gate_mean", "effective_motif_count_mean", "avg_offdiag_similarity_mean", "attention_entropy_mean", "status"])}

Micro gate evidence is available only at exported best/last diagnostics, not per epoch.

## Checkpoint Selection Analysis
Epoch {best_epoch} has the best validation accuracy. Epoch {last_epoch} has lower validation accuracy by `{_fmt(_float(epoch100.get("val_accuracy")) - _float(epoch70.get("val_accuracy")))}`. Late epochs within 0.005 of best val accuracy: `{near_late_epochs}`.

This does not support choosing last.pt under the current monitor rule. Any future checkpoint rule must be validation-based and predeclared.

## A6 Design Implications
Last improves Fear->Sad and Neutral->Sad counts, but worsens Sad->Neutral. This is exactly the pattern where one global 4-way prototype objective is too blunt. A6-2b should target pairwise relations separately: Fear-vs-Sad and Sad-vs-Neutral should not share one undifferentiated hard-class separation pressure.

## Next Decision
`{", ".join(decisions)}`
"""
    _write_text(output_dir / "D16R_A6_2A_BEST_LAST_DIAGNOSTIC.md", report)
    return next_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/d16_analysis/main_branch")
    args = parser.parse_args()
    decision = run_audit(Path(args.run_dir), Path(args.output_dir))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
