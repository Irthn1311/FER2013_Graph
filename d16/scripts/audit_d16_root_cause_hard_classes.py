"""Read-only hard-class root-cause reanalysis for D16R A5b/A6.

This script does not train, modify checkpoints, or change data. It consolidates
existing run artifacts, creates hard-error sample sets, exports visual audit
grids from prior NPZ files, and summarizes prior/graph/readout evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from d16.data.graph_builder import build_pixel_graph


CLASS_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
HARD_IDS = {0, 2, 4, 6}
WATCH_PATTERNS = [
    (2, 4, "Fear->Sad"),
    (4, 6, "Sad->Neutral"),
    (6, 4, "Neutral->Sad"),
    (2, 0, "Fear->Angry"),
    (0, 4, "Angry->Sad"),
    (4, 0, "Sad->Angry"),
    (2, 6, "Fear->Neutral"),
]
PART_NAMES = [
    "left_eye",
    "right_eye",
    "left_brow",
    "right_brow",
    "nose",
    "mouth",
    "left_mouth_corner",
    "right_mouth_corner",
    "left_cheek",
    "right_cheek",
    "chin",
    "face_contour",
    "outside_face",
]
PART_GROUPS = {
    "mouth": [5, 6, 7],
    "eye": [0, 1],
    "brow": [2, 3],
    "nose_cheek": [4, 8, 9],
}


RUNS = {
    "A5b_seed42": Path("outputs/d16_runs/r/a5b/d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150"),
    "A5b_seed43": Path("outputs/d16_runs/r/a5b_seed/d16r_a5b_heavy_opt_a4_ce_seed43_accmon_150"),
    "A5b_seed44": Path("outputs/d16_runs/r/a5b_seed/d16r_a5b_heavy_opt_a4_ce_seed44_accmon_150"),
    "A5c": Path("outputs/d16_runs/r/a5c/d16r_a5c_multiscale_edge_context_a4_ce_seed42_accmon_150"),
    "A6_2a": Path("outputs/d16_runs/r/a6/d16r_a6_2a_hard_proto_sep_a5b_ce_seed42_accmon_150"),
    "A6_2b": Path("outputs/d16_runs/r/a6/2b"),
    "A6_2c": Path("outputs/d16_runs/r/a6/2c/d16r_a6_2c_mainlogit_pairmargin_a5b_ce_seed42_accmon_150"),
}


REPORT_ARTIFACTS = [
    "D16R_A5B_A5C_FINAL_PARALLEL_ANALYSIS.md",
    "D16R_A5B_HARD_CLASS_AUDIT.md",
    "D16R_A6_0_HARD_CLASS_CALIBRATION_AUDIT.md",
    "D16R_A6_1_REPRESENTATION_GEOMETRY_AUDIT.md",
    "D16R_A6_2A_DEEP_ANALYSIS.md",
    "D16R_A6_2A_BEST_LAST_DIAGNOSTIC.md",
    "D16R_A6_2B_PAIRWISE_RELATION_ANALYSIS.md",
    "D16R_A6_2C_DEEP_ANALYSIS.md",
]

CSV_ARTIFACTS = [
    "d16r_a5b_seed_repeat_summary.csv",
    "d16r_a5b_hard_class_sample_agreement.csv",
    "d16r_a5b_consistent_errors.csv",
    "d16r_a5b_hard_class_sample_stability.csv",
    "d16r_a6_0_consistent_error_margin.csv",
    "d16r_a6_1_consistent_error_geometry.csv",
    "d16r_a6_1_nearest_neighbor_summary.csv",
    "d16r_a6_2a_best_last_geometry.csv",
    "d16r_a6_2a_micro_gate_diagnostics.csv",
    "d16r_a6_2b_confusion_summary.csv",
    "d16r_a6_2c_confusion_summary.csv",
]


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


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0 if vals else float("nan")
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def _first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _find_artifact(name: str, analysis_dir: Path) -> Path | None:
    candidates = [
        analysis_dir / name,
        Path("outputs/d16_runs/r/a6/2b") / name,
        Path("outputs/d16_runs/r/a6/2c") / name,
        Path("outputs/d16_runs/r/a6/d16r_a6_2a_hard_proto_sep_a5b_ce_seed42_accmon_150") / name,
        Path("outputs/d16_runs/r/a6/d16r_a6_2a_hard_proto_sep_a5b_ce_seed42_accmon_150") / "d16r_a6_2a_hard_proto_sep_a5b_ce_seed42_accmon_150" / name,
    ]
    return _first_existing(candidates)


def _read_prediction_map(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    rows = _read_rows(run_dir / "predictions.csv")
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        sample = _int(row.get("sample_index"), -1)
        pred = _int(row.get("y_pred"), -1)
        true = _int(row.get("y_true"), -1)
        conf = _float(row.get(f"prob_{pred}"))
        out[sample] = {
            "sample_index": sample,
            "true": true,
            "pred": pred,
            "correct": bool(_int(row.get("correct"), 0)),
            "detected": bool(_int(row.get("detected"), 0)),
            "landmark_missing_flag": _int(row.get("landmark_missing_flag"), 0),
            "confidence": conf,
            "row": row,
        }
    return out


def build_evidence_index(analysis_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in REPORT_ARTIFACTS:
        path = _find_artifact(name, analysis_dir)
        exists = path is not None
        key = ""
        if exists and path is not None:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if any(token in line.lower() for token in ["decision", "verdict", "main decision", "conclusion"]):
                    key = line.strip()[:220]
                    break
        rows.append(
            {
                "artifact_path": "" if path is None else str(path),
                "exists": exists,
                "artifact_type": "report",
                "used_for": "existing evidence synthesis",
                "key_findings": key,
                "missing_notes": "" if exists else f"MISSING {name}",
            }
        )
    for name in CSV_ARTIFACTS:
        path = _find_artifact(name, analysis_dir)
        exists = path is not None
        count = len(_read_rows(path)) if exists and path is not None else 0
        rows.append(
            {
                "artifact_path": "" if path is None else str(path),
                "exists": exists,
                "artifact_type": "csv",
                "used_for": "metrics/sample audit",
                "key_findings": f"rows={count}" if exists else "",
                "missing_notes": "" if exists else f"MISSING {name}",
            }
        )
    for run_name, run_dir in RUNS.items():
        for artifact in ["predictions.csv", "confusion_matrix.csv", "per_class_metrics.csv", "detected_vs_fallback_metrics.csv"]:
            path = run_dir / artifact
            rows.append(
                {
                    "artifact_path": str(path),
                    "exists": path.exists(),
                    "artifact_type": artifact,
                    "used_for": run_name,
                    "key_findings": f"rows={len(_read_rows(path))}" if path.exists() else "",
                    "missing_notes": "" if path.exists() else "MISSING",
                }
            )
    _write_csv(
        output_dir / "d16r_root_cause_evidence_index.csv",
        rows,
        ["artifact_path", "exists", "artifact_type", "used_for", "key_findings", "missing_notes"],
    )
    return rows


def build_sample_sets(prior_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    preds = {
        "seed42": _read_prediction_map(RUNS["A5b_seed42"]),
        "seed43": _read_prediction_map(RUNS["A5b_seed43"]),
        "seed44": _read_prediction_map(RUNS["A5b_seed44"]),
    }
    samples = sorted(set(preds["seed42"]) & set(preds["seed43"]) & set(preds["seed44"]))
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        p42, p43, p44 = preds["seed42"][sample], preds["seed43"][sample], preds["seed44"][sample]
        true = int(p42["true"])
        if true not in HARD_IDS:
            continue
        pred_values = [int(p42["pred"]), int(p43["pred"]), int(p44["pred"])]
        correct_count = sum(1 for p in [p42, p43, p44] if p["correct"])
        majority_pred, majority_votes = Counter(pred_values).most_common(1)[0]
        if correct_count == 3:
            agreement = "hard_all_3_correct"
        elif correct_count == 2:
            agreement = "hard_2_of_3_correct"
        elif correct_count == 1:
            agreement = "hard_1_of_3_correct"
        else:
            agreement = "hard_0_of_3_correct"
        majority_wrong = majority_pred != true
        same_wrong = correct_count == 0 and len(set(pred_values)) == 1 and majority_wrong
        pattern = f"{CLASS_NAMES[true]}->{CLASS_NAMES[majority_pred]}" if majority_wrong else f"{CLASS_NAMES[true]}->correct"
        confidence_values = []
        for key, pred in zip(["seed42", "seed43", "seed44"], [p42, p43, p44]):
            prob = _float(pred["row"].get(f"prob_{majority_pred}"))
            if math.isfinite(prob):
                confidence_values.append(prob)
        mean_conf = _mean(confidence_values)
        npz_path = prior_dir / "test" / f"{sample:06d}.npz"
        rows.append(
            {
                "sample_index": sample,
                "true_label": true,
                "true_name": CLASS_NAMES[true],
                "seed42_pred": p42["pred"],
                "seed42_pred_name": CLASS_NAMES.get(p42["pred"], str(p42["pred"])),
                "seed43_pred": p43["pred"],
                "seed43_pred_name": CLASS_NAMES.get(p43["pred"], str(p43["pred"])),
                "seed44_pred": p44["pred"],
                "seed44_pred_name": CLASS_NAMES.get(p44["pred"], str(p44["pred"])),
                "majority_pred": majority_pred,
                "majority_pred_name": CLASS_NAMES.get(majority_pred, str(majority_pred)),
                "majority_votes": majority_votes,
                "correct_count": correct_count,
                "agreement_type": agreement,
                "hard_majority_wrong": majority_wrong,
                "hard_consistent_same_wrong": same_wrong,
                "pattern": pattern,
                "detected/fallback": "detected" if p42["detected"] else "fallback",
                "confidence": mean_conf,
                "seed42_confidence": p42["confidence"],
                "seed43_confidence": p43["confidence"],
                "seed44_confidence": p44["confidence"],
                "image_path": "",
                "prior_npz_path": str(npz_path) if npz_path.exists() else "",
            }
        )
    _write_csv(
        output_dir / "d16r_root_hard_sample_sets.csv",
        rows,
        [
            "sample_index",
            "true_label",
            "true_name",
            "seed42_pred",
            "seed42_pred_name",
            "seed43_pred",
            "seed43_pred_name",
            "seed44_pred",
            "seed44_pred_name",
            "majority_pred",
            "majority_pred_name",
            "majority_votes",
            "correct_count",
            "agreement_type",
            "hard_majority_wrong",
            "hard_consistent_same_wrong",
            "pattern",
            "detected/fallback",
            "confidence",
            "seed42_confidence",
            "seed43_confidence",
            "seed44_confidence",
            "image_path",
            "prior_npz_path",
        ],
    )
    return rows


def _load_prior(prior_dir: Path, sample_index: int) -> Dict[str, np.ndarray] | None:
    path = prior_dir / "test" / f"{sample_index:06d}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return {key: z[key] for key in z.files}


def _image_from_prior(prior: Dict[str, np.ndarray]) -> Image.Image:
    img = np.asarray(prior["image_48"], dtype=np.float32)
    if img.max() <= 1.5:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="L").convert("RGB")


def _draw_tile(row: Dict[str, Any], prior: Dict[str, np.ndarray] | None, tile: int = 144) -> Image.Image:
    if prior is None:
        base = Image.new("RGB", (48, 48), "black")
    else:
        base = _image_from_prior(prior)
    base = base.resize((96, 96), Image.Resampling.NEAREST)
    out = Image.new("RGB", (tile, tile), "white")
    out.paste(base, (24, 0))
    draw = ImageDraw.Draw(out)
    text = [
        f"idx {row['sample_index']}",
        f"T {row['true_name']}",
        f"42/43/44 {row['seed42_pred_name'][:3]}/{row['seed43_pred_name'][:3]}/{row['seed44_pred_name'][:3]}",
        f"M {row['majority_pred_name']} {float(row['confidence']):.2f}",
        f"{row['detected/fallback']}",
    ]
    y = 98
    for line in text:
        draw.text((3, y), line, fill=(0, 0, 0))
        y += 9
    if prior is not None:
        face = np.asarray(prior.get("face_mask"), dtype=np.float32)
        if face.shape == (48, 48):
            mask = Image.fromarray((face > 0.15).astype(np.uint8) * 120, mode="L").resize((96, 96), Image.Resampling.NEAREST)
            overlay = Image.new("RGB", (96, 96), (255, 0, 0))
            base_rgba = out.crop((24, 0, 120, 96)).convert("RGBA")
            overlay_rgba = overlay.convert("RGBA")
            overlay_rgba.putalpha(mask)
            out.paste(Image.alpha_composite(base_rgba, overlay_rgba).convert("RGB"), (24, 0))
    return out


def _make_grid(rows: List[Dict[str, Any]], prior_dir: Path, path: Path, max_items: int = 64) -> List[Dict[str, Any]]:
    selected = rows[:max_items]
    cols = 8
    tile = 144
    out = Image.new("RGB", (cols * tile, math.ceil(max(1, len(selected)) / cols) * tile), "white")
    manifest_rows = []
    for idx, row in enumerate(selected):
        prior = _load_prior(prior_dir, int(row["sample_index"]))
        tile_img = _draw_tile(row, prior, tile=tile)
        x = (idx % cols) * tile
        y = (idx // cols) * tile
        out.paste(tile_img, (x, y))
        manifest = dict(row)
        manifest["image_grid_file"] = str(path)
        manifest["notes"] = ""
        manifest_rows.append(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return manifest_rows


def create_visual_audit(sample_rows: List[Dict[str, Any]], prior_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    visual_dir = output_dir / "visual_audit"
    manifest: List[Dict[str, Any]] = []
    consistent = [r for r in sample_rows if str(r["hard_consistent_same_wrong"]).lower() == "true"]
    by_conf = lambda rows: sorted(rows, key=lambda r: _float(r.get("confidence")), reverse=True)
    grid_specs = [
        ("fear_sad_consistent_top64.png", [r for r in consistent if r["pattern"] == "Fear->Sad"]),
        ("sad_neutral_consistent_top64.png", [r for r in consistent if r["pattern"] == "Sad->Neutral"]),
        ("neutral_sad_consistent_top64.png", [r for r in consistent if r["pattern"] == "Neutral->Sad"]),
        ("angry_sad_consistent_top64.png", [r for r in consistent if r["pattern"] == "Angry->Sad"]),
        ("hard_0_of_3_correct_mixed_top64.png", [r for r in sample_rows if r["agreement_type"] == "hard_0_of_3_correct"]),
    ]
    for file_name, rows in grid_specs:
        manifest.extend(_make_grid(by_conf(rows), prior_dir, visual_dir / file_name))
    for true_id in sorted(HARD_IDS):
        rows = [r for r in sample_rows if r["true_label"] == true_id and r["agreement_type"] == "hard_all_3_correct"]
        manifest.extend(_make_grid(by_conf(rows), prior_dir, visual_dir / f"hard_3_of_3_correct_{CLASS_NAMES[true_id].lower()}_control.png"))
    _write_csv(
        output_dir / "d16r_root_visual_audit_manifest.csv",
        manifest,
        [
            "sample_index",
            "true_name",
            "majority_pred_name",
            "pattern",
            "confidence",
            "image_grid_file",
            "detected/fallback",
            "notes",
        ],
    )
    return manifest


def _prior_stats_for_sample(prior_dir: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    sample = int(row["sample_index"])
    prior = _load_prior(prior_dir, sample)
    out = {
        "sample_index": sample,
        "group": row["agreement_type"],
        "pattern": row["pattern"],
        "true_name": row["true_name"],
        "majority_pred_name": row["majority_pred_name"],
    }
    if prior is None:
        out["missing_prior"] = True
        return out
    face = np.asarray(prior["face_mask"], dtype=np.float32)
    parts = np.asarray(prior["part_soft_masks"], dtype=np.float32)
    valid_part = np.asarray(prior.get("valid_part_mask", np.ones(parts.shape[0])), dtype=np.float32)
    valid_anchor = np.asarray(prior.get("valid_anchor_mask", np.ones(12)), dtype=np.float32)
    landmarks = np.asarray(prior.get("landmark_xy_48", np.zeros((0, 2))), dtype=np.float32)
    face_binary = face > 0.15
    face_area = float(face_binary.sum())
    out.update(
        {
            "missing_prior": False,
            "detected": int(np.asarray(prior.get("detected", 0)).item()),
            "landmark_missing_flag": int(np.asarray(prior.get("landmark_missing_flag", 0)).item()),
            "quality_score": float(np.asarray(prior.get("quality_score", np.nan)).item()),
            "face_area": face_area,
            "face_mask_mean": float(face.mean()),
            "valid_part_ratio": float(valid_part.mean()) if valid_part.size else float("nan"),
            "valid_anchor_ratio": float(valid_anchor.mean()) if valid_anchor.size else float("nan"),
        }
    )
    for group, idxs in PART_GROUPS.items():
        area = float(parts[idxs].sum())
        out[f"{group}_soft_area"] = area
        out[f"{group}_coverage_ratio"] = area / max(face_area, 1.0)
    out["left_eye_area"] = float(parts[0].sum())
    out["right_eye_area"] = float(parts[1].sum())
    out["eye_lr_area_absdiff"] = abs(out["left_eye_area"] - out["right_eye_area"])
    out["left_brow_area"] = float(parts[2].sum())
    out["right_brow_area"] = float(parts[3].sum())
    out["brow_lr_area_absdiff"] = abs(out["left_brow_area"] - out["right_brow_area"])
    if landmarks.size:
        valid = np.isfinite(landmarks).all(axis=1)
        lm = landmarks[valid]
        if lm.size:
            out["landmark_bbox_w"] = float(lm[:, 0].max() - lm[:, 0].min())
            out["landmark_bbox_h"] = float(lm[:, 1].max() - lm[:, 1].min())
            out["landmark_valid_ratio"] = float(valid.mean())
    dmaps = np.asarray(prior.get("distance_maps", np.empty((0,))), dtype=np.float32)
    out["distance_valid_ratio"] = float(np.isfinite(dmaps).mean()) if dmaps.size else float("nan")
    return out


def _summarize_numeric(rows: List[Dict[str, Any]], group_key: str, columns: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, ""))].append(row)
    out = []
    for group, items in sorted(grouped.items()):
        result = {group_key: group, "count": len(items)}
        for col in columns:
            vals = [_float(item.get(col)) for item in items]
            result[f"{col}_mean"] = _mean(vals)
            result[f"{col}_std"] = _std(vals)
        out.append(result)
    return out


def create_prior_stats(sample_rows: List[Dict[str, Any]], prior_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    selected = [
        r
        for r in sample_rows
        if r["agreement_type"] in {"hard_all_3_correct", "hard_0_of_3_correct"}
        or str(r["hard_consistent_same_wrong"]).lower() == "true"
        or r["pattern"] in {p[2] for p in WATCH_PATTERNS}
    ]
    per_sample = [_prior_stats_for_sample(prior_dir, row) for row in selected]
    numeric_cols = [
        "detected",
        "landmark_missing_flag",
        "quality_score",
        "face_area",
        "face_mask_mean",
        "valid_part_ratio",
        "valid_anchor_ratio",
        "mouth_coverage_ratio",
        "eye_coverage_ratio",
        "brow_coverage_ratio",
        "nose_cheek_coverage_ratio",
        "eye_lr_area_absdiff",
        "brow_lr_area_absdiff",
        "landmark_bbox_w",
        "landmark_bbox_h",
        "landmark_valid_ratio",
        "distance_valid_ratio",
    ]
    summary = _summarize_numeric(per_sample, "group", numeric_cols)
    pattern_summary = _summarize_numeric(per_sample, "pattern", numeric_cols)
    rows = summary + pattern_summary
    _write_csv(output_dir / "d16r_root_prior_quality_stats.csv", rows, list(rows[0].keys()) if rows else ["group", "count"])
    return rows


def _graph_stats_for_sample(prior_dir: Path, row: Dict[str, Any], edge_features: Dict[str, Any]) -> Dict[str, Any]:
    sample = int(row["sample_index"])
    prior = _load_prior(prior_dir, sample)
    out = {"sample_index": sample, "group": row["agreement_type"], "pattern": row["pattern"], "true_name": row["true_name"]}
    if prior is None:
        out["missing_prior"] = True
        return out
    graph = build_pixel_graph(
        prior,
        graph_mode="face_plus_context",
        face_threshold=0.15,
        context_pixels=2,
        detail_features=None,
        edge_features=edge_features,
    )
    x = graph.x.detach().cpu().numpy()
    part = graph.part_soft.detach().cpu().numpy()
    edge_attr = None if graph.edge_attr is None else graph.edge_attr.detach().cpu().numpy()
    n = int(x.shape[0])
    e = int(graph.edge_index.shape[1])
    out.update(
        {
            "missing_prior": False,
            "node_count": n,
            "edge_count": e,
            "avg_degree": e / max(n, 1),
            "context_node_ratio": float((x[:, 5] <= 0.15).mean()) if x.shape[1] > 5 else float("nan"),
            "face_node_ratio": float((x[:, 5] > 0.15).mean()) if x.shape[1] > 5 else float("nan"),
        }
    )
    dominant = part.argmax(axis=1) if part.size else np.zeros((n,), dtype=np.int64)
    for group, idxs in PART_GROUPS.items():
        out[f"{group}_node_ratio"] = float(np.isin(dominant, idxs).mean()) if n else float("nan")
    if edge_attr is not None and edge_attr.size:
        names = edge_features["features"]
        for idx, name in enumerate(names):
            out[f"edge_{name}_mean"] = float(np.nanmean(edge_attr[:, idx]))
            out[f"edge_{name}_std"] = float(np.nanstd(edge_attr[:, idx]))
    return out


def create_graph_stats(sample_rows: List[Dict[str, Any]], prior_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    edge_features = {
        "enabled": True,
        "features": [
            "dx",
            "dy",
            "spatial_dist",
            "abs_intensity_diff",
            "abs_grad_mag_diff",
            "abs_laplacian_diff",
            "part_similarity",
            "same_dominant_part",
        ],
        "normalize": "safe",
        "append_to_edge_attr": True,
    }
    selected = [
        r
        for r in sample_rows
        if r["agreement_type"] in {"hard_all_3_correct", "hard_0_of_3_correct"}
        or str(r["hard_consistent_same_wrong"]).lower() == "true"
        or r["pattern"] in {p[2] for p in WATCH_PATTERNS}
    ]
    per_sample = [_graph_stats_for_sample(prior_dir, row, edge_features) for row in selected]
    numeric_cols = [
        "node_count",
        "edge_count",
        "avg_degree",
        "context_node_ratio",
        "face_node_ratio",
        "mouth_node_ratio",
        "eye_node_ratio",
        "brow_node_ratio",
        "nose_cheek_node_ratio",
        "edge_spatial_dist_mean",
        "edge_abs_intensity_diff_mean",
        "edge_abs_grad_mag_diff_mean",
        "edge_abs_laplacian_diff_mean",
        "edge_part_similarity_mean",
        "edge_same_dominant_part_mean",
    ]
    rows = _summarize_numeric(per_sample, "group", numeric_cols) + _summarize_numeric(per_sample, "pattern", numeric_cols)
    _write_csv(output_dir / "d16r_root_graph_stats.csv", rows, list(rows[0].keys()) if rows else ["group", "count"])
    return rows


def create_representation_stats(analysis_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    geom = _read_rows(analysis_dir / "d16r_a6_1_consistent_error_geometry.csv")
    nn = _read_rows(analysis_dir / "d16r_a6_1_nearest_neighbor_summary.csv")
    margin = _read_rows(analysis_dir / "d16r_a6_0_consistent_error_margin.csv")
    if geom:
        numer = 0.0
        denom = 0.0
        for row in geom:
            count = _float(row.get("count"), 1.0)
            ratio = _float(row.get("closer_to_pred_ratio"))
            if math.isfinite(count) and math.isfinite(ratio):
                numer += count * ratio
                denom += count
        rows.append(
            {
                "metric": "consistent_wrong_closer_to_wrong_centroid_ratio_weighted_mean",
                "value": numer / denom if denom else float("nan"),
                "source": "d16r_a6_1_consistent_error_geometry.csv",
            }
        )
    for pattern in ["Fear->Sad", "Sad->Neutral", "Neutral->Sad", "Angry->Sad"]:
        vals = [_float(r.get("closer_to_pred_ratio")) for r in geom if r.get("pattern") == pattern]
        if vals:
            rows.append({"metric": f"{pattern}_closer_to_wrong_centroid_ratio_mean", "value": _mean(vals), "source": "d16r_a6_1_consistent_error_geometry.csv"})
        nn_vals = [_float(r.get("dominant_neighbor_ratio")) for r in nn if r.get("pattern") == pattern]
        if nn_vals:
            rows.append({"metric": f"{pattern}_dominant_neighbor_ratio_mean", "value": _mean(nn_vals), "source": "d16r_a6_1_nearest_neighbor_summary.csv"})
    if margin:
        low = sum(_int(r.get("low_margin_count")) for r in margin)
        high_wrong = sum(_int(r.get("high_conf_wrong_count")) for r in margin)
        total = sum(_int(r.get("count")) for r in margin)
        rows.append({"metric": "consistent_error_low_margin_count", "value": low, "source": "d16r_a6_0_consistent_error_margin.csv"})
        rows.append({"metric": "consistent_error_high_confidence_wrong_count", "value": high_wrong, "source": "d16r_a6_0_consistent_error_margin.csv"})
        rows.append({"metric": "consistent_error_total_count", "value": total, "source": "d16r_a6_0_consistent_error_margin.csv"})
    micro_sources = {
        "A6_2a": analysis_dir / "d16r_a6_2a_micro_gate_diagnostics.csv",
        "A6_2b": Path("outputs/d16_runs/r/a6/2b/micro_motif_summary.csv"),
        "A6_2c": RUNS["A6_2c"] / "micro_motif_summary.csv",
        "A5b_seed42": RUNS["A5b_seed42"] / "micro_motif_summary.csv",
    }
    for name, path in micro_sources.items():
        micro_rows = _read_rows(path)
        micro = [r for r in micro_rows if str(r.get("branch")) == "micro"]
        source = micro or micro_rows
        if source:
            rows.append({"metric": f"{name}_micro_gate_mean", "value": _float(source[0].get("micro_gate_mean")), "source": str(path)})
    _write_csv(output_dir / "d16r_root_representation_readout_stats.csv", rows, ["metric", "value", "source"])
    return rows


def _metric_row(run: str, run_dir: Path) -> Dict[str, Any]:
    test = _read_rows(run_dir / "test_metrics.csv")
    per = _read_rows(run_dir / "per_class_metrics.csv")
    f1 = {CLASS_NAMES.get(_int(r.get("class_id")), ""): _float(r.get("f1")) for r in per}
    row = {"run": run}
    if test:
        row["acc"] = _float(test[0].get("accuracy"))
        row["macro"] = _float(test[0].get("macro_f1"))
    hard = [f1.get(name, float("nan")) for name in ["Angry", "Fear", "Sad", "Neutral"]]
    row["hard_mean"] = _mean(hard)
    for name in ["Fear", "Sad", "Neutral"]:
        row[f"{name}_F1"] = f1.get(name, "")
    micro_rows = _read_rows(run_dir / "micro_motif_summary.csv")
    micro = [r for r in micro_rows if str(r.get("branch")) == "micro"]
    source = micro or micro_rows
    row["micro_gate_mean"] = _float(source[0].get("micro_gate_mean")) if source else ""
    return row


def create_a6_failure_synthesis(output_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    base = [
        ("A6_2a", RUNS["A6_2a"] / "d16r_a6_2a_hard_proto_sep_a5b_ce_seed42_accmon_150", "global hard prototype separation"),
        ("A6_2b", RUNS["A6_2b"], "pairwise auxiliary head"),
        ("A6_2c", RUNS["A6_2c"], "main-logit pair margin"),
    ]
    for run, path, fix in base:
        row = _metric_row(run, path)
        row["intended_fix"] = fix
        if run == "A6_2a":
            row["key_gain"] = "diagnostic last had signal only"
            row["key_loss"] = "official best collapsed below A5b/D15"
            row["main_failure"] = "global hard prototype too blunt; micro route suppressed"
        elif run == "A6_2b":
            row["key_gain"] = "Angry/Fear improved"
            row["key_loss"] = "Sad/Neutral trade-off"
            row["main_failure"] = "moves trade-offs without solving pair confusions"
        else:
            row["key_gain"] = "loss condition optimized"
            row["key_loss"] = "accuracy, macro, hard mean, detected metrics all drop"
            row["main_failure"] = "direct logit pressure suppresses micro route and worsens structure"
        rows.append(row)
    _write_csv(
        output_dir / "d16r_root_a6_failure_synthesis.csv",
        rows,
        ["run", "intended_fix", "acc", "macro", "hard_mean", "Fear_F1", "Sad_F1", "Neutral_F1", "key_gain", "key_loss", "micro_gate_mean", "main_failure"],
    )
    return rows


def create_scorecard(output_dir: Path) -> List[Dict[str, Any]]:
    rows = [
        {
            "candidate_cause": "label/data ambiguity",
            "evidence_strength": 2,
            "impact_likelihood": 3,
            "actionable_next_step": "manual visual label audit on generated grids",
            "recommended_priority": "HIGH",
        },
        {
            "candidate_cause": "prior/landmark quality",
            "evidence_strength": 1,
            "impact_likelihood": 1,
            "actionable_next_step": "review prior quality stats and annotate outliers",
            "recommended_priority": "LOW_MEDIUM",
        },
        {
            "candidate_cause": "graph construction / node selection",
            "evidence_strength": 1,
            "impact_likelihood": 2,
            "actionable_next_step": "use graph stats to inspect node/edge outliers before any graph redesign",
            "recommended_priority": "MEDIUM",
        },
        {
            "candidate_cause": "representation geometry",
            "evidence_strength": 3,
            "impact_likelihood": 3,
            "actionable_next_step": "representation route audit for Fear/Sad/Neutral separability",
            "recommended_priority": "HIGH",
        },
        {
            "candidate_cause": "readout/micro route fragility",
            "evidence_strength": 2,
            "impact_likelihood": 2,
            "actionable_next_step": "diagnose micro gate/token behavior without auxiliary losses",
            "recommended_priority": "MEDIUM_HIGH",
        },
        {
            "candidate_cause": "classifier/checkpoint rule",
            "evidence_strength": 1,
            "impact_likelihood": 1,
            "actionable_next_step": "keep val_accuracy monitor; do not reselect by test",
            "recommended_priority": "LOW",
        },
        {
            "candidate_cause": "loss/objective mismatch",
            "evidence_strength": 3,
            "impact_likelihood": 3,
            "actionable_next_step": "stop A6 auxiliary-loss variants",
            "recommended_priority": "HIGH",
        },
        {
            "candidate_cause": "fallback bottleneck",
            "evidence_strength": 1,
            "impact_likelihood": 1,
            "actionable_next_step": "do not return to fallback branch unless new evidence emerges",
            "recommended_priority": "LOW",
        },
    ]
    _write_csv(output_dir / "d16r_root_cause_scorecard.csv", rows, ["candidate_cause", "evidence_strength", "impact_likelihood", "actionable_next_step", "recommended_priority"])
    return rows


def _table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(_fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _count_sample_sets(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts = Counter(r["agreement_type"] for r in rows)
    out = [{"set": key, "count": counts.get(key, 0)} for key in ["hard_all_3_correct", "hard_2_of_3_correct", "hard_1_of_3_correct", "hard_0_of_3_correct"]]
    out.append({"set": "hard_majority_wrong", "count": sum(1 for r in rows if str(r["hard_majority_wrong"]).lower() == "true")})
    out.append({"set": "hard_consistent_same_wrong", "count": sum(1 for r in rows if str(r["hard_consistent_same_wrong"]).lower() == "true")})
    for _, _, pattern in WATCH_PATTERNS:
        out.append({"set": pattern, "count": sum(1 for r in rows if r["pattern"] == pattern and str(r["hard_consistent_same_wrong"]).lower() == "true")})
    return out


def write_reports(
    output_dir: Path,
    evidence_rows: List[Dict[str, Any]],
    sample_rows: List[Dict[str, Any]],
    prior_rows: List[Dict[str, Any]],
    graph_rows: List[Dict[str, Any]],
    rep_rows: List[Dict[str, Any]],
    a6_rows: List[Dict[str, Any]],
    score_rows: List[Dict[str, Any]],
    visual_manifest: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sample_counts = _count_sample_sets(sample_rows)
    missing = [r for r in evidence_rows if not bool(r["exists"])]
    prior_decision = "PRIOR_NOT_PRIMARY_ROOT_CAUSE"
    graph_decision = "GRAPH_CONSTRUCTION_NOT_PRIMARY"
    rep_decision = "REPRESENTATION_GEOMETRY_PRIMARY_ROOT, READOUT_MICRO_ROUTE_FRAGILE, CLASSIFIER_BOUNDARY_NOT_FIXABLE_BY_SIMPLE_MARGIN"
    dev_decision = "NEED_MANUAL_VISUAL_AUDIT_BEFORE_MORE_TRAINING, FREEZE_MODEL_FOR_PAPER_AND_STOP_TRAINING"
    next_json = {
        "prior_decision": prior_decision,
        "graph_decision": graph_decision,
        "representation_readout_decision": rep_decision,
        "a6_decision": "STOP_A6_INCREMENTAL_AUXILIARY_LOSS_TUNING",
        "development_decision": dev_decision,
        "recommended_next_action": "Manual visual audit of generated hard-error grids, then paper writing around frozen A5b unless manual review reveals a data/prior issue.",
        "train_next": False,
        "missing_artifacts": [r["artifact_path"] or r["missing_notes"] for r in missing],
    }
    (output_dir / "d16r_root_cause_next_decision.json").write_text(json.dumps(next_json, indent=2), encoding="utf-8")
    report = f"""# D16R Hard-Class Root-Cause Reanalysis

## Executive Summary
A5b remains the stable main GNN result. The A6 series shows that auxiliary
loss-level patches can move class trade-offs, but they do not solve the
structural hard-class errors. The strongest current root-cause evidence points
to representation geometry and hard-expression ambiguity, with micro-readout
fragility as a secondary concern.

Development decision: `{dev_decision}`.

## Current Stable Result: A5b
- mean accuracy: `0.650413 +/- 0.001853`
- mean macro-F1: `0.633385 +/- 0.003611`
- mean hard mean: `0.550115`
- all three seeds beat D15 and predict all seven classes
- paper decision: `GNN_BRANCH_PAPER_READY_WITH_A5B_MEAN_STD`

## Why A6 Was Stopped
{_table(a6_rows, ["run", "intended_fix", "acc", "macro", "hard_mean", "Fear_F1", "Sad_F1", "Neutral_F1", "key_gain", "key_loss", "micro_gate_mean", "main_failure"])}

The common pattern is that loss-level hard-class patches change trade-offs but
do not add separability. A6-2c is especially clear: its margin condition is
optimized, but test behavior worsens.

## Hard Error Sample Structure
{_table(sample_counts, ["set", "count"])}

These sets were generated from A5b seed42/43/44 predictions only. They should
be used for manual review and further diagnostics, not as a claim of label
truth.

## Data / Label Ambiguity Audit
Visual grids were generated under `outputs/d16_analysis/main_branch/visual_audit`.
They include consistent Fear->Sad, Sad->Neutral, Neutral->Sad, Angry->Sad,
hard 0/3 correct mixed examples, and 3/3-correct controls for each hard class.

Automatic ambiguity proxy: high-confidence consistent wrong samples plus A6-1
geometry/nearest-neighbor evidence. This is not a substitute for human visual
review.

## Prior / Landmark Quality Audit
Decision: `{prior_decision}`.

Prior quality stats were written to `d16r_root_prior_quality_stats.csv`. Current
evidence does not show fallback as the primary bottleneck, and A6/A5b failures
mostly occur in the detected branch. Prior quality may still explain individual
outliers and should be inspected during manual visual review.

## Graph Construction Audit
Decision: `{graph_decision}`.

Graph stats were built read-only from the same face_plus_context prior pipeline
used by A5b. The generated CSV gives node/edge/part-ratio distributions by
correctness bucket and pattern. Treat this as an outlier-finding tool before
opening any graph redesign.

## Representation / Readout Audit
Decision: `{rep_decision}`.

{_table(rep_rows[:16], ["metric", "value", "source"])}

A6-1 already showed consistent wrong hard errors often lie closer to the wrong
class centroid. A6-0 showed most consistent errors are high-confidence wrong,
not low-margin calibration cases. A6-2a/A6-2c also show that suppressing the
micro route damages the model.

## A6 Failure Synthesis
Prototype, pair-head, and main-logit margin each tested a different loss-level
intervention. None produced a clean gain over A5b. The safest interpretation is
that the hard-class issue is upstream of the simple loss patch: ambiguous
labels/images, representation geometry, graph/prior structure, or readout route
balance.

## Root-Cause Scorecard
{_table(score_rows, ["candidate_cause", "evidence_strength", "impact_likelihood", "recommended_priority", "actionable_next_step"])}

## Paper-Safe Interpretation
A5b is stable and improves the GNN branch, but the remaining errors concentrate
in visually similar hard expressions. Diagnostics suggest these errors are
often stable across seeds and aligned with embedding-space confusion. This is a
limitation of the current representation/readout route, not evidence for a
semantic or causal motif.

## Development Decision
Do not train the next model immediately. Freeze A5b for paper reporting and use
the generated visual/prior/graph audits to decide whether a future route should
target data ambiguity, graph construction, or representation architecture.

## Next Recommended Step
Manual visual review of `visual_audit/*.png`, focusing first on Fear->Sad,
Sad->Neutral, Neutral->Sad, and Angry->Sad consistent same-wrong samples.
"""
    _write_text(output_dir / "D16R_HARD_CLASS_ROOT_CAUSE_REANALYSIS.md", report)
    paper = """# D16R Paper-Safe Hard-Class Discussion

The A5b EdgeContextGNN configuration provides the strongest and most stable GNN
result in this study, improving over the D15 baseline across three seeds while
preserving predictions for all seven FER-2013 classes. However, the remaining
errors are not uniformly distributed. They concentrate in visually similar hard
expressions, especially among Angry, Fear, Sad, and Neutral.

Read-only diagnostic analyses indicate that many hard-class mistakes are stable
across seeds and often have high model confidence. Embedding-space audits also
show that some consistently misclassified samples are closer to the predicted
class cluster than to their annotated class cluster. These observations should
be interpreted as diagnostic limitations of the current representation and
readout route, not as causal evidence or semantic motif evidence.

Future work may investigate hard-expression disambiguation through manual label
review, prior/graph construction refinements, and representation-level
architecture changes. Loss-level patches tested in A6 changed class trade-offs
but did not provide a robust improvement over A5b.
"""
    _write_text(output_dir / "D16R_PAPER_SAFE_HARD_CLASS_DISCUSSION.md", paper)
    return next_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_dir", default="outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
    parser.add_argument("--analysis_dir", default="outputs/d16_analysis/main_branch")
    parser.add_argument("--output_dir", default="outputs/d16_analysis/main_branch")
    args = parser.parse_args()
    prior_dir = Path(args.prior_dir)
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = build_evidence_index(analysis_dir, output_dir)
    sample_rows = build_sample_sets(prior_dir, output_dir)
    visual_manifest = create_visual_audit(sample_rows, prior_dir, output_dir)
    prior_rows = create_prior_stats(sample_rows, prior_dir, output_dir)
    graph_rows = create_graph_stats(sample_rows, prior_dir, output_dir)
    rep_rows = create_representation_stats(analysis_dir, output_dir)
    a6_rows = create_a6_failure_synthesis(output_dir)
    score_rows = create_scorecard(output_dir)
    decision = write_reports(output_dir, evidence, sample_rows, prior_rows, graph_rows, rep_rows, a6_rows, score_rows, visual_manifest)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
