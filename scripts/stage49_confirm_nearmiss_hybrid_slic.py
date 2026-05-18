#!/usr/bin/env python
"""Stage 4.9 direct visual audit prep + confirmatory deletion for a near-miss Hybrid SLIC selector.

Scope is intentionally narrow:
- no Stage 5 opening by default;
- no motif, semantic-part, SupCon, learned-selector, or new-grid expansion;
- only one Stage 4.8 near-miss plus the original Hybrid SLIC baseline and the
  fixed control set are rechecked under the fixed-original protocol.

If no direct vision reviewer is available, this script writes a direct-audit
sheet marked ``NEEDS_MANUAL_REVIEW`` and keeps the visual gate closed.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.graph_repository import GraphRepositoryReader
from data.graph_resolver import GraphResolver
from data.labels import EMOTION_NAMES
from stage1_pixel_region_selection import ensure_dir, fmt, normalize01, resolve_graph_repo, safe_name, save_png, to_u8, try_import_slic
from stage2_retention_deletion_test import class_name, make_original_dataset
from stage36_structure_aware_diagnostics import aggregate_stats as aggregate_structure_stats
from stage36_structure_aware_diagnostics import load_split_records, mask_structure_stats
from stage4_refinement_run import eval_probe, train_probe
from stage4_refinement_v2 import local_mean_apply
from stage46_gemini_visual_audit import SUMMARY_SELECTOR_COLUMNS as STAGE46_SUMMARY_SELECTOR_COLUMNS
from stage48_narrow_hybrid_slic_refinement import (
    BASE_HYBRID_PIXEL,
    BASE_SELECTOR,
    SourceSpec,
    add_control_gaps,
    make_mask,
    maybe_cap,
    parse_optional_int,
    read_csv,
    safe_float,
    write_csv,
)


NEARMISS_SELECTOR = "hybrid_slic_region__E_grad_50__b0p1__max_mean_mix"
RATIOS = (0.15, 0.20)

NEARMISS_SPEC = SourceSpec(
    name=NEARMISS_SELECTOR,
    source_type="refined_variant",
    weight_id="E_grad_50",
    aggregation="max_mean_mix",
)

CONTROL_SPECS: Tuple[SourceSpec, ...] = (
    SourceSpec("random_pixel", "control"),
    SourceSpec("center_prior", "control"),
    SourceSpec("gradient_topk", "control"),
    SourceSpec("delta_edge_topk", "control"),
    SourceSpec("slic_region_proposal", "control"),
    SourceSpec(BASE_SELECTOR, "control_hybrid"),
    SourceSpec(BASE_HYBRID_PIXEL, "control_hybrid"),
)

ALL_SPECS = (NEARMISS_SPEC,) + CONTROL_SPECS

AUDIT_COLUMNS = [
    "audit_id",
    "selector_name",
    "class_name",
    "graph_id",
    "ratio",
    "figure_path",
    "selected_eye_eyebrow",
    "selected_mouth_nasolabial",
    "selected_face_muscle_cheek_wrinkle",
    "selected_hair_glasses",
    "selected_border_background",
    "long_contour_dominant",
    "center_collapse",
    "fragmented_pixel_dust",
    "region_like",
    "facial_evidence_like",
    "overall_visual_pass",
    "notes",
]


def require_inputs(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Stage 4.9 inputs:\n- " + "\n- ".join(missing))


def load_visual_summary(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in read_csv(path):
        name = row.get("selector_name", "")
        if not name:
            continue
        out[name] = {
            "reviewed_count": safe_float(row.get("reviewed_count"), 0.0),
            "visual_partial_pass_rate": safe_float(row.get("partial_or_pass_rate")),
            "avg_facial_evidence_like": safe_float(row.get("avg_facial_evidence_like")),
            "avg_hair_glasses": safe_float(row.get("avg_hair_glasses")),
            "avg_border_background": safe_float(row.get("avg_border_background")),
            "avg_center_collapse": safe_float(row.get("avg_center_collapse")),
        }
    return out


def reviewed_visual_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get("overall_visual_pass", "")).upper() in {"PASS", "PARTIAL", "FAIL"}]


def mean_score(rows: Sequence[Dict[str, Any]], field: str) -> str:
    vals = [safe_float(row.get(field)) for row in rows]
    finite = [float(value) for value in vals if math.isfinite(float(value))]
    return f"{float(np.mean(finite)):.6f}" if finite else ""


def summarize_visual_audit(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["selector_name"])].append(dict(row))
    summary: List[Dict[str, Any]] = []
    for selector_name, group_rows in sorted(grouped.items()):
        reviewed = reviewed_visual_rows(group_rows)
        counts = Counter(str(row.get("overall_visual_pass", "")).upper() for row in group_rows)
        reviewed_count = len(reviewed)
        pass_count = counts.get("PASS", 0)
        partial_count = counts.get("PARTIAL", 0)
        fail_count = counts.get("FAIL", 0)
        summary.append(
            {
                "selector_name": selector_name,
                "source_stage": "stage49_direct",
                "reviewed_count": reviewed_count,
                "pass_count": pass_count,
                "partial_count": partial_count,
                "fail_count": fail_count,
                "unreviewed_count": len(group_rows) - reviewed_count,
                "pass_rate": f"{pass_count / reviewed_count:.6f}" if reviewed_count else "",
                "partial_or_pass_rate": f"{(pass_count + partial_count) / reviewed_count:.6f}" if reviewed_count else "",
                "avg_eye_eyebrow": mean_score(reviewed, "selected_eye_eyebrow"),
                "avg_mouth_nasolabial": mean_score(reviewed, "selected_mouth_nasolabial"),
                "avg_face_muscle_cheek_wrinkle": mean_score(reviewed, "selected_face_muscle_cheek_wrinkle"),
                "avg_hair_glasses": mean_score(reviewed, "selected_hair_glasses"),
                "avg_border_background": mean_score(reviewed, "selected_border_background"),
                "avg_long_contour": mean_score(reviewed, "long_contour_dominant"),
                "avg_center_collapse": mean_score(reviewed, "center_collapse"),
                "avg_fragmentation": mean_score(reviewed, "fragmented_pixel_dust"),
                "avg_region_like": mean_score(reviewed, "region_like"),
                "avg_facial_evidence_like": mean_score(reviewed, "facial_evidence_like"),
                "main_failure_reason": "NEEDS_MANUAL_REVIEW" if not reviewed_count else "",
            }
        )
    return summary


def build_visual_join(
    stage46_summary: Dict[str, Dict[str, float]],
    stage49_summary_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    stage49_lookup = {row["selector_name"]: row for row in stage49_summary_rows}
    out: Dict[str, Dict[str, Any]] = {}
    for spec in ALL_SPECS:
        if spec.name == NEARMISS_SELECTOR:
            summary = stage49_lookup.get(spec.name, {})
            reviewed = int(safe_float(summary.get("reviewed_count"), 0.0))
            direct_ready = reviewed > 0
            out[spec.name] = {
                "visual_join_status": "direct" if direct_ready else "needs_manual_review",
                "visual_metrics_usable_for_gate": int(direct_ready),
                "visual_partial_pass_rate": safe_float(summary.get("partial_or_pass_rate")),
                "avg_facial_evidence_like": safe_float(summary.get("avg_facial_evidence_like")),
                "avg_hair_glasses": safe_float(summary.get("avg_hair_glasses")),
                "avg_border_background": safe_float(summary.get("avg_border_background")),
                "avg_center_collapse": safe_float(summary.get("avg_center_collapse")),
            }
        elif spec.name in stage46_summary:
            values = stage46_summary[spec.name]
            out[spec.name] = {
                "visual_join_status": "direct_stage46",
                "visual_metrics_usable_for_gate": 1,
                **values,
            }
        else:
            out[spec.name] = {
                "visual_join_status": "missing",
                "visual_metrics_usable_for_gate": 0,
                "visual_partial_pass_rate": float("nan"),
                "avg_facial_evidence_like": float("nan"),
                "avg_hair_glasses": float("nan"),
                "avg_border_background": float("nan"),
                "avg_center_collapse": float("nan"),
            }
    return out


def overlay_rgb(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = np.repeat(to_u8(normalize01(original)).reshape(48, 48, 1), 3, axis=2).astype(np.float32)
    overlay = base.copy()
    m = np.asarray(mask, dtype=bool).reshape(48, 48)
    overlay[m, 0] = 255.0
    overlay[m, 1] *= 0.35
    overlay[m, 2] *= 0.35
    return np.clip(overlay, 0, 255).astype(np.uint8)


def grayscale_rgb(values: np.ndarray) -> np.ndarray:
    arr = to_u8(normalize01(values)).reshape(48, 48)
    return np.repeat(arr[:, :, None], 3, axis=2)


def slic_visual(record: Dict[str, Any], slic_fn: Any, args: argparse.Namespace) -> np.ndarray:
    if slic_fn is None:
        return np.zeros((48, 48), dtype=np.float32)
    segments = slic_fn(
        record["intensity"].reshape(48, 48).astype(np.float32),
        n_segments=int(args.slic_segments),
        compactness=float(args.slic_compactness),
        start_label=0,
        channel_axis=None,
    )
    return normalize01(np.asarray(segments, dtype=np.float32))


def render_figure(
    path: Path,
    record: Dict[str, Any],
    mask: np.ndarray,
    slic_fn: Any,
    args: argparse.Namespace,
) -> None:
    from PIL import Image, ImageDraw

    original = normalize01(record["intensity"]).reshape(48, 48)
    only_selected = local_mean_apply(record["intensity"], mask, "only_selected").reshape(48, 48)
    delete_selected = local_mean_apply(record["intensity"], mask, "delete_selected").reshape(48, 48)
    panels: List[Tuple[str, np.ndarray]] = [
        ("original", grayscale_rgb(original)),
        ("mask", np.repeat((np.asarray(mask).reshape(48, 48).astype(np.uint8) * 255)[:, :, None], 3, axis=2)),
        ("overlay", overlay_rgb(record["intensity"], mask)),
        ("only_selected", grayscale_rgb(only_selected)),
        ("delete_selected", grayscale_rgb(delete_selected)),
        ("grad_map", grayscale_rgb(record["grad_mag"])),
        ("delta_map", grayscale_rgb(record["delta_edge_node"])),
        ("slic_regions", grayscale_rgb(slic_visual(record, slic_fn, args))),
    ]
    tile = 48
    scale = 4
    label_h = 18
    cols = 4
    rows = 2
    canvas = Image.new("RGB", (cols * tile * scale, rows * (tile * scale + label_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, image) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = col * tile * scale
        y = row * (tile * scale + label_h)
        panel = Image.fromarray(image).resize((tile * scale, tile * scale), Image.Resampling.NEAREST)
        canvas.paste(panel, (x, y))
        draw.text((x + 4, y + tile * scale + 2), title, fill=(0, 0, 0))
    ensure_dir(path.parent)
    canvas.save(path)


def risk_score_from_structure(stat: Dict[str, Any]) -> float:
    return (
        float(stat["border_ratio"])
        + float(stat["center_ratio"])
        + float(stat["selected_long_contour_ratio"])
        + min(float(stat["connected_components"]) / 18.0, 1.5)
    )


def audit_candidates(
    eval_records: Sequence[Dict[str, Any]],
    slic_fn: Any,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    structures: Dict[int, Dict[str, Any]] = {}
    for record in eval_records:
        mask, _score, region_count, base_variant = make_mask(NEARMISS_SPEC, record, 0.15, slic_fn, args)
        _base, comp, _coord = mask_structure_stats(
            record,
            NEARMISS_SPEC.source_type,
            NEARMISS_SPEC.name,
            0.15,
            mask,
            region_count,
            base_variant,
        )
        comp = dict(comp)
        comp["risk_score"] = risk_score_from_structure(comp)
        structures[int(record["graph_id"])] = comp
        by_class[class_name(record["label"])].append(record)

    rng = random.Random(int(args.seed))
    selected: List[Dict[str, Any]] = []
    for emotion in EMOTION_NAMES:
        group = list(by_class.get(emotion, []))
        if not group:
            continue
        group_sorted = sorted(group, key=lambda row: int(row["graph_id"]))
        random_k = min(int(args.audit_random_per_class), len(group_sorted))
        random_rows = rng.sample(group_sorted, k=random_k)
        random_ids = {int(row["graph_id"]) for row in random_rows}
        risk_pool = sorted(
            [row for row in group_sorted if int(row["graph_id"]) not in random_ids],
            key=lambda row: structures[int(row["graph_id"])]["risk_score"],
            reverse=True,
        )
        if len(risk_pool) < int(args.audit_risk_per_class):
            risk_pool = sorted(
                group_sorted,
                key=lambda row: structures[int(row["graph_id"])]["risk_score"],
                reverse=True,
            )
        risk_rows = risk_pool[: min(int(args.audit_risk_per_class), len(risk_pool))]
        seen: set[int] = set()
        for source, rows in (("random", random_rows), ("high_risk", risk_rows)):
            for record in rows:
                graph_id = int(record["graph_id"])
                if graph_id in seen:
                    continue
                seen.add(graph_id)
                selected.append(
                    {
                        "record": record,
                        "sample_source": source,
                        "risk_score": structures[graph_id]["risk_score"],
                    }
                )
    return selected, structures


def build_visual_audit_sheet(
    output_dir: Path,
    eval_records: Sequence[Dict[str, Any]],
    slic_fn: Any,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    figures_dir = ensure_dir(output_dir / "figures")
    masks_dir = ensure_dir(output_dir / "masks")
    candidates, _structures = audit_candidates(eval_records, slic_fn, args)
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(candidates, start=1):
        record = item["record"]
        mask, _score, _region_count, _base_variant = make_mask(NEARMISS_SPEC, record, 0.15, slic_fn, args)
        emotion = class_name(record["label"])
        stem = f"{safe_name(NEARMISS_SELECTOR)}__class_{safe_name(emotion)}__graph_{int(record['graph_id']):06d}"
        figure_path = figures_dir / safe_name(NEARMISS_SELECTOR) / f"class_{safe_name(emotion)}" / f"{stem}.png"
        mask_path = masks_dir / safe_name(NEARMISS_SELECTOR) / f"class_{safe_name(emotion)}" / f"{stem}_mask.png"
        render_figure(figure_path, record, mask, slic_fn, args)
        save_png(mask_path, np.asarray(mask, dtype=np.uint8).reshape(48, 48) * 255)
        rows.append(
            {
                "audit_id": f"STAGE49_{idx:04d}",
                "selector_name": NEARMISS_SELECTOR,
                "class_name": emotion,
                "graph_id": int(record["graph_id"]),
                "ratio": 0.15,
                "figure_path": str(figure_path),
                "selected_eye_eyebrow": "",
                "selected_mouth_nasolabial": "",
                "selected_face_muscle_cheek_wrinkle": "",
                "selected_hair_glasses": "",
                "selected_border_background": "",
                "long_contour_dominant": "",
                "center_collapse": "",
                "fragmented_pixel_dust": "",
                "region_like": "",
                "facial_evidence_like": "",
                "overall_visual_pass": "NEEDS_MANUAL_REVIEW",
                "notes": f"NEEDS_MANUAL_REVIEW; sample_source={item['sample_source']}; risk_score={item['risk_score']:.6f}",
            }
        )
    return rows


def evaluate_source(
    spec: SourceSpec,
    ratio: float,
    eval_records: Sequence[Dict[str, Any]],
    original_probe: Any,
    original_metrics: Dict[str, Any],
    visual_join: Dict[str, Dict[str, Any]],
    slic_fn: Any,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    x_eval_only: List[np.ndarray] = []
    x_eval_delete: List[np.ndarray] = []
    stat_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    for record in eval_records:
        mask, score, region_count, base_variant = make_mask(spec, record, ratio, slic_fn, args)
        x_eval_only.append(local_mean_apply(record["intensity"], mask, "only_selected"))
        x_eval_delete.append(local_mean_apply(record["intensity"], mask, "delete_selected"))
        _base, comp, coord = mask_structure_stats(
            record,
            spec.source_type,
            spec.name,
            ratio,
            mask,
            region_count,
            base_variant,
        )
        stat_rows.append(comp)
        _ = coord
        _ = score
    only_metrics, _ = eval_probe(original_probe, np.stack(x_eval_only, axis=0).astype(np.float32), args.y_eval)
    delete_metrics, _ = eval_probe(original_probe, np.stack(x_eval_delete, axis=0).astype(np.float32), args.y_eval)
    structure = aggregate_structure_stats(stat_rows)
    visual = visual_join[spec.name]
    metric_row = {
        "selector_name": spec.name,
        "source_type": spec.source_type,
        "weight_id": spec.weight_id,
        "aggregation": spec.aggregation,
        "retention_ratio": float(ratio),
        "original_macro_f1": original_metrics["macro_f1"],
        "only_selected_macro_f1": only_metrics["macro_f1"],
        "delete_selected_macro_f1": delete_metrics["macro_f1"],
        "deletion_drop": float(original_metrics["macro_f1"]) - float(delete_metrics["macro_f1"]),
        "components": structure["mean_connected_components"],
        "long_contour": structure["mean_selected_long_contour_ratio"],
        "border_ratio": structure["mean_border_ratio"],
        "center_ratio": structure["mean_center_ratio"],
        "visual_join_status": visual["visual_join_status"],
        "visual_metrics_usable_for_gate": visual["visual_metrics_usable_for_gate"],
        "visual_partial_pass_rate": visual["visual_partial_pass_rate"],
        "avg_facial_evidence_like": visual["avg_facial_evidence_like"],
        "avg_hair_glasses": visual["avg_hair_glasses"],
        "avg_border_background": visual["avg_border_background"],
        "avg_center_collapse": visual["avg_center_collapse"],
    }
    for eval_mode, metrics in (
        ("original", original_metrics),
        ("only_selected", only_metrics),
        ("delete_selected", delete_metrics),
    ):
        for emotion in EMOTION_NAMES:
            per_class_rows.append(
                {
                    "selector_name": spec.name,
                    "source_type": spec.source_type,
                    "weight_id": spec.weight_id,
                    "aggregation": spec.aggregation,
                    "retention_ratio": float(ratio),
                    "eval_mode": eval_mode,
                    "class_name": emotion,
                    "f1": metrics[f"per_class_f1_{emotion}"],
                }
            )
    return metric_row, per_class_rows


def build_gate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row["selector_name"] != NEARMISS_SELECTOR:
            continue
        deletion_ok = float(row["deletion_drop"]) >= 0.02
        random_ok = float(row["gap_vs_random"]) >= 0.02
        components_ok = float(row["components"]) < 18.0
        long_ok = float(row["long_contour"]) < 0.20
        direct_visual = int(row["visual_metrics_usable_for_gate"]) == 1
        visual_rate_ok = direct_visual and float(row["visual_partial_pass_rate"]) >= 0.60
        facial_ok = direct_visual and float(row["avg_facial_evidence_like"]) >= 1.20
        center_ok = float(row["center_ratio"]) < 0.35 and direct_visual and float(row["avg_center_collapse"]) < 1.0
        shortcut_ok = direct_visual and float(row["avg_hair_glasses"]) < 1.0 and float(row["avg_border_background"]) < 1.0
        all_ok = all([deletion_ok, random_ok, components_ok, long_ok, visual_rate_ok, facial_ok, center_ok, shortcut_ok])
        out.append(
            {
                "selector_name": row["selector_name"],
                "retention_ratio": row["retention_ratio"],
                "deletion_drop": row["deletion_drop"],
                "gap_vs_random": row["gap_vs_random"],
                "components": row["components"],
                "long_contour": row["long_contour"],
                "center_ratio": row["center_ratio"],
                "visual_join_status": row["visual_join_status"],
                "visual_partial_pass_rate": row["visual_partial_pass_rate"],
                "avg_facial_evidence_like": row["avg_facial_evidence_like"],
                "avg_hair_glasses": row["avg_hair_glasses"],
                "avg_border_background": row["avg_border_background"],
                "avg_center_collapse": row["avg_center_collapse"],
                "gate_deletion_drop": int(deletion_ok),
                "gate_gap_vs_random": int(random_ok),
                "gate_components": int(components_ok),
                "gate_long_contour": int(long_ok),
                "gate_direct_visual_partial_pass_rate": int(visual_rate_ok),
                "gate_avg_facial_evidence_like": int(facial_ok),
                "gate_no_center_collapse": int(center_ok),
                "gate_no_dominant_shortcut": int(shortcut_ok),
                "stage5_diagnostic_candidate": int(all_ok),
            }
        )
    return out


def pick_best_nearmiss(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = [row for row in rows if row["selector_name"] == NEARMISS_SELECTOR]
    return max(
        candidates,
        key=lambda row: (
            int(float(row["deletion_drop"]) >= 0.02),
            int(float(row["gap_vs_random"]) >= 0.02),
            float(row["gap_vs_random"]),
            float(row["deletion_drop"]),
        ),
    )


def decide(gates: Sequence[Dict[str, Any]], best: Dict[str, Any]) -> str:
    if any(int(row["stage5_diagnostic_candidate"]) == 1 for row in gates):
        return "OPEN_STAGE5_DIAGNOSTIC_FOR_ONE_SELECTOR"
    near_miss = (
        float(best["deletion_drop"]) >= 0.02
        and float(best["components"]) < 18.0
        and float(best["long_contour"]) < 0.20
    )
    if near_miss:
        return "KEEP_STAGE5_LOCKED_BUT_DOCUMENT_NEAR_MISS"
    return "STOP_STAGE5_PATH_DOCUMENT_FINDINGS"


def per_class_summary(per_class_rows: Sequence[Dict[str, Any]], best: Dict[str, Any]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    ratio = float(best["retention_ratio"])
    lookup = {
        (row["selector_name"], float(row["retention_ratio"]), row["eval_mode"], row["class_name"]): float(row["f1"])
        for row in per_class_rows
    }
    diffs: List[Tuple[str, float]] = []
    for emotion in EMOTION_NAMES:
        best_f1 = lookup[(best["selector_name"], ratio, "only_selected", emotion)]
        base_f1 = lookup[(BASE_SELECTOR, ratio, "only_selected", emotion)]
        diffs.append((emotion, best_f1 - base_f1))
    gains = sorted([item for item in diffs if item[1] > 0], key=lambda item: item[1], reverse=True)
    losses = sorted([item for item in diffs if item[1] < 0], key=lambda item: item[1])
    return gains, losses


def write_report(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    audit_rows: Sequence[Dict[str, Any]],
    audit_summary_rows: Sequence[Dict[str, Any]],
    gates: Sequence[Dict[str, Any]],
    best: Dict[str, Any],
    decision: str,
    gains: Sequence[Tuple[str, float]],
    losses: Sequence[Tuple[str, float]],
) -> None:
    gate_best = next(row for row in gates if float(row["retention_ratio"]) == float(best["retention_ratio"]))
    reviewed = reviewed_visual_rows(audit_rows)
    pass_count = sum(1 for row in reviewed if row["overall_visual_pass"] == "PASS")
    partial_count = sum(1 for row in reviewed if row["overall_visual_pass"] == "PARTIAL")
    fail_count = sum(1 for row in reviewed if row["overall_visual_pass"] == "FAIL")
    visual_rate = (pass_count + partial_count) / len(reviewed) if reviewed else float("nan")
    gate_failures = [
        label
        for label, key in [
            ("deletion_drop", "gate_deletion_drop"),
            ("gap_vs_random", "gate_gap_vs_random"),
            ("components", "gate_components"),
            ("long_contour", "gate_long_contour"),
            ("direct_visual_partial_pass_rate", "gate_direct_visual_partial_pass_rate"),
            ("avg_facial_evidence_like", "gate_avg_facial_evidence_like"),
            ("no_center_collapse", "gate_no_center_collapse"),
            ("no_dominant_shortcut", "gate_no_dominant_shortcut"),
        ]
        if int(gate_best.get(key, 0)) == 0
    ]
    summary_lookup = {row["selector_name"]: row for row in audit_summary_rows}
    direct_summary = summary_lookup.get(NEARMISS_SELECTOR, {})
    risk_case_count = sum(1 for row in audit_rows if "high_risk" in str(row.get("notes", "")))
    base_same_ratio = next(row for row in rows if row["selector_name"] == BASE_SELECTOR and float(row["retention_ratio"]) == float(best["retention_ratio"]))

    lines: List[str] = [
        "# Stage 4.9 Confirm Near-miss Hybrid SLIC Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Best selector: `{best['selector_name']}` @ `{fmt(best['retention_ratio'])}`.",
        f"- Gate result: `{'PASS' if int(gate_best['stage5_diagnostic_candidate']) else 'FAIL'}`.",
        f"- Stage 5: `{'yes' if decision.startswith('OPEN_') else 'no'}`.",
        f"- Main reason if no: `{', '.join(gate_failures) if gate_failures else 'none'}`.",
        f"- Final decision: `{decision}`.",
        "",
        "## 2. Direct Visual Audit",
        "",
        f"- Direct figures generated for near-miss selector: `{len(audit_rows)}`.",
        f"- Reviewed count: `{len(reviewed)}`; pass `{pass_count}` / partial `{partial_count}` / fail `{fail_count}`.",
        f"- Risk-case rows prepared: `{risk_case_count}`.",
        f"- Direct partial/pass rate: `{fmt(visual_rate) if math.isfinite(visual_rate) else 'NEEDS_MANUAL_REVIEW'}`.",
        f"- Facial-evidence candidate? `{'yes' if len(reviewed) and safe_float(direct_summary.get('avg_facial_evidence_like')) >= 1.2 else 'not confirmed'}`.",
        "- Current visual status is direct-sheet-only: no proxy family audit is accepted for the near-miss gate.",
        "",
        "## 3. Confirmatory Deletion",
        "",
        "| Selector | Ratio | Only F1 | Delete F1 | Drop | Gap random | Gap center | Gap SLIC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["selector_name"], float(item["retention_ratio"]))):
        if row["selector_name"] not in {NEARMISS_SELECTOR, BASE_SELECTOR}:
            continue
        lines.append(
            f"| `{row['selector_name']}` | {fmt(row['retention_ratio'])} | {fmt(row['only_selected_macro_f1'])} | "
            f"{fmt(row['delete_selected_macro_f1'])} | {fmt(row['deletion_drop'])} | {fmt(row['gap_vs_random'])} | "
            f"{fmt(row['gap_vs_center'])} | {fmt(row['gap_vs_slic'])} |"
        )
    lines.extend(
        [
            "",
            f"- Baseline at same ratio: drop `{fmt(base_same_ratio['deletion_drop'])}`, gap random `{fmt(base_same_ratio['gap_vs_random'])}`.",
            "",
            "### Per-class behavior",
        ]
    )
    if gains:
        lines.append("- Gains vs original hybrid at same ratio: " + ", ".join(f"{name} `{fmt(delta)}`" for name, delta in gains[:3]) + ".")
    else:
        lines.append("- Gains vs original hybrid at same ratio: none.")
    if losses:
        lines.append("- Losses vs original hybrid at same ratio: " + ", ".join(f"{name} `{fmt(delta)}`" for name, delta in losses[:3]) + ".")
    else:
        lines.append("- Losses vs original hybrid at same ratio: none.")
    lines.extend(
        [
            "",
            "## 4. Structure Quality",
            "",
            "| Selector | Ratio | Components | Long | Border | Center |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["selector_name"], float(item["retention_ratio"]))):
        if row["selector_name"] not in {NEARMISS_SELECTOR, BASE_SELECTOR}:
            continue
        lines.append(
            f"| `{row['selector_name']}` | {fmt(row['retention_ratio'])} | {fmt(row['components'])} | "
            f"{fmt(row['long_contour'])} | {fmt(row['border_ratio'])} | {fmt(row['center_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## 5. Gate Table",
            "",
            "| Ratio | Drop | Gap random | Components | Long | Direct visual | Facial | Center | Shortcut | Gate |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(gates, key=lambda item: float(item["retention_ratio"])):
        lines.append(
            f"| {fmt(row['retention_ratio'])} | {row['gate_deletion_drop']} | {row['gate_gap_vs_random']} | "
            f"{row['gate_components']} | {row['gate_long_contour']} | {row['gate_direct_visual_partial_pass_rate']} | "
            f"{row['gate_avg_facial_evidence_like']} | {row['gate_no_center_collapse']} | "
            f"{row['gate_no_dominant_shortcut']} | {row['stage5_diagnostic_candidate']} |"
        )
    lines.extend(
        [
            "",
            "## 6. Final Decision",
            "",
        ]
    )
    if decision == "OPEN_STAGE5_DIAGNOSTIC_FOR_ONE_SELECTOR":
        lines.append("- A. OPEN_STAGE5_DIAGNOSTIC_FOR_ONE_SELECTOR")
    elif decision == "KEEP_STAGE5_LOCKED_BUT_DOCUMENT_NEAR_MISS":
        lines.append("- B. KEEP_STAGE5_LOCKED_BUT_DOCUMENT_NEAR_MISS")
    else:
        lines.append("- C. STOP_STAGE5_PATH_DOCUMENT_FINDINGS")
    lines.extend(
        [
            "",
            "## 7. What Not To Claim",
            "",
            "- Không motif.",
            "- Không semantic part.",
            "- Không causal quá mức.",
            "- Không Q1.",
        ]
    )
    (output_dir / "stage49_confirm_nearmiss_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph_repo", default="artifacts/graph-repo/graph_repo")
    parser.add_argument("--visual_summary", default="outputs/stage4_6_visual_audit/visual_audit_summary_by_selector_filled.csv")
    parser.add_argument("--stage48_vs_controls", default="outputs/stage4_8_narrow_hybrid_slic_refinement/stage48_vs_controls.csv")
    parser.add_argument("--output_dir", default="outputs/stage4_9_confirm_nearmiss_hybrid_slic")
    parser.add_argument("--max_train_samples", type=parse_optional_int, default=5000)
    parser.add_argument("--max_val_samples", type=parse_optional_int, default=1000)
    parser.add_argument("--probe_train_cap", type=parse_optional_int, default=2000)
    parser.add_argument("--probe_eval_cap", type=parse_optional_int, default=1000)
    parser.add_argument("--ratios", nargs="+", type=float, default=list(RATIOS))
    parser.add_argument("--slic_segments", type=int, default=64)
    parser.add_argument("--slic_compactness", type=float, default=0.10)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--classifier_max_iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--audit_random_per_class", type=int, default=10)
    parser.add_argument("--audit_risk_per_class", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    output_dir = ensure_dir(Path(args.output_dir))
    require_inputs([Path(args.visual_summary), Path(args.stage48_vs_controls)])

    graph_repo = resolve_graph_repo(args.graph_repo)
    reader = GraphRepositoryReader(graph_repo)
    resolver = GraphResolver(reader.load_shared())
    slic_fn, slic_error = try_import_slic()
    if slic_fn is None:
        print(f"[Stage4.9] SLIC unavailable: {slic_error}")

    print("[Stage4.9] loading records")
    train_records_full = load_split_records(reader, resolver, "train", args.max_train_samples)
    eval_records_full = load_split_records(reader, resolver, "val", args.max_val_samples)
    train_records = maybe_cap(train_records_full, args.probe_train_cap)
    eval_records = maybe_cap(eval_records_full, args.probe_eval_cap)
    x_train_original, y_train, _ = make_original_dataset(train_records)
    x_eval_original, y_eval, _ = make_original_dataset(eval_records)
    args.y_eval = y_eval
    print(f"[Stage4.9] train_records={len(train_records)} eval_records={len(eval_records)}")

    audit_rows = build_visual_audit_sheet(output_dir, eval_records, slic_fn, args)
    audit_summary_rows = summarize_visual_audit(audit_rows)
    write_csv(output_dir / "stage49_visual_audit_sheet.csv", audit_rows, AUDIT_COLUMNS)
    write_csv(output_dir / "stage49_visual_audit_summary.csv", audit_summary_rows, STAGE46_SUMMARY_SELECTOR_COLUMNS)

    original_probe = train_probe(x_train_original, y_train, args.seed, args.classifier_max_iter)
    original_metrics, _ = eval_probe(original_probe, x_eval_original, y_eval)

    stage46_summary = load_visual_summary(Path(args.visual_summary))
    visual_join = build_visual_join(stage46_summary, audit_summary_rows)

    metric_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    for spec in ALL_SPECS:
        for ratio in [float(value) for value in args.ratios]:
            metric_row, per_class = evaluate_source(
                spec,
                ratio,
                eval_records,
                original_probe,
                original_metrics,
                visual_join,
                slic_fn,
                args,
            )
            metric_rows.append(metric_row)
            per_class_rows.extend(per_class)
            print(
                f"[Stage4.9] {spec.name} ratio={ratio:.2f} only_f1={fmt(metric_row['only_selected_macro_f1'])} "
                f"delete_f1={fmt(metric_row['delete_selected_macro_f1'])} drop={fmt(metric_row['deletion_drop'])}"
            )

    vs_controls = add_control_gaps(metric_rows)
    gate_rows = build_gate_rows(vs_controls)
    best = pick_best_nearmiss(vs_controls)
    decision = decide(gate_rows, best)
    gains, losses = per_class_summary(per_class_rows, best)

    write_csv(output_dir / "stage49_confirm_deletion_metrics.csv", vs_controls)
    write_csv(output_dir / "stage49_gate_table.csv", gate_rows)
    write_csv(output_dir / "stage49_per_class_metrics.csv", per_class_rows)
    write_report(output_dir, vs_controls, audit_rows, audit_summary_rows, gate_rows, best, decision, gains, losses)

    reviewed = reviewed_visual_rows(audit_rows)
    partial_pass_rate = (
        sum(1 for row in reviewed if row["overall_visual_pass"] in {"PASS", "PARTIAL"}) / len(reviewed)
        if reviewed
        else float("nan")
    )
    print(f"[Stage4.9] output_dir={output_dir}")
    print(f"[Stage4.9] selector={best['selector_name']} ratio={fmt(best['retention_ratio'])}")
    print(f"[Stage4.9] drop={fmt(best['deletion_drop'])}")
    print(f"[Stage4.9] gap_vs_random={fmt(best['gap_vs_random'])}")
    print(f"[Stage4.9] visual_partial_pass_rate={fmt(partial_pass_rate) if math.isfinite(partial_pass_rate) else 'NEEDS_MANUAL_REVIEW'}")
    print(f"[Stage4.9] components={fmt(best['components'])}")
    print(f"[Stage4.9] long_contour={fmt(best['long_contour'])}")
    print(f"[Stage4.9] final_decision={decision}")


if __name__ == "__main__":
    main()
