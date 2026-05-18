#!/usr/bin/env python
"""Stage 4.8 narrow heuristic refinement for hybrid_slic_region.

This stage stays deliberately small:
- no learned selector;
- no new model training;
- no motif, SupCon, part grouping, or Stage 5 implementation.

It only evaluates a fixed, narrow hybrid-SLIC grid around the near-pass
selector from Stage 4.7 under the same fixed-original + local-mean probe.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
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
from stage1_pixel_region_selection import ensure_dir, fmt, resolve_graph_repo, topk_mask, try_import_slic
from stage2_retention_deletion_test import make_original_dataset
from stage3_hybrid_evidence_selector import WEIGHT_GRID as STAGE3_WEIGHT_GRID, build_hybrid_mask, hybrid_score
from stage36_structure_aware_diagnostics import aggregate_stats as aggregate_structure_stats
from stage36_structure_aware_diagnostics import load_split_records, mask_structure_stats
from stage4_learned_evidence_selector import build_control_mask, score_correlations
from stage4_refinement_run import eval_probe, train_probe
from stage4_refinement_v2 import local_mean_apply


BASE_SELECTOR = "hybrid_slic_region__E_balanced__b0p1"
BASE_HYBRID_PIXEL = "hybrid_pixel_score__E_balanced__b0p0"
RATIOS = (0.10, 0.15, 0.20)
GATE_RATIOS = (0.15, 0.20)
AGGREGATIONS = ("mean", "top20_mean", "top30_mean", "max_mean_mix")
WEIGHT_VARIANTS: Dict[str, Tuple[float, float, float]] = {
    "E_balanced": (0.40, 0.40, 0.20),
    "E_delta_grad_45_45_10": (0.45, 0.45, 0.10),
    "E_delta_50": (0.50, 0.40, 0.10),
    "E_grad_50": (0.40, 0.50, 0.10),
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    source_type: str
    weight_id: str = ""
    aggregation: str = ""


REFINED_VARIANTS: Tuple[SourceSpec, ...] = tuple(
    SourceSpec(
        name=f"hybrid_slic_region__{weight_id}__b0p1__{aggregation}",
        source_type="refined_variant",
        weight_id=weight_id,
        aggregation=aggregation,
    )
    for weight_id in WEIGHT_VARIANTS
    for aggregation in AGGREGATIONS
)

CONTROL_SPECS: Tuple[SourceSpec, ...] = tuple(
    [
        SourceSpec("random_pixel", "control"),
        SourceSpec("center_prior", "control"),
        SourceSpec("gradient_topk", "control"),
        SourceSpec("delta_edge_topk", "control"),
        SourceSpec("slic_region_proposal", "control"),
        SourceSpec(BASE_SELECTOR, "control_hybrid"),
        SourceSpec(BASE_HYBRID_PIXEL, "control_hybrid"),
    ]
)

ALL_SPECS = REFINED_VARIANTS + CONTROL_SPECS


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    if not fields:
        fields = ["note"]
        rows = [{"note": "NO_ROWS"}]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def parse_optional_int(value: str) -> Optional[int]:
    if value.lower() in {"none", "full", "all"}:
        return None
    return int(value)


def maybe_cap(records: Sequence[Dict[str, Any]], cap: Optional[int]) -> List[Dict[str, Any]]:
    if cap is None:
        return list(records)
    return list(records[: min(len(records), int(cap))])


def require_inputs(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Stage 4.8 inputs:\n- " + "\n- ".join(missing))


def load_visual_summary(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in read_csv(path):
        name = row.get("selector_name", "")
        if not name:
            continue
        out[name] = {
            "visual_partial_pass_rate": safe_float(row.get("partial_or_pass_rate")),
            "avg_facial_evidence_like": safe_float(row.get("avg_facial_evidence_like")),
            "avg_hair_glasses": safe_float(row.get("avg_hair_glasses")),
            "avg_border_background": safe_float(row.get("avg_border_background")),
        }
    return out


def build_visual_join_rows(summary: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_visual = summary.get(BASE_SELECTOR, {})
    for spec in ALL_SPECS:
        if spec.name in summary:
            source = spec.name
            status = "direct"
            usable = 1
            values = summary[spec.name]
        elif spec.source_type == "refined_variant" and base_visual:
            source = BASE_SELECTOR
            status = "family_proxy_only"
            usable = 0
            values = base_visual
        else:
            source = ""
            status = "missing"
            usable = 0
            values = {}
        rows.append(
            {
                "selector_name": spec.name,
                "source_type": spec.source_type,
                "visual_join_status": status,
                "visual_source_selector_name": source,
                "visual_metrics_usable_for_gate": usable,
                "visual_partial_pass_rate": values.get("visual_partial_pass_rate", float("nan")),
                "avg_facial_evidence_like": values.get("avg_facial_evidence_like", float("nan")),
                "avg_hair_glasses": values.get("avg_hair_glasses", float("nan")),
                "avg_border_background": values.get("avg_border_background", float("nan")),
            }
        )
    return rows


def region_score(values: np.ndarray, aggregation: str) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 0:
        return 0.0
    if aggregation == "mean":
        return float(values.mean())
    if aggregation == "top20_mean":
        k = max(1, int(math.ceil(values.size * 0.20)))
        return float(np.partition(values, values.size - k)[values.size - k :].mean())
    if aggregation == "top30_mean":
        k = max(1, int(math.ceil(values.size * 0.30)))
        return float(np.partition(values, values.size - k)[values.size - k :].mean())
    if aggregation == "max_mean_mix":
        return float(0.70 * values.mean() + 0.30 * values.max())
    raise ValueError(f"Unknown aggregation: {aggregation}")


def refined_slic_region_mask(
    intensity: np.ndarray,
    score: np.ndarray,
    k: int,
    slic_fn: Any,
    n_segments: int,
    compactness: float,
    aggregation: str,
) -> Tuple[np.ndarray, int]:
    if slic_fn is None:
        return topk_mask(score, k), 0
    segments = slic_fn(
        intensity.reshape(48, 48).astype(np.float32),
        n_segments=int(n_segments),
        compactness=float(compactness),
        start_label=0,
        channel_axis=None,
    ).reshape(-1)
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    ranked: List[Tuple[float, int]] = []
    for region_id in np.unique(segments):
        values = score[segments == region_id]
        ranked.append((region_score(values, aggregation), int(region_id)))
    ranked.sort(reverse=True)
    selected = np.zeros_like(score, dtype=bool)
    region_count = 0
    for _, region_id in ranked:
        selected[segments == region_id] = True
        region_count += 1
        if int(selected.sum()) >= int(k):
            break
    return selected, region_count


def make_mask(
    spec: SourceSpec,
    record: Dict[str, Any],
    ratio: float,
    slic_fn: Any,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    if spec.source_type == "refined_variant":
        weights = WEIGHT_VARIANTS[spec.weight_id]
        score = hybrid_score(record, weights, 0.10)
        k = max(1, int(round(score.size * float(ratio))))
        mask, region_count = refined_slic_region_mask(
            record["intensity"],
            score,
            k,
            slic_fn,
            args.slic_segments,
            args.slic_compactness,
            spec.aggregation,
        )
        return mask, score, region_count, spec.weight_id
    if spec.source_type == "control_hybrid" and spec.name == BASE_SELECTOR:
        mask, score, region_count = build_hybrid_mask(
            record,
            "hybrid_slic_region",
            STAGE3_WEIGHT_GRID["E_balanced"],
            0.10,
            ratio,
            slic_fn,
            args.slic_segments,
            args.slic_compactness,
            args.smooth_sigma,
        )
        return mask, score, region_count, "E_balanced"
    if spec.source_type == "control_hybrid" and spec.name == BASE_HYBRID_PIXEL:
        mask, score, region_count = build_hybrid_mask(
            record,
            "hybrid_pixel_score",
            STAGE3_WEIGHT_GRID["E_balanced"],
            0.0,
            ratio,
            slic_fn,
            args.slic_segments,
            args.slic_compactness,
            args.smooth_sigma,
        )
        return mask, score, region_count, "E_balanced"
    mask, score, region_count = build_control_mask(
        record,
        spec.name,
        ratio,
        slic_fn,
        args.slic_segments,
        args.slic_compactness,
        args.smooth_sigma,
        args.seed,
    )
    return mask, score, region_count, ""


def evaluate_source(
    spec: SourceSpec,
    ratio: float,
    eval_records: Sequence[Dict[str, Any]],
    original_probe: Any,
    original_metrics: Dict[str, Any],
    visual_join: Dict[str, Dict[str, Any]],
    slic_fn: Any,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    x_eval_only: List[np.ndarray] = []
    x_eval_delete: List[np.ndarray] = []
    stat_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    for record in eval_records:
        mask, score, region_count, base_variant = make_mask(spec, record, ratio, slic_fn, args)
        x_eval_only.append(local_mean_apply(record["intensity"], mask, "only_selected"))
        x_eval_delete.append(local_mean_apply(record["intensity"], mask, "delete_selected"))
        _, comp, coord = mask_structure_stats(
            record,
            spec.source_type,
            spec.name,
            ratio,
            mask,
            region_count,
            base_variant,
        )
        corr = score_correlations(record, score)
        comp.update(corr)
        coord.update(corr)
        stat_rows.append(comp)
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
    }
    structure_row = {
        "selector_name": spec.name,
        "source_type": spec.source_type,
        "weight_id": spec.weight_id,
        "aggregation": spec.aggregation,
        "retention_ratio": float(ratio),
        **structure,
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
    return metric_row, structure_row, per_class_rows


def add_control_gaps(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {(row["selector_name"], float(row["retention_ratio"])): row for row in rows}
    out: List[Dict[str, Any]] = []
    for row in rows:
        ratio = float(row["retention_ratio"])
        random_row = lookup[("random_pixel", ratio)]
        center_row = lookup[("center_prior", ratio)]
        slic_row = lookup[("slic_region_proposal", ratio)]
        original_hybrid = lookup[(BASE_SELECTOR, ratio)]
        out.append(
            {
                **row,
                "gap_vs_random": float(row["only_selected_macro_f1"]) - float(random_row["only_selected_macro_f1"]),
                "gap_vs_center": float(row["only_selected_macro_f1"]) - float(center_row["only_selected_macro_f1"]),
                "gap_vs_slic": float(row["only_selected_macro_f1"]) - float(slic_row["only_selected_macro_f1"]),
                "gap_vs_original_hybrid_slic": float(row["only_selected_macro_f1"]) - float(original_hybrid["only_selected_macro_f1"]),
            }
        )
    return out


def build_gate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row["source_type"] != "refined_variant" or float(row["retention_ratio"]) not in GATE_RATIOS:
            continue
        deletion_ok = float(row["deletion_drop"]) >= 0.02
        random_ok = float(row["gap_vs_random"]) >= 0.02
        components_ok = float(row["components"]) < 18.0
        long_ok = float(row["long_contour"]) < 0.20
        center_ok = float(row["center_ratio"]) < 0.35
        visual_direct = int(row["visual_metrics_usable_for_gate"]) == 1
        visual_rate_ok = visual_direct and float(row["visual_partial_pass_rate"]) >= 0.60
        facial_ok = visual_direct and float(row["avg_facial_evidence_like"]) >= 1.20
        all_ok = all([deletion_ok, random_ok, components_ok, long_ok, center_ok, visual_rate_ok, facial_ok])
        out.append(
            {
                "selector_name": row["selector_name"],
                "weight_id": row["weight_id"],
                "aggregation": row["aggregation"],
                "retention_ratio": row["retention_ratio"],
                "deletion_drop": row["deletion_drop"],
                "gap_vs_random": row["gap_vs_random"],
                "gap_vs_original_hybrid_slic": row["gap_vs_original_hybrid_slic"],
                "components": row["components"],
                "long_contour": row["long_contour"],
                "center_ratio": row["center_ratio"],
                "visual_join_status": row["visual_join_status"],
                "gate_deletion_drop": int(deletion_ok),
                "gate_gap_vs_random": int(random_ok),
                "gate_components": int(components_ok),
                "gate_long_contour": int(long_ok),
                "gate_center_not_collapsed": int(center_ok),
                "gate_visual_partial_pass_rate": int(visual_rate_ok),
                "gate_avg_facial_evidence_like": int(facial_ok),
                "stage5_diagnostic_candidate": int(all_ok),
            }
        )
    return out


def load_stage47_base(path: Path) -> Dict[str, Dict[float, Dict[str, str]]]:
    rows = read_csv(path)
    out: Dict[str, Dict[float, Dict[str, str]]] = {}
    for row in rows:
        selector = row.get("selector_name", "")
        ratio = safe_float(row.get("retention_ratio"))
        if not selector or not math.isfinite(ratio):
            continue
        out.setdefault(selector, {})[ratio] = row
    return out


def pick_best_variant(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = [row for row in rows if row["source_type"] == "refined_variant" and float(row["retention_ratio"]) in GATE_RATIOS]
    return max(
        candidates,
        key=lambda row: (
            int(float(row["deletion_drop"]) >= 0.02)
            + int(float(row["gap_vs_random"]) >= 0.02)
            + int(float(row["components"]) < 18.0)
            + int(float(row["long_contour"]) < 0.20)
            + int(float(row["center_ratio"]) < 0.35),
            int(float(row["deletion_drop"]) >= 0.02),
            int(float(row["gap_vs_random"]) >= 0.02),
            float(row["gap_vs_random"]),
            float(row["deletion_drop"]),
            -float(row["components"]),
            -float(row["long_contour"]),
        ),
    )


def decide(gates: Sequence[Dict[str, Any]], best: Dict[str, Any], stage47_base_20: Dict[str, str]) -> str:
    if any(int(row["stage5_diagnostic_candidate"]) == 1 for row in gates):
        return "OPEN_STAGE5_DIAGNOSTIC_FOR_HEURISTIC_SELECTOR"
    numeric_promising = any(
        int(row["gate_deletion_drop"]) == 1
        and int(row["gate_gap_vs_random"]) == 1
        and int(row["gate_components"]) == 1
        and int(row["gate_long_contour"]) == 1
        and int(row["gate_center_not_collapsed"]) == 1
        for row in gates
    )
    improved_near_miss = (
        float(best["deletion_drop"]) >= safe_float(stage47_base_20.get("deletion_drop"))
        and float(best["gap_vs_random"]) > safe_float(stage47_base_20.get("gap_vs_random"))
        and float(best["components"]) < 18.0
        and float(best["long_contour"]) < 0.20
    )
    if numeric_promising or improved_near_miss:
        return "KEEP_STAGE5_LOCKED_BUT_HEURISTIC_PROMISING"
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
    gates: Sequence[Dict[str, Any]],
    best: Dict[str, Any],
    decision: str,
    stage47_base_20: Dict[str, str],
    gains: Sequence[Tuple[str, float]],
    losses: Sequence[Tuple[str, float]],
) -> None:
    refined_rows = [row for row in rows if row["source_type"] == "refined_variant"]
    base_control = next(row for row in rows if row["selector_name"] == BASE_SELECTOR and float(row["retention_ratio"]) == float(best["retention_ratio"]))
    stage47_drop = safe_float(stage47_base_20.get("deletion_drop"))
    stage47_gap = safe_float(stage47_base_20.get("gap_vs_random"))
    exceeded_stage47 = float(best["deletion_drop"]) > stage47_drop or float(best["gap_vs_random"]) > stage47_gap
    direct_visual_count = sum(1 for row in rows if row["source_type"] == "refined_variant" and row["visual_join_status"] == "direct")
    gate_lookup = [row for row in gates if row["selector_name"] == best["selector_name"] and float(row["retention_ratio"]) == float(best["retention_ratio"])]
    best_gate = gate_lookup[0] if gate_lookup else {}
    gate_failures = [
        label
        for label, key in [
            ("deletion_drop", "gate_deletion_drop"),
            ("gap_vs_random", "gate_gap_vs_random"),
            ("components", "gate_components"),
            ("long_contour", "gate_long_contour"),
            ("center_ratio", "gate_center_not_collapsed"),
            ("visual_partial_pass_rate", "gate_visual_partial_pass_rate"),
            ("avg_facial_evidence_like", "gate_avg_facial_evidence_like"),
        ]
        if int(best_gate.get(key, 0)) == 0
    ]
    lines: List[str] = [
        "# Stage 4.8 Narrow Hybrid SLIC Refinement Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Best variant: `{best['selector_name']}` @ `{fmt(best['retention_ratio'])}`.",
        f"- Gate result: `{'PASS' if int(best_gate.get('stage5_diagnostic_candidate', 0)) else 'FAIL'}`.",
        f"- Stage 5: `{'yes, diagnostic only' if decision.startswith('OPEN_') else 'no'}`.",
        f"- Có vượt Stage 4.7 không? `{'yes' if exceeded_stage47 else 'no'}` "
        f"(Stage 4.7 base @20 drop `{fmt(stage47_drop)}`, gap_random `{fmt(stage47_gap)}`; "
        f"best current drop `{fmt(best['deletion_drop'])}`, gap_random `{fmt(best['gap_vs_random'])}`).",
        f"- Final decision: `{decision}`.",
        "",
        "## 2. Setup",
        "",
        "- Selector family: `hybrid_slic_region` only.",
        f"- Refined variants: `{len(REFINED_VARIANTS)}` = 4 aggregations x 4 weight variants.",
        "- Aggregations: `mean`, `top20_mean`, `top30_mean`, `max_mean_mix`.",
        "- Weight variants: `E_balanced`, `E_delta_grad_45_45_10`, `E_delta_50`, `E_grad_50`.",
        "- Ratios: `0.10`, `0.15`, `0.20`.",
        "- Controls: `random_pixel`, `center_prior`, `gradient_topk`, `delta_edge_topk`, `slic_region_proposal`, original `hybrid_slic_region__E_balanced__b0p1`, `hybrid_pixel_score__E_balanced__b0p0`.",
        "- Protocol: `fixed_original_probe` only; train probe on original images; evaluate original / only_selected / delete_selected; fill mode `local_mean`; cap `2000 train / 1000 val`.",
        "",
        "## 3. Deletion Results",
        "",
        "| Ratio | Variant | Only F1 | Delete F1 | Drop | Gap random | Gap center | Gap SLIC | Gap original hybrid |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        refined_rows,
        key=lambda item: (float(item["retention_ratio"]), -float(item["deletion_drop"]), item["selector_name"]),
    ):
        lines.append(
            f"| {fmt(row['retention_ratio'])} | `{row['selector_name']}` | {fmt(row['only_selected_macro_f1'])} | "
            f"{fmt(row['delete_selected_macro_f1'])} | {fmt(row['deletion_drop'])} | {fmt(row['gap_vs_random'])} | "
            f"{fmt(row['gap_vs_center'])} | {fmt(row['gap_vs_slic'])} | {fmt(row['gap_vs_original_hybrid_slic'])} |"
        )
    lines.extend(
        [
            "",
            f"Original hybrid control at the best ratio `{fmt(best['retention_ratio'])}`: "
            f"drop `{fmt(base_control['deletion_drop'])}`, gap random `{fmt(base_control['gap_vs_random'])}`.",
            "",
            "## 4. Structure Quality",
            "",
            "| Variant | Ratio | Components | Long | Border | Center |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        refined_rows,
        key=lambda item: (float(item["retention_ratio"]), float(item["components"]), float(item["long_contour"])),
    ):
        lines.append(
            f"| `{row['selector_name']}` | {fmt(row['retention_ratio'])} | {fmt(row['components'])} | "
            f"{fmt(row['long_contour'])} | {fmt(row['border_ratio'])} | {fmt(row['center_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## 5. Visual Audit Join",
            "",
            f"- Direct visual joins for new refined variants: `{direct_visual_count}`.",
            "- New Stage 4.8 variants do not have direct visual-audit rows yet; they carry only `family_proxy_only` context from the Stage 4.6 base selector.",
            "- Proxy context is reported but **not** accepted for the visual gate.",
            "",
            "## 6. Per-class Analysis",
            "",
        ]
    )
    if gains:
        gain_text = ", ".join(f"{name} `{fmt(delta)}`" for name, delta in gains[:3])
        lines.append(f"- Classes improved vs original hybrid at the same ratio: {gain_text}.")
    else:
        lines.append("- No class improved vs original hybrid at the same ratio.")
    if losses:
        loss_text = ", ".join(f"{name} `{fmt(delta)}`" for name, delta in losses[:3])
        lines.append(f"- Classes that lost vs original hybrid at the same ratio: {loss_text}.")
    else:
        lines.append("- No class lost vs original hybrid at the same ratio.")
    lines.extend(
        [
            "",
            "## 7. Gate Decision",
            "",
            "| Variant | Ratio | Drop | Gap random | Components | Long | Center | Visual direct | Gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(gates, key=lambda item: (float(item["retention_ratio"]), item["selector_name"])):
        lines.append(
            f"| `{row['selector_name']}` | {fmt(row['retention_ratio'])} | {row['gate_deletion_drop']} | "
            f"{row['gate_gap_vs_random']} | {row['gate_components']} | {row['gate_long_contour']} | "
            f"{row['gate_center_not_collapsed']} | {1 if row['visual_join_status'] == 'direct' else 0} | "
            f"{row['stage5_diagnostic_candidate']} |"
        )
    lines.extend(
        [
            "",
            f"- Best-variant gate failures: `{', '.join(gate_failures) if gate_failures else 'none'}`.",
            "",
            "## 8. Final Decision",
            "",
        ]
    )
    if decision == "OPEN_STAGE5_DIAGNOSTIC_FOR_HEURISTIC_SELECTOR":
        lines.append("- Pass: chỉ mở Stage 5 diagnostic rất hẹp cho đúng selector đã qua gate.")
    elif decision == "KEEP_STAGE5_LOCKED_BUT_HEURISTIC_PROMISING":
        lines.append("- Promising but fail: không mở Stage 5; ghi nhận đây là near-miss và cần visual audit trực tiếp nếu còn muốn xét tiếp.")
    else:
        lines.append("- Fail rõ: dừng đường Stage 5 và document findings.")
    lines.extend(
        [
            "",
            "## 9. What Not To Claim",
            "",
            "- Không motif.",
            "- Không semantic part.",
            "- Không causal quá mức.",
            "- Không Q1.",
        ]
    )
    (output_dir / "stage48_narrow_hybrid_slic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph_repo", default="artifacts/graph-repo/graph_repo")
    parser.add_argument("--stage47_vs_controls", default="outputs/stage4_7_fixed_original_deletion_shortlist/stage47_vs_controls.csv")
    parser.add_argument("--visual_summary", default="outputs/stage4_6_visual_audit/visual_audit_summary_by_selector_filled.csv")
    parser.add_argument("--output_dir", default="outputs/stage4_8_narrow_hybrid_slic_refinement")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = ensure_dir(Path(args.output_dir))
    require_inputs([Path(args.stage47_vs_controls), Path(args.visual_summary)])
    graph_repo = resolve_graph_repo(args.graph_repo)
    reader = GraphRepositoryReader(graph_repo)
    resolver = GraphResolver(reader.load_shared())
    slic_fn, slic_error = try_import_slic()
    if slic_fn is None:
        print(f"[Stage4.8] SLIC unavailable: {slic_error}")
    print("[Stage4.8] loading records")
    train_records_full = load_split_records(reader, resolver, "train", args.max_train_samples)
    eval_records_full = load_split_records(reader, resolver, "val", args.max_val_samples)
    train_records = maybe_cap(train_records_full, args.probe_train_cap)
    eval_records = maybe_cap(eval_records_full, args.probe_eval_cap)
    x_train_original, y_train, _ = make_original_dataset(train_records)
    x_eval_original, y_eval, _ = make_original_dataset(eval_records)
    args.y_eval = y_eval
    print(f"[Stage4.8] train_records={len(train_records)} eval_records={len(eval_records)}")
    original_probe = train_probe(x_train_original, y_train, args.seed, args.classifier_max_iter)
    original_metrics, _ = eval_probe(original_probe, x_eval_original, y_eval)

    visual_summary = load_visual_summary(Path(args.visual_summary))
    visual_join_rows = build_visual_join_rows(visual_summary)
    visual_join = {row["selector_name"]: row for row in visual_join_rows}

    metric_rows: List[Dict[str, Any]] = []
    structure_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    for spec in ALL_SPECS:
        for ratio in [float(value) for value in args.ratios]:
            metric_row, structure_row, per_class = evaluate_source(
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
            structure_rows.append(structure_row)
            per_class_rows.extend(per_class)
            print(
                f"[Stage4.8] {spec.name} ratio={ratio:.2f} only_f1={fmt(metric_row['only_selected_macro_f1'])} "
                f"delete_f1={fmt(metric_row['delete_selected_macro_f1'])} drop={fmt(metric_row['deletion_drop'])}"
            )

    vs_controls = add_control_gaps(metric_rows)
    gate_table = build_gate_rows(vs_controls)
    stage47 = load_stage47_base(Path(args.stage47_vs_controls))
    stage47_base_20 = stage47[BASE_SELECTOR][0.20]
    best = pick_best_variant(vs_controls)
    decision = decide(gate_table, best, stage47_base_20)
    gains, losses = per_class_summary(per_class_rows, best)

    write_csv(output_dir / "stage48_deletion_metrics.csv", metric_rows)
    write_csv(output_dir / "stage48_vs_controls.csv", vs_controls)
    write_csv(output_dir / "stage48_gate_table.csv", gate_table)
    write_csv(output_dir / "stage48_per_class_metrics.csv", per_class_rows)
    write_csv(output_dir / "stage48_selector_structure_stats.csv", structure_rows)
    write_csv(output_dir / "stage48_visual_audit_join.csv", visual_join_rows)
    write_report(output_dir, vs_controls, gate_table, best, decision, stage47_base_20, gains, losses)

    print(f"[Stage4.8] output_dir={output_dir}")
    print(f"[Stage4.8] original_macro_f1={fmt(original_metrics['macro_f1'])}")
    print(f"[Stage4.8] best_variant={best['selector_name']} ratio={fmt(best['retention_ratio'])}")
    print(f"[Stage4.8] best_drop={fmt(best['deletion_drop'])} best_gap_random={fmt(best['gap_vs_random'])}")
    print(f"[Stage4.8] decision={decision}")


if __name__ == "__main__":
    main()
