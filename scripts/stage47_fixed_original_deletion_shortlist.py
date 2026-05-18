#!/usr/bin/env python
"""Stage 4.7 fixed-original deletion check for the visual-audited shortlist.

This stage is analysis-only:
- no new selector training;
- no learned-selector expansion;
- no motif/SupCon/Stage 5 implementation.

It trains exactly one lightweight probe on original images, then evaluates
original / only-selected / delete-selected views for the visual-audited
shortlist plus controls under the fixed-original + local-mean protocol.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
from stage1_pixel_region_selection import ensure_dir, fmt, resolve_graph_repo, safe_name, topk_mask, try_import_slic
from stage2_retention_deletion_test import class_name, make_original_dataset
from stage36_structure_aware_diagnostics import load_split_records
from stage4_learned_evidence_selector import (
    build_control_mask,
    build_model,
    build_teacher,
    input_feature_names,
    mask_structure_stats,
    score_correlations,
)
from stage4_refinement_run import eval_probe, train_probe
from stage4_refinement_v2 import local_mean_apply, predict_score


RATIOS = (0.10, 0.20)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    source_type: str
    visual_selector_name: str
    variant_name: str = ""
    checkpoint_name: str = ""


SHORTLIST = (
    SourceSpec(
        "pixel_mlp_no_xy_r0p1_baseline_rerun",
        "learned_auxiliary",
        "pixel_mlp_no_xy_r0p1_baseline_rerun",
        checkpoint_name="pixel_mlp_no_xy_r0p1_baseline_rerun.pt",
    ),
    SourceSpec(
        "tiny_conv_struct_reg_r0p1_soft_teacher",
        "learned_auxiliary",
        "tiny_conv_struct_reg_r0p1_soft_teacher",
        checkpoint_name="tiny_conv_struct_reg_r0p1_soft_teacher.pt",
    ),
    SourceSpec(
        "tiny_conv_struct_reg_r0p1_long_border_penalty_light",
        "learned_auxiliary",
        "tiny_conv_struct_reg_r0p1_long_border_penalty_light",
        checkpoint_name="tiny_conv_struct_reg_r0p1_long_border_penalty_light.pt",
    ),
    SourceSpec(
        "structure_slic_region",
        "heuristic",
        "structure_slic_region",
        variant_name="structure_slic_region__B_balanced__s0p2_o0p1_sm0p2_l0p2_b0p1",
    ),
    SourceSpec(
        "hybrid_slic_region__E_balanced__b0p1",
        "heuristic",
        "hybrid_slic_region__E_balanced__b0p1",
        variant_name="hybrid_slic_region__E_balanced__b0p1",
    ),
    SourceSpec(
        "hybrid_pixel_score__E_balanced__b0p0",
        "heuristic",
        "hybrid_pixel_score__E_balanced__b0p0",
        variant_name="hybrid_pixel_score__E_balanced__b0p0",
    ),
)

CONTROLS = tuple(
    SourceSpec(name, "control", "")
    for name in (
        "random_pixel",
        "center_prior",
        "gradient_topk",
        "delta_edge_topk",
        "slic_region_proposal",
    )
)

ALL_SOURCES = SHORTLIST + CONTROLS


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


def load_visual_summary(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_csv(path)
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
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


def load_learned_models(checkpoint_dir: Path, device: torch.device) -> Tuple[Dict[str, torch.nn.Module], Dict[str, str]]:
    models: Dict[str, torch.nn.Module] = {}
    input_variants: Dict[str, str] = {}
    for spec in SHORTLIST:
        if spec.source_type != "learned_auxiliary":
            continue
        ckpt_path = checkpoint_dir / spec.checkpoint_name
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        variant = dict(payload["variant"])
        input_variant = str(variant["input_variant"])
        arch = str(variant["arch"])
        input_dim = int(payload.get("input_dim", len(input_feature_names(input_variant))))
        model = build_model(arch, input_dim).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        models[spec.name] = model
        input_variants[spec.name] = input_variant
    return models, input_variants


def make_mask(
    spec: SourceSpec,
    record: Dict[str, Any],
    ratio: float,
    models: Dict[str, torch.nn.Module],
    input_variants: Dict[str, str],
    slic_fn: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    if spec.source_type == "control":
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
    if spec.source_type == "heuristic":
        score, mask, variant_name = build_teacher(
            record,
            spec.variant_name,
            ratio,
            slic_fn,
            args.slic_segments,
            args.slic_compactness,
            args.smooth_sigma,
        )
        return mask.astype(bool), score, 0, variant_name
    model = models[spec.name]
    input_variant = input_variants[spec.name]
    score = predict_score(model, record, input_variant, device)
    k = max(1, int(round(score.size * float(ratio))))
    return topk_mask(score, k), score, 0, input_variant


def aggregate_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    fields = [
        "border_ratio",
        "center_ratio",
        "connected_components",
        "selected_long_contour_ratio",
    ]
    out: Dict[str, float] = {}
    for field in fields:
        vals = [float(row.get(field, 0.0)) for row in rows]
        out[field] = float(np.mean(vals)) if vals else float("nan")
    return out


def evaluate_source(
    spec: SourceSpec,
    ratio: float,
    train_records: Sequence[Dict[str, Any]],
    eval_records: Sequence[Dict[str, Any]],
    original_probe: Any,
    original_metrics: Dict[str, Any],
    models: Dict[str, torch.nn.Module],
    input_variants: Dict[str, str],
    visual_summary: Dict[str, Dict[str, float]],
    slic_fn: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    x_eval_only: List[np.ndarray] = []
    x_eval_delete: List[np.ndarray] = []
    stat_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []

    for record in eval_records:
        mask, score, region_count, variant_or_input = make_mask(
            spec, record, ratio, models, input_variants, slic_fn, args, device
        )
        x_eval_only.append(local_mean_apply(record["intensity"], mask, "only_selected"))
        x_eval_delete.append(local_mean_apply(record["intensity"], mask, "delete_selected"))
        _, comp, coord = mask_structure_stats(
            record,
            spec.source_type,
            spec.name,
            ratio,
            mask,
            region_count,
            variant_or_input,
        )
        comp.update(score_correlations(record, score))
        coord.update(score_correlations(record, score))
        stat_rows.append(comp)

    only_metrics, _ = eval_probe(original_probe, np.stack(x_eval_only, axis=0).astype(np.float32), args.y_eval)
    delete_metrics, _ = eval_probe(original_probe, np.stack(x_eval_delete, axis=0).astype(np.float32), args.y_eval)
    stats = aggregate_stats(stat_rows)
    visual = visual_summary.get(spec.visual_selector_name, {})

    row = {
        "selector_name": spec.name,
        "source_type": spec.source_type,
        "retention_ratio": float(ratio),
        "original_macro_f1": original_metrics["macro_f1"],
        "only_selected_macro_f1": only_metrics["macro_f1"],
        "delete_selected_macro_f1": delete_metrics["macro_f1"],
        "deletion_drop": float(original_metrics["macro_f1"]) - float(delete_metrics["macro_f1"]),
        "components": stats["connected_components"],
        "long_contour": stats["selected_long_contour_ratio"],
        "border_ratio": stats["border_ratio"],
        "center_ratio": stats["center_ratio"],
        "visual_partial_pass_rate": visual.get("visual_partial_pass_rate", float("nan")),
        "avg_facial_evidence_like": visual.get("avg_facial_evidence_like", float("nan")),
        "avg_hair_glasses": visual.get("avg_hair_glasses", float("nan")),
        "avg_border_background": visual.get("avg_border_background", float("nan")),
    }
    for eval_mode, metrics in (
        ("original", original_metrics),
        ("only_selected", only_metrics),
        ("delete_selected", delete_metrics),
    ):
        for class_name_value in EMOTION_NAMES:
            per_class_rows.append(
                {
                    "selector_name": spec.name,
                    "source_type": spec.source_type,
                    "retention_ratio": float(ratio),
                    "eval_mode": eval_mode,
                    "class_name": class_name_value,
                    "f1": metrics[f"per_class_f1_{class_name_value}"],
                }
            )
    return row, per_class_rows


def add_control_gaps(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {(row["selector_name"], float(row["retention_ratio"])): row for row in rows}
    out: List[Dict[str, Any]] = []
    for row in rows:
        ratio = float(row["retention_ratio"])
        random_row = lookup[("random_pixel", ratio)]
        center_row = lookup[("center_prior", ratio)]
        slic_row = lookup[("slic_region_proposal", ratio)]
        out.append(
            {
                **row,
                "gap_vs_random": float(row["only_selected_macro_f1"]) - float(random_row["only_selected_macro_f1"]),
                "gap_vs_center": float(row["only_selected_macro_f1"]) - float(center_row["only_selected_macro_f1"]),
                "gap_vs_slic": float(row["only_selected_macro_f1"]) - float(slic_row["only_selected_macro_f1"]),
            }
        )
    return out


def gate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {(row["selector_name"], float(row["retention_ratio"])): row for row in rows}
    out: List[Dict[str, Any]] = []
    for spec in SHORTLIST:
        row = by_key[(spec.name, 0.20)]
        deletion_ok = float(row["deletion_drop"]) >= 0.02
        random_ok = float(row["gap_vs_random"]) >= 0.02
        components_ok = float(row["components"]) < 18.0
        long_ok = float(row["long_contour"]) < 0.20
        visual_rate_ok = float(row["visual_partial_pass_rate"]) >= 0.60
        facial_ok = float(row["avg_facial_evidence_like"]) >= 1.20
        center_not_collapsed = float(row["center_ratio"]) < 0.35
        all_ok = all(
            [
                deletion_ok,
                random_ok,
                components_ok,
                long_ok,
                visual_rate_ok,
                facial_ok,
                center_not_collapsed,
            ]
        )
        out.append(
            {
                "selector_name": spec.name,
                "deletion_drop_20": row["deletion_drop"],
                "gap_vs_random_20": row["gap_vs_random"],
                "components_20": row["components"],
                "long_contour_20": row["long_contour"],
                "center_ratio_20": row["center_ratio"],
                "visual_partial_pass_rate": row["visual_partial_pass_rate"],
                "avg_facial_evidence_like": row["avg_facial_evidence_like"],
                "gate_deletion_drop": int(deletion_ok),
                "gate_gap_vs_random": int(random_ok),
                "gate_components": int(components_ok),
                "gate_long_contour": int(long_ok),
                "gate_visual_partial_pass_rate": int(visual_rate_ok),
                "gate_avg_facial_evidence_like": int(facial_ok),
                "gate_center_not_collapsed": int(center_not_collapsed),
                "stage5_ready_diagnostic_source": int(all_ok),
            }
        )
    return out


def decide(gates: Sequence[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    passed = [row for row in gates if int(row["stage5_ready_diagnostic_source"]) == 1]
    if passed:
        best = max(passed, key=lambda row: (float(row["deletion_drop_20"]), float(row["gap_vs_random_20"])))
        return "OPEN_STAGE5_DIAGNOSTIC_FOR_ONE_SELECTOR", str(best["selector_name"])
    promising = [
        row
        for row in gates
        if int(row["gate_visual_partial_pass_rate"]) == 1 or int(row["gate_avg_facial_evidence_like"]) == 1
    ]
    if promising:
        return "KEEP_STAGE5_LOCKED_REFINE_HEURISTIC", None
    return "STOP_STAGE5_PATH_DOCUMENT_FINDINGS", None


def write_report(output_dir: Path, rows: Sequence[Dict[str, Any]], gates: Sequence[Dict[str, Any]], decision: str, chosen: Optional[str]) -> None:
    rows20 = [row for row in rows if float(row["retention_ratio"]) == 0.20 and row["source_type"] != "control"]
    rows20 = sorted(rows20, key=lambda row: float(row["deletion_drop"]), reverse=True)
    gate_lookup = {row["selector_name"]: row for row in gates}
    lines: List[str] = [
        "# Stage 4.7 Fixed-original Deletion Check Report",
        "",
        "## 1. Scope",
        "",
        "- Protocol: `fixed_original_probe` only.",
        "- Probe training data: original images only.",
        "- Evaluation views: original / only_selected / delete_selected.",
        "- Fill mode: `local_mean`.",
        "- Ratios: `0.10`, `0.20`.",
        "- No new selector training, no motif, no SupCon, no Stage 5 implementation.",
        "",
        "## 2. Executive Decision",
        "",
        f"- Decision: `{decision}`.",
    ]
    if chosen:
        lines.append(f"- Diagnostic source selected: `{chosen}`.")
    lines.extend(
        [
            "",
            "## 3. Candidate Results @20",
            "",
            "| Selector | Only F1 | Delete F1 | Drop | Gap random | Gap center | Gap SLIC | Components | Long | Border | Center | Visual partial/pass | Facial evidence |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows20:
        lines.append(
            f"| `{row['selector_name']}` | {fmt(row['only_selected_macro_f1'])} | {fmt(row['delete_selected_macro_f1'])} | "
            f"{fmt(row['deletion_drop'])} | {fmt(row['gap_vs_random'])} | {fmt(row['gap_vs_center'])} | {fmt(row['gap_vs_slic'])} | "
            f"{fmt(row['components'])} | {fmt(row['long_contour'])} | {fmt(row['border_ratio'])} | {fmt(row['center_ratio'])} | "
            f"{fmt(row['visual_partial_pass_rate'])} | {fmt(row['avg_facial_evidence_like'])} |"
        )
    lines.extend(
        [
            "",
            "## 4. Gate Table",
            "",
            "| Selector | Drop>=.02 | Gap random>=.02 | Components<18 | Long<.20 | Visual>=.60 | Facial>=1.2 | Center ok | Stage5-ready diagnostic |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for spec in SHORTLIST:
        row = gate_lookup[spec.name]
        lines.append(
            f"| `{spec.name}` | {row['gate_deletion_drop']} | {row['gate_gap_vs_random']} | {row['gate_components']} | "
            f"{row['gate_long_contour']} | {row['gate_visual_partial_pass_rate']} | {row['gate_avg_facial_evidence_like']} | "
            f"{row['gate_center_not_collapsed']} | {row['stage5_ready_diagnostic_source']} |"
        )
    lines.extend(
        [
            "",
            "## 5. Interpretation",
            "",
            "- `deletion_drop` is evidence under the fixed-original probe only; it is not a free-standing causal claim.",
            "- Visual audit scores are carried in from Stage 4.6 and used as gating context, not as proof of semantic parts.",
            "- A selector can retain emotion signal yet still fail the diagnostic gate if it remains fragmented, contour-heavy, or visually mixed.",
            "",
            "## 6. Files",
            "",
            "- `stage47_deletion_metrics.csv`",
            "- `stage47_vs_controls.csv`",
            "- `stage47_gate_table.csv`",
            "- `stage47_per_class_metrics.csv`",
            "",
            "## 7. What Not To Claim",
            "",
            "- Không motif.",
            "- Không semantic part.",
            "- Không causal quá mức.",
            "- Không mở Stage 5 nếu gate chưa đạt.",
        ]
    )
    (output_dir / "stage47_fixed_original_deletion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_optional_int(value: str) -> Optional[int]:
    if value.lower() in {"none", "full", "all"}:
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph_repo", default="artifacts/graph-repo/graph_repo")
    parser.add_argument("--visual_summary", default="outputs/stage4_6_visual_audit/visual_audit_summary_by_selector_filled.csv")
    parser.add_argument("--visual_sheet", default="outputs/stage4_6_visual_audit/visual_audit_sheet_filled.csv")
    parser.add_argument("--vision_completion_report", default="outputs/stage4_6_visual_audit/stage46_vision_audit_completion_report.md")
    parser.add_argument("--stage4v2_checkpoint_dir", default="outputs/stage4_learned_evidence_selector/refinement_v2/checkpoints")
    parser.add_argument("--output_dir", default="outputs/stage4_7_fixed_original_deletion_shortlist")
    parser.add_argument("--max_train_samples", type=parse_optional_int, default=None)
    parser.add_argument("--max_val_samples", type=parse_optional_int, default=None)
    parser.add_argument("--probe_train_cap", type=parse_optional_int, default=None)
    parser.add_argument("--probe_eval_cap", type=parse_optional_int, default=None)
    parser.add_argument("--ratios", nargs="+", type=float, default=list(RATIOS))
    parser.add_argument("--slic_segments", type=int, default=64)
    parser.add_argument("--slic_compactness", type=float, default=0.10)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--classifier_max_iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    return parser


def maybe_cap(records: Sequence[Dict[str, Any]], cap: Optional[int]) -> List[Dict[str, Any]]:
    if cap is None:
        return list(records)
    return list(records[: min(len(records), int(cap))])


def require_inputs(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Stage 4.7 inputs:\n- " + "\n- ".join(missing))


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = ensure_dir(Path(args.output_dir))
    require_inputs(
        [
            Path(args.visual_sheet),
            Path(args.visual_summary),
            Path(args.vision_completion_report),
            Path(args.stage4v2_checkpoint_dir),
        ]
    )
    graph_repo = resolve_graph_repo(args.graph_repo)
    reader = GraphRepositoryReader(graph_repo)
    resolver = GraphResolver(reader.load_shared())
    slic_fn, slic_error = try_import_slic()
    if slic_fn is None:
        print(f"[Stage4.7] SLIC unavailable: {slic_error}")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"[Stage4.7] device={device}")
    print("[Stage4.7] loading records")
    train_records_full = load_split_records(reader, resolver, "train", args.max_train_samples)
    eval_records_full = load_split_records(reader, resolver, "val", args.max_val_samples)
    train_records = maybe_cap(train_records_full, args.probe_train_cap)
    eval_records = maybe_cap(eval_records_full, args.probe_eval_cap)
    x_train_original, y_train, _ = make_original_dataset(train_records)
    x_eval_original, y_eval, _ = make_original_dataset(eval_records)
    args.y_eval = y_eval
    print(f"[Stage4.7] train_records={len(train_records)} eval_records={len(eval_records)}")

    visual_summary = load_visual_summary(Path(args.visual_summary))
    models, input_variants = load_learned_models(Path(args.stage4v2_checkpoint_dir), device)
    original_probe = train_probe(x_train_original, y_train, args.seed, args.classifier_max_iter)
    original_metrics, _ = eval_probe(original_probe, x_eval_original, y_eval)

    metric_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    for spec in ALL_SOURCES:
        for ratio in [float(value) for value in args.ratios]:
            row, per_class = evaluate_source(
                spec,
                ratio,
                train_records,
                eval_records,
                original_probe,
                original_metrics,
                models,
                input_variants,
                visual_summary,
                slic_fn,
                args,
                device,
            )
            metric_rows.append(row)
            per_class_rows.extend(per_class)
            print(
                f"[Stage4.7] {spec.name} ratio={ratio:.2f} only_f1={fmt(row['only_selected_macro_f1'])} "
                f"delete_f1={fmt(row['delete_selected_macro_f1'])} drop={fmt(row['deletion_drop'])}"
            )

    vs_controls = add_control_gaps(metric_rows)
    gates = gate_rows(vs_controls)
    decision, chosen = decide(gates)

    write_csv(output_dir / "stage47_deletion_metrics.csv", metric_rows)
    write_csv(output_dir / "stage47_vs_controls.csv", vs_controls)
    write_csv(output_dir / "stage47_gate_table.csv", gates)
    write_csv(output_dir / "stage47_per_class_metrics.csv", per_class_rows)
    write_report(output_dir, vs_controls, gates, decision, chosen)

    print(f"[Stage4.7] output_dir={output_dir}")
    print(f"[Stage4.7] original_macro_f1={fmt(original_metrics['macro_f1'])}")
    print(f"[Stage4.7] decision={decision}")
    if chosen:
        print(f"[Stage4.7] chosen_selector={chosen}")


if __name__ == "__main__":
    main()
