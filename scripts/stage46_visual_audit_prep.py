#!/usr/bin/env python
"""Prepare Stage 4.6 visual audit sheets for heuristic selector review.

This script does not train, modify graph artifacts, or open Stage 5. It only
indexes existing figure files, creates an audit sheet template, and writes a
report describing the manual review workflow.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


STAGE3_PRIORITY_VARIANTS = [
    "hybrid_pixel_score__E_balanced__b0p0",
    "hybrid_pixel_smooth__C_delta_grad__b0p1",
    "hybrid_slic_region__E_balanced__b0p1",
    "hybrid_slic_region_center_control__B_grad__b0p1",
]

STAGE36_PRIORITY_ROOTS = [
    "structure_pixel_score",
    "structure_pixel_smooth",
    "structure_slic_region",
    "diagnostics",
    "comparisons",
]

STAGE36_PRIORITY_VARIANTS = [
    "structure_slic_region__B_balanced__s0p2_o0p1_sm0p2_l0p2_b0p1",
    "structure_pixel_smooth__A_delta_grad__s0p2_o0p1_sm0p2_l0p4_b0p1",
    "structure_pixel_score__C_contrast_assisted__s0p2_o0p1_sm0p2_l0p2_b0p1",
]

STAGE4_LEARNED_VARIANTS = [
    "pixel_mlp_no_xy_r0p1_baseline_rerun",
    "tiny_conv_struct_reg_r0p1_soft_teacher",
    "tiny_conv_struct_reg_r0p1_long_border_penalty_light",
]

REVIEW_PACK_GROUPS = {
    "hybrid_pixel_score__E_balanced__b0p0": "hybrid_pixel_score__E_balanced__b0p0",
    "hybrid_slic_region__E_balanced__b0p1": "hybrid_slic_region__E_balanced__b0p1",
    "hybrid_slic_region_center_control__B_grad__b0p1": "hybrid_slic_region_center_control__B_grad__b0p1",
    "structure_slic_region__B_balanced__s0p2_o0p1_sm0p2_l0p2_b0p1": "structure_slic_region__B_balanced",
    "structure_pixel_smooth__A_delta_grad__s0p2_o0p1_sm0p2_l0p4_b0p1": "structure_pixel_smooth__A_delta_grad",
    "pixel_mlp_no_xy_r0p1_baseline_rerun": "learned_auxiliary",
    "tiny_conv_struct_reg_r0p1_soft_teacher": "learned_auxiliary",
    "tiny_conv_struct_reg_r0p1_long_border_penalty_light": "learned_auxiliary",
}


MANIFEST_COLUMNS = [
    "audit_id",
    "source_stage",
    "selector_name",
    "variant_name",
    "class_name",
    "graph_id",
    "ratio",
    "figure_path",
    "figure_type",
    "priority",
    "reason_for_review",
    "metric_high_center_hint",
    "metric_high_border_hint",
    "metric_high_long_hint",
    "metric_high_fragmentation_hint",
    "metric_region_like_hint",
]

AUDIT_COLUMNS = MANIFEST_COLUMNS[:8] + [
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
    "metric_high_center_hint",
    "metric_high_border_hint",
    "metric_high_long_hint",
    "metric_high_fragmentation_hint",
    "metric_region_like_hint",
]

SUMMARY_SELECTOR_COLUMNS = [
    "selector_name",
    "reviewed_count",
    "pass_count",
    "partial_count",
    "fail_count",
    "pass_rate",
    "avg_eye_eyebrow",
    "avg_mouth_nasolabial",
    "avg_face_muscle_cheek_wrinkle",
    "avg_hair_glasses",
    "avg_border_background",
    "avg_long_contour",
    "avg_center_collapse",
    "avg_fragmentation",
    "avg_region_like",
    "avg_facial_evidence_like",
]

SUMMARY_CLASS_COLUMNS = ["class_name"] + SUMMARY_SELECTOR_COLUMNS[1:]

RISK_CASE_COLUMNS = AUDIT_COLUMNS


def safe_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def ratio_from_token(token: str) -> str:
    m = re.search(r"ratio[_-](\d+)", token)
    if not m:
        return "UNKNOWN"
    raw = m.group(1)
    if len(raw) == 1:
        return f"0.{raw}"
    return f"{int(raw) / 100:.2f}".rstrip("0").rstrip(".")


def parse_class(parts: Iterable[str]) -> str:
    for part in parts:
        m = re.match(r"class_\d+_(.+)", part)
        if m:
            return m.group(1)
    return "UNKNOWN"


def parse_graph_id(name: str) -> str:
    m = re.search(r"graph_(\d+)", name)
    if m:
        return m.group(1)
    m = re.search(r"sample[_-]?(\d+)", name, re.IGNORECASE)
    if m:
        return m.group(1)
    return "UNKNOWN"


def figure_type(name: str, parts: Iterable[str]) -> str:
    text = "/".join(list(parts) + [name]).lower()
    if "comparison" in text:
        return "comparison"
    if "overlay" in text:
        return "overlay"
    if "mask" in text:
        return "mask"
    if "diagnostic" in text:
        return "diagnostic"
    if "structure_grid" in text or "grid" in text:
        return "diagnostic"
    return "unknown"


def load_metric_hints(stage45_dir: Path) -> Dict[str, Dict[str, str]]:
    shortlist = stage45_dir / "stage45_selector_shortlist.csv"
    hints: Dict[str, Dict[str, str]] = {}
    if not shortlist.exists():
        return hints
    with shortlist.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            selector = row.get("selector_name", "")
            if not selector:
                continue
            center = safe_float(row.get("center"))
            border = safe_float(row.get("border"))
            long = safe_float(row.get("long_contour"))
            components = safe_float(row.get("components"))
            hints[selector] = {
                "metric_high_center_hint": str(center is not None and center > 0.32),
                "metric_high_border_hint": str(border is not None and border > 0.34),
                "metric_high_long_hint": str(long is not None and long > 0.20),
                "metric_high_fragmentation_hint": str(components is not None and components > 18),
                "metric_region_like_hint": str(
                    components is not None and long is not None and components < 8 and long < 0.20
                ),
            }
    aliases = {
        "pixel_mlp_no_xy_r0p1_baseline_rerun": "pixel_mlp__no_xy_basic__main_hybrid__r0p1",
        "tiny_conv_struct_reg_r0p1_soft_teacher": "tiny_conv_struct_reg_r0p1_soft_teacher",
        "tiny_conv_struct_reg_r0p1_long_border_penalty_light": "tiny_conv_struct_reg_r0p1_long_border_penalty_light",
    }
    for alias, canonical in aliases.items():
        if canonical in hints and alias not in hints:
            hints[alias] = hints[canonical]
    return hints


def metric_hint_for(hints: Dict[str, Dict[str, str]], selector: str, variant: str) -> Dict[str, str]:
    default = {
        "metric_high_center_hint": "False",
        "metric_high_border_hint": "False",
        "metric_high_long_hint": "False",
        "metric_high_fragmentation_hint": "False",
        "metric_region_like_hint": "False",
    }
    return dict(hints.get(variant) or hints.get(selector) or default)


def add_manifest_row(
    rows: List[Dict[str, str]],
    path: Path,
    root: Path,
    source_stage: str,
    selector_name: str,
    variant_name: str,
    priority: str,
    reason: str,
    hints: Dict[str, Dict[str, str]],
) -> None:
    rel_parts = path.relative_to(root).parts
    ratio = "UNKNOWN"
    for part in rel_parts:
        parsed = ratio_from_token(part)
        if parsed != "UNKNOWN":
            ratio = parsed
            break
    row = {
        "audit_id": f"AUDIT_{len(rows)+1:06d}",
        "source_stage": source_stage,
        "selector_name": selector_name,
        "variant_name": variant_name,
        "class_name": parse_class(rel_parts),
        "graph_id": parse_graph_id(path.name),
        "ratio": ratio,
        "figure_path": str(path),
        "figure_type": figure_type(path.name, rel_parts),
        "priority": priority,
        "reason_for_review": reason,
    }
    row.update(metric_hint_for(hints, selector_name, variant_name))
    rows.append(row)


def scan_stage3(stage3_figures: Path, hints: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for variant in STAGE3_PRIORITY_VARIANTS:
        root = stage3_figures / variant
        if not root.exists():
            continue
        reason = "main heuristic suite; high priority visual audit"
        if "slic" in variant:
            reason = "region/continuity heuristic; check center and region quality"
        for path in sorted(root.rglob("*.png")):
            add_manifest_row(rows, path, stage3_figures, "Stage3", variant, variant, "high", reason, hints)
    return rows


def scan_stage36(stage36_figures: Path, hints: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for folder in STAGE36_PRIORITY_ROOTS:
        root = stage36_figures / folder
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.png")):
            rel = path.relative_to(root).parts
            variant = rel[0] if rel else folder
            if folder in {"structure_pixel_score", "structure_pixel_smooth", "structure_slic_region"}:
                if variant not in STAGE36_PRIORITY_VARIANTS:
                    continue
                priority = "high"
                reason = "structure-aware priority variant; review facial evidence vs center/contour risk"
            elif folder == "diagnostics":
                priority = "medium"
                reason = "diagnostic control/selector figure; use for risk comparison"
            else:
                priority = "medium"
                reason = "comparison figure from Stage 3.6"
            add_manifest_row(rows, path, stage36_figures, "Stage3.6", folder, variant, priority, reason, hints)
    return rows


def scan_stage4(stage4_figures: Path, hints: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for variant in STAGE4_LEARNED_VARIANTS:
        root = stage4_figures / variant
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.png")):
            add_manifest_row(
                rows,
                path,
                stage4_figures,
                "Stage4-v2",
                variant,
                variant,
                "high",
                "learned auxiliary comparator only; not main path",
                hints,
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def audit_rows_from_manifest(manifest: List[Dict[str, str]]) -> List[Dict[str, str]]:
    audit_rows = []
    for row in manifest:
        out = {col: row.get(col, "") for col in AUDIT_COLUMNS}
        for field in [
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
        ]:
            out[field] = ""
        out["overall_visual_pass"] = "UNREVIEWED"
        out["notes"] = ""
        audit_rows.append(out)
    return audit_rows


def is_reviewed(row: Dict[str, str]) -> bool:
    return row.get("overall_visual_pass", "UNREVIEWED") in {"PASS", "PARTIAL", "FAIL"}


def mean_or_blank(values: List[float]) -> str:
    if not values:
        return ""
    return f"{sum(values)/len(values):.6f}"


def summarize(audit_rows: List[Dict[str, str]], key: str, columns: List[str]) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in audit_rows:
        grouped[row.get(key, "UNKNOWN") or "UNKNOWN"].append(row)
    out = []
    score_fields = [
        ("avg_eye_eyebrow", "selected_eye_eyebrow"),
        ("avg_mouth_nasolabial", "selected_mouth_nasolabial"),
        ("avg_face_muscle_cheek_wrinkle", "selected_face_muscle_cheek_wrinkle"),
        ("avg_hair_glasses", "selected_hair_glasses"),
        ("avg_border_background", "selected_border_background"),
        ("avg_long_contour", "long_contour_dominant"),
        ("avg_center_collapse", "center_collapse"),
        ("avg_fragmentation", "fragmented_pixel_dust"),
        ("avg_region_like", "region_like"),
        ("avg_facial_evidence_like", "facial_evidence_like"),
    ]
    for group, rows in sorted(grouped.items()):
        reviewed = [r for r in rows if is_reviewed(r)]
        counts = Counter(r.get("overall_visual_pass") for r in reviewed)
        row = {
            columns[0]: group,
            "reviewed_count": len(reviewed),
            "pass_count": counts.get("PASS", 0),
            "partial_count": counts.get("PARTIAL", 0),
            "fail_count": counts.get("FAIL", 0),
            "pass_rate": f"{counts.get('PASS', 0) / len(reviewed):.6f}" if reviewed else "",
        }
        for out_name, field in score_fields:
            vals = [safe_float(r.get(field)) for r in reviewed]
            row[out_name] = mean_or_blank([v for v in vals if v is not None])
        out.append(row)
    return out


def risk_cases(audit_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cases = []
    for row in audit_rows:
        if not is_reviewed(row):
            continue
        risk = (
            (safe_float(row.get("selected_hair_glasses")) or 0) >= 2
            or (safe_float(row.get("selected_border_background")) or 0) >= 2
            or (safe_float(row.get("long_contour_dominant")) or 0) >= 2
            or (safe_float(row.get("center_collapse")) or 0) >= 2
            or (safe_float(row.get("fragmented_pixel_dust")) or 0) >= 2
            or row.get("overall_visual_pass") == "FAIL"
        )
        if risk:
            cases.append(row)
    return cases


def choose_review_pack_rows(
    manifest: List[Dict[str, str]],
    max_random_per_class: int,
    max_high_risk_per_class: int,
) -> List[Dict[str, str]]:
    grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    high_risk_grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in manifest:
        selector = row["variant_name"] if row["variant_name"] in REVIEW_PACK_GROUPS else row["selector_name"]
        if selector not in REVIEW_PACK_GROUPS:
            continue
        key = (selector, row["class_name"])
        grouped[key].append(row)
        high_risk = (
            row.get("metric_high_center_hint") == "True"
            or row.get("metric_high_border_hint") == "True"
            or row.get("metric_high_long_hint") == "True"
            or row.get("metric_high_fragmentation_hint") == "True"
        )
        if high_risk:
            high_risk_grouped[key].append(row)

    rng = random.Random(46)
    selected: Dict[str, Dict[str, str]] = {}
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["figure_path"])
        risk_rows = sorted(high_risk_grouped.get(key, []), key=lambda r: r["figure_path"])
        for row in risk_rows[:max_high_risk_per_class]:
            selected[row["audit_id"]] = row
        rest = [r for r in rows if r["audit_id"] not in selected]
        if len(rest) > max_random_per_class:
            rest = rng.sample(rest, max_random_per_class)
        for row in rest:
            selected[row["audit_id"]] = row
    return sorted(selected.values(), key=lambda r: (r["variant_name"], r["class_name"], r["figure_path"]))


def prepare_review_pack(rows: List[Dict[str, str]], output_dir: Path, copy_mode: str) -> Path:
    pack_dir = output_dir / "review_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    index_lines = [
        "# Stage 4.6 Review Pack Index",
        "",
        f"Copy mode: `{copy_mode}`.",
        "",
    ]
    for row in rows:
        selector = row["variant_name"] if row["variant_name"] in REVIEW_PACK_GROUPS else row["selector_name"]
        folder = REVIEW_PACK_GROUPS.get(selector, "other")
        source = Path(row["figure_path"])
        rel_name = f"{row['audit_id']}__{row['class_name']}__ratio_{row['ratio']}__graph_{row['graph_id']}__{source.name}"
        dest_dir = pack_dir / folder / row["class_name"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / rel_name
        if copy_mode == "copy":
            shutil.copy2(source, dest)
            index_target = dest
        elif copy_mode == "symlink":
            try:
                if not dest.exists():
                    os.symlink(source.resolve(), dest)
                index_target = dest
            except OSError:
                index_target = source
        else:
            index_target = source
        index_lines.append(f"- `{row['audit_id']}` `{selector}` `{row['class_name']}` ratio `{row['ratio']}`: `{index_target}`")
    (pack_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return pack_dir


def counts_by(rows: List[Dict[str, str]], key: str) -> Counter:
    return Counter(row.get(key, "UNKNOWN") or "UNKNOWN" for row in rows)


def write_report(
    output_dir: Path,
    manifest: List[Dict[str, str]],
    audit_rows: List[Dict[str, str]],
    pack_dir: Path,
    missing: List[str],
) -> None:
    reviewed = [r for r in audit_rows if is_reviewed(r)]
    by_stage = counts_by(manifest, "source_stage")
    by_selector = counts_by(manifest, "selector_name")
    by_class = counts_by(manifest, "class_name")
    priority = counts_by(manifest, "priority")

    def top_counts(counter: Counter, n: int = 20) -> str:
        if not counter:
            return "- none\n"
        return "\n".join(f"- `{k}`: `{v}`" for k, v in counter.most_common(n)) + "\n"

    missing_text = "\n".join(f"- `{m}`" for m in missing) if missing else "- none"
    report = f"""# Stage 4.6 Visual Audit Report

## 1. Executive Summary

- Đây là bước visual audit, không train.
- Stage 5 vẫn khóa.
- Manifest rows: `{len(manifest)}`.
- Review sheet rows: `{len(audit_rows)}`.
- Manual review completed: `{'yes' if reviewed else 'no'}`.
- Review pack path: `{pack_dir}`.
- Missing priority folders/files: 
{missing_text}

## 2. Why Visual Audit Is Required

F1/retention không đủ để biết mask chọn facial evidence hay shortcut. Stage 4.5 đã chốt missing evidence chính là manual visual audit và fixed-original deletion cho heuristic suite. Visual audit giúp phân biệt mask chọn mắt/miệng/lông mày với mask chọn tóc/kính/viền/nền hoặc center shortcut.

## 3. Reviewed Selector Suite

Main information heuristic:

- `hybrid_pixel_score__E_balanced__b0p0`

Region/continuity heuristic:

- `hybrid_slic_region__E_balanced__b0p1`
- `hybrid_slic_region_center_control__B_grad__b0p1`

Structure prior:

- `structure_slic_region__B_balanced__s0p2_o0p1_sm0p2_l0p2_b0p1`
- `structure_pixel_smooth__A_delta_grad__s0p2_o0p1_sm0p2_l0p4_b0p1`
- `structure_pixel_score__C_contrast_assisted__s0p2_o0p1_sm0p2_l0p2_b0p1`

Learned auxiliary comparator only:

- `pixel_mlp_no_xy_r0p1_baseline_rerun`
- `tiny_conv_struct_reg_r0p1_soft_teacher`
- `tiny_conv_struct_reg_r0p1_long_border_penalty_light`

## 4. Audit Checklist

Score rule: `0 = không thấy / không đáng kể`, `1 = có một phần / vừa phải`, `2 = rõ ràng / dominant`.

- `selected_eye_eyebrow`: mask có chọn mắt/lông mày không.
- `selected_mouth_nasolabial`: mask có chọn miệng/rãnh mũi má không.
- `selected_face_muscle_cheek_wrinkle`: mask có chọn má/nếp nhăn/cơ mặt không.
- `selected_hair_glasses`: mask có chọn tóc/kính rõ không.
- `selected_border_background`: mask có chọn viền ảnh/nền rõ không.
- `long_contour_dominant`: contour dài chiếm ưu thế không.
- `center_collapse`: mask có collapse vào vùng center không.
- `fragmented_pixel_dust`: mask có rời rạc kiểu bụi pixel không.
- `region_like`: mask có liền vùng/region-like không.
- `facial_evidence_like`: tổng thể có giống evidence biểu cảm trên mặt không.
- `overall_visual_pass`: `PASS`, `PARTIAL`, `FAIL`, hoặc `UNREVIEWED`.

## 5. Manifest / Review Pack

Manifest by stage:

{top_counts(by_stage)}

Manifest by priority:

{top_counts(priority)}

Top selectors by figure count:

{top_counts(by_selector, 30)}

Classes:

{top_counts(by_class, 20)}

Review pack:

- `{pack_dir}`
- Default `copy_mode=index` chỉ tạo index đường dẫn, không copy ảnh.

## 6. Current Audit Status

Manual review status: `{'COMPLETED' if reviewed else 'UNREVIEWED'}`.

Nếu chưa review, các summary/pass-fail hiện chưa có ý nghĩa kết luận thị giác. Các metric hints trong sheet chỉ là gợi ý từ CSV, không phải visual conclusion.

Files:

- `visual_audit_manifest.csv`
- `visual_audit_sheet.csv`
- `visual_audit_summary_by_selector.csv`
- `visual_audit_summary_by_class.csv`
- `visual_audit_risk_cases.csv`

## 7. Decision Rule After Manual Review

Heuristic selector được xem là visually acceptable nếu:

- `facial_evidence_like` trung bình cao.
- `selected_hair_glasses` và `selected_border_background` thấp.
- `long_contour_dominant` thấp.
- `center_collapse` thấp.
- SLIC/structure selector có `region_like` đủ tốt.
- Pass rate đủ tốt theo class, không chỉ Happy/Surprise.

Nếu không đạt:

- không chạy fixed-original deletion nữa;
- không Stage 5.

Nếu đạt:

- chạy analysis-only fixed-original + local_mean deletion cho heuristic suite.

## 8. Next Step

- Nếu audit sheet chưa điền: user/AI cần review `visual_audit_sheet.csv`.
- Nếu audit pass: triển khai fixed-original deletion measurement heuristic suite.
- Nếu audit fail: quay lại heuristic refinement hoặc stop.

## 9. What Not To Claim

- Không motif.
- Không semantic part.
- Không causal.
- Không Q1.
- Không Stage 5.
"""
    (output_dir / "stage46_visual_audit_report.md").write_text(report, encoding="utf-8")


def write_pass_fail_decision(output_dir: Path, audit_rows: List[Dict[str, str]]) -> None:
    reviewed = [r for r in audit_rows if is_reviewed(r)]
    if not reviewed:
        text = """# Stage 4.6 Visual Audit Pass/Fail Decision

Decision: `UNREVIEWED`.

Manual review has not been completed yet. No selector can be promoted based on this audit sheet.

Stage 5 status: `LOCKED`.

Next action: fill `visual_audit_sheet.csv` using the 0/1/2 checklist and set `overall_visual_pass` for each reviewed figure.
"""
    else:
        counts = Counter(r["overall_visual_pass"] for r in reviewed)
        text = f"""# Stage 4.6 Visual Audit Pass/Fail Decision

Decision: `REVIEW_COMPLETED_NEEDS_SUMMARY_INTERPRETATION`.

Reviewed rows: `{len(reviewed)}`.

- PASS: `{counts.get('PASS', 0)}`
- PARTIAL: `{counts.get('PARTIAL', 0)}`
- FAIL: `{counts.get('FAIL', 0)}`

Stage 5 status remains locked until fixed-original deletion and gate criteria are satisfied.
"""
    (output_dir / "visual_audit_pass_fail_decision.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stage 4.6 visual audit tables from existing figures.")
    parser.add_argument("--stage45_dir", default="outputs/stage4_5_hybrid_structure_freeze")
    parser.add_argument("--stage3_figures", default="outputs/stage3_hybrid_evidence_selector/figures")
    parser.add_argument("--stage36_figures", default="outputs/stage36_structure_aware_diagnostics/figures")
    parser.add_argument("--stage4_figures", default="outputs/stage4_learned_evidence_selector/refinement_v2/figures")
    parser.add_argument("--output_dir", default="outputs/stage4_6_visual_audit")
    parser.add_argument("--max_random_per_class", type=int, default=5)
    parser.add_argument("--max_high_risk_per_class", type=int, default=3)
    parser.add_argument("--copy_mode", choices=["copy", "symlink", "index"], default="index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage45_dir = Path(args.stage45_dir)
    stage3_figures = Path(args.stage3_figures)
    stage36_figures = Path(args.stage36_figures)
    stage4_figures = Path(args.stage4_figures)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing: List[str] = []
    for path in [
        stage45_dir / "stage45_hybrid_structure_freeze_report.md",
        stage45_dir / "stage45_next_action_plan.md",
        stage45_dir / "figures_index.md",
    ]:
        if not path.exists():
            missing.append(str(path))

    hints = load_metric_hints(stage45_dir)
    manifest: List[Dict[str, str]] = []
    manifest.extend(scan_stage3(stage3_figures, hints) if stage3_figures.exists() else [])
    if not stage3_figures.exists():
        missing.append(str(stage3_figures))
    manifest.extend(scan_stage36(stage36_figures, hints) if stage36_figures.exists() else [])
    if not stage36_figures.exists():
        missing.append(str(stage36_figures))
    manifest.extend(scan_stage4(stage4_figures, hints) if stage4_figures.exists() else [])
    if not stage4_figures.exists():
        missing.append(str(stage4_figures))

    # Reassign audit IDs after all scans so they are stable and contiguous.
    for i, row in enumerate(sorted(manifest, key=lambda r: (r["source_stage"], r["selector_name"], r["variant_name"], r["figure_path"])), 1):
        row["audit_id"] = f"AUDIT_{i:06d}"
    manifest = sorted(manifest, key=lambda r: r["audit_id"])

    audit_rows = audit_rows_from_manifest(manifest)
    pack_rows = choose_review_pack_rows(manifest, args.max_random_per_class, args.max_high_risk_per_class)
    pack_dir = prepare_review_pack(pack_rows, output_dir, args.copy_mode)

    write_csv(output_dir / "visual_audit_manifest.csv", manifest, MANIFEST_COLUMNS)
    write_csv(output_dir / "visual_audit_sheet.csv", audit_rows, AUDIT_COLUMNS)
    write_csv(output_dir / "visual_audit_summary_by_selector.csv", summarize(audit_rows, "selector_name", SUMMARY_SELECTOR_COLUMNS), SUMMARY_SELECTOR_COLUMNS)
    write_csv(output_dir / "visual_audit_summary_by_class.csv", summarize(audit_rows, "class_name", SUMMARY_CLASS_COLUMNS), SUMMARY_CLASS_COLUMNS)
    write_csv(output_dir / "visual_audit_risk_cases.csv", risk_cases(audit_rows), RISK_CASE_COLUMNS)
    write_report(output_dir, manifest, audit_rows, pack_dir, missing)
    write_pass_fail_decision(output_dir, audit_rows)

    priority_selectors = sorted({r["variant_name"] for r in manifest if r["priority"] == "high"})
    expected = set(STAGE3_PRIORITY_VARIANTS + STAGE36_PRIORITY_VARIANTS + STAGE4_LEARNED_VARIANTS)
    found = set(priority_selectors)
    missing_selectors = sorted(expected - found)
    print(f"[Stage4.6] manifest_rows={len(manifest)}")
    print(f"[Stage4.6] review_sheet_rows={len(audit_rows)}")
    print(f"[Stage4.6] priority_selectors_found={len(found)}")
    print(f"[Stage4.6] priority_selectors_missing={missing_selectors}")
    print(f"[Stage4.6] review_pack_path={pack_dir}")
    print("[Stage4.6] manual_audit_completed=no")
    print("[Stage4.6] stage5_status=locked")
    print("[Stage4.6] next_action=fill visual_audit_sheet.csv")


if __name__ == "__main__":
    main()
