#!/usr/bin/env python
"""Stage 4.6 full visual audit scan wrapper.

This script intentionally does not infer visual labels from metric hints. A real
full visual audit requires a multimodal reviewer or a manually filled sheet. If
the current environment cannot perform batch vision review and --require_vision
is true, the script writes a NO_VISION_CAPABILITY report and preserves all rows
as UNREVIEWED.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


VISUAL_SCORE_FIELDS = [
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
]

SUMMARY_SELECTOR_COLUMNS = [
    "selector_name",
    "source_stage",
    "reviewed_count",
    "pass_count",
    "partial_count",
    "fail_count",
    "unreviewed_count",
    "pass_rate",
    "partial_or_pass_rate",
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
    "main_failure_reason",
]

SUMMARY_CLASS_COLUMNS = [
    "class_name",
    "selector_name",
    "reviewed_count",
    "pass_rate",
    "avg_eye_eyebrow",
    "avg_mouth_nasolabial",
    "avg_hair_glasses",
    "avg_border_background",
    "avg_long_contour",
    "avg_center_collapse",
    "avg_fragmentation",
    "avg_facial_evidence_like",
]


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def safe_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def is_reviewed(row: Dict[str, str]) -> bool:
    return row.get("overall_visual_pass") in {"PASS", "PARTIAL", "FAIL"}


def mean_score(rows: List[Dict[str, str]], field: str) -> str:
    values = [safe_float(r.get(field)) for r in rows]
    values = [v for v in values if v is not None]
    if not values:
        return ""
    return f"{sum(values) / len(values):.6f}"


def failure_reason(rows: List[Dict[str, str]]) -> str:
    reviewed = [r for r in rows if is_reviewed(r)]
    if not reviewed:
        return "UNREVIEWED"
    risks = {
        "hair_glasses": mean_score(reviewed, "selected_hair_glasses"),
        "border_background": mean_score(reviewed, "selected_border_background"),
        "long_contour": mean_score(reviewed, "long_contour_dominant"),
        "center_collapse": mean_score(reviewed, "center_collapse"),
        "fragmentation": mean_score(reviewed, "fragmented_pixel_dust"),
        "low_facial_evidence": mean_score(reviewed, "facial_evidence_like"),
    }
    numeric = {k: safe_float(v) for k, v in risks.items()}
    if numeric["low_facial_evidence"] is not None and numeric["low_facial_evidence"] < 1.2:
        return "low_facial_evidence"
    high = [(k, v) for k, v in numeric.items() if v is not None and k != "low_facial_evidence"]
    if not high:
        return "none"
    high.sort(key=lambda kv: kv[1], reverse=True)
    return high[0][0]


def summarize_by_selector(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("selector_name", "UNKNOWN") or "UNKNOWN"].append(row)

    out: List[Dict[str, object]] = []
    for selector, group_rows in sorted(grouped.items()):
        reviewed = [r for r in group_rows if is_reviewed(r)]
        counts = Counter(r.get("overall_visual_pass", "UNREVIEWED") for r in group_rows)
        reviewed_count = len(reviewed)
        pass_count = counts.get("PASS", 0)
        partial_count = counts.get("PARTIAL", 0)
        fail_count = counts.get("FAIL", 0)
        stages = sorted({r.get("source_stage", "UNKNOWN") for r in group_rows})
        row = {
            "selector_name": selector,
            "source_stage": ";".join(stages),
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
            "main_failure_reason": failure_reason(group_rows),
        }
        out.append(row)
    return out


def summarize_by_class(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("class_name", "UNKNOWN") or "UNKNOWN", row.get("selector_name", "UNKNOWN") or "UNKNOWN")].append(row)

    out: List[Dict[str, object]] = []
    for (class_name, selector), group_rows in sorted(grouped.items()):
        reviewed = [r for r in group_rows if is_reviewed(r)]
        pass_count = sum(1 for r in reviewed if r.get("overall_visual_pass") == "PASS")
        row = {
            "class_name": class_name,
            "selector_name": selector,
            "reviewed_count": len(reviewed),
            "pass_rate": f"{pass_count / len(reviewed):.6f}" if reviewed else "",
            "avg_eye_eyebrow": mean_score(reviewed, "selected_eye_eyebrow"),
            "avg_mouth_nasolabial": mean_score(reviewed, "selected_mouth_nasolabial"),
            "avg_hair_glasses": mean_score(reviewed, "selected_hair_glasses"),
            "avg_border_background": mean_score(reviewed, "selected_border_background"),
            "avg_long_contour": mean_score(reviewed, "long_contour_dominant"),
            "avg_center_collapse": mean_score(reviewed, "center_collapse"),
            "avg_fragmentation": mean_score(reviewed, "fragmented_pixel_dust"),
            "avg_facial_evidence_like": mean_score(reviewed, "facial_evidence_like"),
        }
        out.append(row)
    return out


def collect_risk_cases(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        note = (row.get("notes") or "").lower()
        risk = (
            (safe_float(row.get("selected_hair_glasses")) or 0) >= 2
            or (safe_float(row.get("selected_border_background")) or 0) >= 2
            or (safe_float(row.get("long_contour_dominant")) or 0) >= 2
            or (safe_float(row.get("center_collapse")) or 0) >= 2
            or (safe_float(row.get("fragmented_pixel_dust")) or 0) >= 2
            or row.get("overall_visual_pass") == "FAIL"
            or any(token in note for token in ["unclear", "missing", "unreadable"])
        )
        if risk:
            out.append(row)
    return out


def normalize_sheet(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        normalized = dict(row)
        if not normalized.get("overall_visual_pass"):
            normalized["overall_visual_pass"] = "UNREVIEWED"
        figure_path = normalized.get("figure_path")
        if figure_path and not Path(figure_path).exists() and not is_reviewed(normalized):
            normalized["overall_visual_pass"] = "UNREVIEWED"
            existing_note = normalized.get("notes", "")
            normalized["notes"] = (existing_note + "; missing_file").strip("; ")
        out.append(normalized)
    return out


def selector_visual_decisions(summary_rows: List[Dict[str, object]]) -> List[Dict[str, str]]:
    decisions = []
    for row in summary_rows:
        reviewed = int(row.get("reviewed_count") or 0)
        if reviewed <= 0:
            decision = "VISUAL_UNREVIEWED"
            reason = "no reviewed rows"
        else:
            pass_rate = safe_float(row.get("pass_rate")) or 0
            partial_or_pass = safe_float(row.get("partial_or_pass_rate")) or 0
            fail_count = int(row.get("fail_count") or 0)
            fail_rate = fail_count / reviewed if reviewed else 1
            avg_facial = safe_float(row.get("avg_facial_evidence_like")) or 0
            avg_hair = safe_float(row.get("avg_hair_glasses")) or 0
            avg_border = safe_float(row.get("avg_border_background")) or 0
            avg_long = safe_float(row.get("avg_long_contour")) or 0
            avg_center = safe_float(row.get("avg_center_collapse")) or 0
            avg_frag = safe_float(row.get("avg_fragmentation")) or 0
            ok = (
                partial_or_pass >= 0.60
                and fail_rate <= 0.30
                and avg_facial >= 1.2
                and avg_hair <= 0.8
                and avg_border <= 0.8
                and avg_long <= 0.8
                and avg_center <= 0.8
                and avg_frag <= 1.0
            )
            decision = "VISUAL_PASS_FOR_DELETION_MEASUREMENT" if ok else "VISUAL_FAIL_OR_HOLD"
            reason = (
                f"partial_or_pass={partial_or_pass:.3f}, fail_rate={fail_rate:.3f}, "
                f"facial={avg_facial:.3f}, hair={avg_hair:.3f}, border={avg_border:.3f}, "
                f"long={avg_long:.3f}, center={avg_center:.3f}, fragmentation={avg_frag:.3f}"
            )
        decisions.append({"selector_name": str(row.get("selector_name", "")), "decision": decision, "reason": reason})
    return decisions


def write_pass_fail_decision(path: Path, summary_rows: List[Dict[str, object]], no_vision: bool) -> None:
    decisions = selector_visual_decisions(summary_rows)
    passed = [d for d in decisions if d["decision"] == "VISUAL_PASS_FOR_DELETION_MEASUREMENT"]
    if no_vision:
        body = [
            "# Stage 4.6 Visual Audit Pass/Fail Decision Filled",
            "",
            "Decision: `NO_VISION_CAPABILITY`.",
            "",
            "The current script/runtime cannot perform real multimodal visual review. No selector is visually promoted.",
            "",
            "Stage 5 status: `LOCKED`.",
        ]
    elif not passed:
        body = [
            "# Stage 4.6 Visual Audit Pass/Fail Decision Filled",
            "",
            "Decision: `VISUAL_AUDIT_UNREVIEWED_OR_NO_PASS`.",
            "",
            "No selector passed the visual criteria. Stage 5 remains locked.",
        ]
    else:
        body = [
            "# Stage 4.6 Visual Audit Pass/Fail Decision Filled",
            "",
            "Decision: `VISUAL_AUDIT_PASS_RUN_DELETION_MEASUREMENT`.",
            "",
            "Selectors eligible for next deletion measurement:",
            "",
        ]
        body.extend(f"- `{d['selector_name']}`: {d['reason']}" for d in passed)
        body.extend(["", "Stage 5 status: `LOCKED` until fixed-original deletion gates pass."])
    body.extend(["", "## Selector Decisions", ""])
    for d in decisions:
        body.append(f"- `{d['selector_name']}`: `{d['decision']}` ({d['reason']})")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    rows: List[Dict[str, str]],
    summary_rows: List[Dict[str, object]],
    risk_rows: List[Dict[str, str]],
    no_vision: bool,
) -> None:
    reviewed = [r for r in rows if is_reviewed(r)]
    unreviewed = len(rows) - len(reviewed)
    selectors = sorted({r.get("selector_name", "UNKNOWN") for r in rows})
    top = sorted(
        summary_rows,
        key=lambda r: (
            safe_float(r.get("partial_or_pass_rate")) or -1,
            safe_float(r.get("avg_facial_evidence_like")) or -1,
        ),
        reverse=True,
    )[:10]
    top_lines = "\n".join(
        f"| `{r['selector_name']}` | {r['reviewed_count']} | {r['partial_or_pass_rate']} | {r['avg_facial_evidence_like']} | {r['main_failure_reason']} |"
        for r in top
    )
    if not top_lines:
        top_lines = "| none | 0 |  |  |  |"
    status = "NO_VISION_CAPABILITY" if no_vision else ("yes" if reviewed else "no")
    report = f"""# Stage 4.6 Full Visual Audit Scan Report

## 1. Executive Summary

- Total rows: `{len(rows)}`.
- Reviewed rows: `{len(reviewed)}`.
- Unreviewed rows: `{unreviewed}`.
- Selectors reviewed/indexed: `{len(selectors)}`.
- Visual audit completed: `{status}`.
- Stage 5 status: `locked`.

## 2. Method

The intended scoring is `0/1/2` for each visual field and `PASS/PARTIAL/FAIL/UNREVIEWED` for overall visual pass. However, this run did **not** perform image-based scoring because the current coding-agent runtime does not provide a batch multimodal vision channel inside the script.

Metric hints were not used as visual conclusions. Existing rows are preserved as `UNREVIEWED` unless a human or separate multimodal reviewer fills the sheet later.

No motif, semantic part, causal, or Q1 claim is made.

## 3. Selector-level Results

| Selector | Reviewed | Partial/pass rate | Facial evidence avg | Main failure reason |
|---|---:|---:|---:|---|
{top_lines}

Because reviewed rows are currently zero, this table is a placeholder summary only.

## 4. Class-level Results

Class-level visual evidence is not available until `visual_audit_sheet_filled.csv` contains reviewed rows. Angry/Sad/Neutral and Happy/Surprise still need explicit manual or multimodal review.

## 5. Risk Cases

Risk cases rows: `{len(risk_rows)}`.

At this stage risk cases only include rows already marked by a reviewer or rows with missing/unreadable notes. Metric hints are not converted into visual risk cases.

## 6. Recommended Selector For Next Deletion Measurement

No selector is recommended from visual audit yet.

The planned candidates remain:

- Information-rich heuristic: `hybrid_pixel_score__E_balanced__b0p0`.
- Region/continuity heuristic: `hybrid_slic_region__E_balanced__b0p1`.
- Structure-aware heuristic: `structure_slic_region__B_balanced__s0p2_o0p1_sm0p2_l0p2_b0p1` or `structure_pixel_smooth__A_delta_grad__s0p2_o0p1_sm0p2_l0p4_b0p1`.

These require real visual review before fixed-original deletion measurement.

## 7. Decision

Decision: `NO_VISION_CAPABILITY`.

Do not run Stage 5. Do not run part grouping. The next action is to fill `visual_audit_sheet_filled.csv` using a human reviewer or a batch-capable multimodal model.

## 8. What Not To Claim

- Not motif.
- Not semantic part.
- Not causal.
- Not Q1.
- Stage 5 still locked.
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4.6 full visual audit scan summarizer.")
    parser.add_argument("--audit_dir", default="outputs/stage4_6_visual_audit")
    parser.add_argument("--sheet", default="outputs/stage4_6_visual_audit/visual_audit_sheet.csv")
    parser.add_argument("--manifest", default="outputs/stage4_6_visual_audit/visual_audit_manifest.csv")
    parser.add_argument("--output_sheet", default="outputs/stage4_6_visual_audit/visual_audit_sheet_filled.csv")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--resume", default="true")
    parser.add_argument("--require_vision", default="true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir)
    sheet_path = Path(args.sheet)
    output_sheet = Path(args.output_sheet)
    audit_dir.mkdir(parents=True, exist_ok=True)

    # This script has no embedded multimodal image understanding. Keep this
    # explicit so it cannot silently convert metric hints into visual labels.
    has_batch_vision = False
    require_vision = parse_bool(args.require_vision)
    no_vision = require_vision and not has_batch_vision

    source_sheet = output_sheet if parse_bool(args.resume) and output_sheet.exists() else sheet_path
    rows = normalize_sheet(read_csv(source_sheet))
    if not rows:
        rows = normalize_sheet(read_csv(sheet_path))

    columns = list(rows[0].keys()) if rows else []
    write_csv(output_sheet, rows, columns)

    selector_summary = summarize_by_selector(rows)
    class_summary = summarize_by_class(rows)
    risk_rows = collect_risk_cases(rows)

    write_csv(audit_dir / "visual_audit_summary_by_selector_filled.csv", selector_summary, SUMMARY_SELECTOR_COLUMNS)
    write_csv(audit_dir / "visual_audit_summary_by_class_filled.csv", class_summary, SUMMARY_CLASS_COLUMNS)
    write_csv(audit_dir / "visual_audit_risk_cases_filled.csv", risk_rows, columns)
    write_pass_fail_decision(audit_dir / "stage46_visual_audit_pass_fail_decision_filled.md", selector_summary, no_vision)
    write_report(audit_dir / "stage46_full_visual_audit_report.md", rows, selector_summary, risk_rows, no_vision)

    reviewed = [r for r in rows if is_reviewed(r)]
    ranked = sorted(
        selector_summary,
        key=lambda r: safe_float(r.get("partial_or_pass_rate")) or -1,
        reverse=True,
    )
    top3 = [r["selector_name"] for r in ranked[:3] if safe_float(r.get("partial_or_pass_rate")) is not None]
    risk_ranked = sorted(
        selector_summary,
        key=lambda r: safe_float(r.get("avg_hair_glasses")) or 0,
        reverse=True,
    )
    risk3 = [r["selector_name"] for r in risk_ranked[:3] if safe_float(r.get("avg_hair_glasses")) is not None]

    print(f"[Stage4.6Full] visual_audit_rows_reviewed={len(reviewed)}")
    print(f"[Stage4.6Full] total_rows={len(rows)}")
    print(f"[Stage4.6Full] no_vision_capability={no_vision}")
    print(f"[Stage4.6Full] top3_selectors_by_pass_partial_rate={top3}")
    print(f"[Stage4.6Full] top3_risk_selectors={risk3}")
    print("[Stage4.6Full] recommended_next_action=use human or multimodal batch reviewer to fill visual_audit_sheet_filled.csv")
    print("[Stage4.6Full] stage5=no")


if __name__ == "__main__":
    main()
