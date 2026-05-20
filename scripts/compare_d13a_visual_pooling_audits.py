"""Compare two D13A visual pooling / assignment audit folders.

This script compares traceability and reliability audit artifacts only. It
does not make motif, semantic-region, or causal-evidence claims.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


K256_SCORE_CONTEXT = {
    "test_macro_f1": 0.5866,
    "test_acc": 0.6227,
    "assignment_entropy": 1.0,
}
K144_SCORE_CONTEXT = {
    "test_macro_f1": 0.5829,
    "test_acc": 0.6166,
    "assignment_entropy": 0.8589,
}


def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _find_one(audit_dir: Path, suffix: str) -> Optional[Path]:
    matches = sorted(audit_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _parse_decision(report_path: Optional[Path]) -> str:
    if report_path is None or not report_path.exists():
        return ""
    text = report_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"## Final Decision\s+([A-Z0-9_]+)", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if line.lower().startswith("## final decision") and idx + 1 < len(lines):
            return lines[idx + 1]
    return ""


def _dominant_area(area_df: pd.DataFrame) -> Tuple[str, float]:
    if area_df.empty or "count" not in area_df or "main_visible_area" not in area_df:
        return "", 0.0
    work = area_df.copy()
    work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(0)
    if work["count"].sum() <= 0:
        return "", 0.0
    row = work.sort_values("count", ascending=False).iloc[0]
    return str(row["main_visible_area"]), float(row["count"] / work["count"].sum())


def _risk_counts(sheet: pd.DataFrame) -> Dict[str, int]:
    if sheet.empty:
        return {
            "hair_glasses_risk_ge2": 0,
            "background_border_risk_ge2": 0,
            "center_shortcut_risk_ge2": 0,
            "softness_issue_ge2": 0,
            "assignment_interpretability_fail": 0,
            "traceability_fail": 0,
        }
    out = {}
    for col, key in [
        ("hair_glasses_risk", "hair_glasses_risk_ge2"),
        ("background_border_risk", "background_border_risk_ge2"),
        ("center_shortcut_risk", "center_shortcut_risk_ge2"),
        ("region_softness_issue", "softness_issue_ge2"),
    ]:
        out[key] = int((pd.to_numeric(sheet.get(col), errors="coerce") >= 2).sum()) if col in sheet else 0
    out["assignment_interpretability_fail"] = (
        int((pd.to_numeric(sheet.get("assignment_interpretability_score"), errors="coerce") == 0).sum())
        if "assignment_interpretability_score" in sheet
        else 0
    )
    out["traceability_fail"] = (
        int((pd.to_numeric(sheet.get("region_traceability_score"), errors="coerce") == 0).sum())
        if "region_traceability_score" in sheet
        else 0
    )
    return out


def _load_audit(audit_dir: Path, name: str) -> Dict[str, object]:
    summary_path = _find_one(audit_dir, "_visual_pooling_audit_summary.csv")
    area_path = _find_one(audit_dir, "_visual_pooling_area_summary.csv")
    risk_path = _find_one(audit_dir, "_visual_pooling_risk_cases.csv")
    report_path = _find_one(audit_dir, "_visual_pooling_audit_report.md")
    sheet_path = _first_existing(sorted(audit_dir.glob("*_visual_pooling_audit_sheet_filled.csv")))
    if sheet_path is None:
        sheet_path = _first_existing(sorted(audit_dir.glob("*_visual_pooling_audit_sheet.csv")))

    summary = _read_csv(summary_path)
    area = _read_csv(area_path)
    risk = _read_csv(risk_path)
    sheet = _read_csv(sheet_path)
    if summary.empty:
        row = {}
    else:
        row = summary.iloc[0].to_dict()
    dominant_area, dominant_rate = _dominant_area(area)
    decision = _parse_decision(report_path)
    row.update(
        {
            "run_name": name,
            "decision": decision,
            "dominant_area": dominant_area,
            "dominant_area_rate": dominant_rate,
            "risk_case_count": int(len(risk)),
            "audit_dir": str(audit_dir),
            "summary_path": str(summary_path) if summary_path else "",
            "sheet_path": str(sheet_path) if sheet_path else "",
        }
    )
    return {"row": row, "summary": summary, "area": area, "risk": risk, "sheet": sheet, "report_path": report_path}


def _paired_compare(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> pd.DataFrame:
    if a.empty or b.empty or "sample_id" not in a or "sample_id" not in b:
        return pd.DataFrame()
    cols = [
        "sample_id", "label", "visual_status", "assignment_interpretability_score",
        "region_softness_issue", "center_shortcut_risk", "region_traceability_score", "main_visible_area",
    ]
    aa = a[[c for c in cols if c in a.columns]].copy()
    bb = b[[c for c in cols if c in b.columns]].copy()
    merged = aa.merge(bb, on="sample_id", suffixes=(f"_{name_a}", f"_{name_b}"))
    if merged.empty:
        return merged

    rows = []
    for _, row in merged.iterrows():
        trace_a = pd.to_numeric(pd.Series([row.get(f"region_traceability_score_{name_a}")]), errors="coerce").iloc[0]
        trace_b = pd.to_numeric(pd.Series([row.get(f"region_traceability_score_{name_b}")]), errors="coerce").iloc[0]
        soft_a = pd.to_numeric(pd.Series([row.get(f"region_softness_issue_{name_a}")]), errors="coerce").iloc[0]
        soft_b = pd.to_numeric(pd.Series([row.get(f"region_softness_issue_{name_b}")]), errors="coerce").iloc[0]
        center_a = pd.to_numeric(pd.Series([row.get(f"center_shortcut_risk_{name_a}")]), errors="coerce").iloc[0]
        center_b = pd.to_numeric(pd.Series([row.get(f"center_shortcut_risk_{name_b}")]), errors="coerce").iloc[0]
        better = "tie"
        if pd.notna(soft_a) and pd.notna(soft_b) and pd.notna(center_a) and pd.notna(center_b):
            if soft_b < soft_a and center_b <= center_a and (pd.isna(trace_a) or pd.isna(trace_b) or trace_b >= trace_a):
                better = name_b
            elif soft_a < soft_b and center_a <= center_b and (pd.isna(trace_a) or pd.isna(trace_b) or trace_a >= trace_b):
                better = name_a
            elif center_b < center_a and soft_b <= soft_a:
                better = name_b
            elif center_a < center_b and soft_a <= soft_b:
                better = name_a
        rows.append(
            {
                "sample_id": row["sample_id"],
                "label": row.get(f"label_{name_a}", row.get(f"label_{name_b}", "")),
                f"{name_a}_visual_status": row.get(f"visual_status_{name_a}", ""),
                f"{name_b}_visual_status": row.get(f"visual_status_{name_b}", ""),
                f"{name_a}_assignment_interpretability": row.get(f"assignment_interpretability_score_{name_a}", ""),
                f"{name_b}_assignment_interpretability": row.get(f"assignment_interpretability_score_{name_b}", ""),
                f"{name_a}_softness_issue": row.get(f"region_softness_issue_{name_a}", ""),
                f"{name_b}_softness_issue": row.get(f"region_softness_issue_{name_b}", ""),
                f"{name_a}_center_shortcut": row.get(f"center_shortcut_risk_{name_a}", ""),
                f"{name_b}_center_shortcut": row.get(f"center_shortcut_risk_{name_b}", ""),
                f"{name_a}_main_area": row.get(f"main_visible_area_{name_a}", ""),
                f"{name_b}_main_area": row.get(f"main_visible_area_{name_b}", ""),
                "better_run": better,
                "notes": "Paired traceability comparison only; no motif or causal-evidence claim.",
            }
        )
    return pd.DataFrame(rows)


def _recommend(summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    by_name = {str(r["run_name"]): r for _, r in summary.iterrows()}
    k256 = next((r for n, r in by_name.items() if "k256" in n.lower()), None)
    k144 = next((r for n, r in by_name.items() if "k144" in n.lower() or "baseline" in n.lower()), None)
    if k144 is None or k256 is None:
        return "KEEP_D13B_LOCKED"

    k144_good = (
        float(k144.get("pass_partial_rate", 0) or 0) >= 0.60
        and float(k144.get("avg_assignment_interpretability_score", 0) or 0) >= 1.2
        and float(k144.get("avg_region_softness_issue", 9) or 9) < 1.2
        and float(k144.get("avg_center_shortcut_risk", 9) or 9) < 1.0
    )
    if k144_good:
        return "USE_K144_FOR_D13B_DIAGNOSTIC"

    k144_soft = float(k144.get("avg_region_softness_issue", 9) or 9)
    k256_soft = float(k256.get("avg_region_softness_issue", 9) or 9)
    k144_interp = float(k144.get("avg_assignment_interpretability_score", 0) or 0)
    k256_interp = float(k256.get("avg_assignment_interpretability_score", 0) or 0)
    k144_center = float(k144.get("avg_center_shortcut_risk", 9) or 9)
    k256_center = float(k256.get("avg_center_shortcut_risk", 9) or 9)
    if k144_soft < k256_soft or k144_center < k256_center:
        return "AUDIT_ANNEAL_BEFORE_D13B"
    if float(k256.get("pass_partial_rate", 0) or 0) >= 0.60 and k144_interp <= k256_interp + 0.1:
        return "USE_K256_FOR_D13B_DIAGNOSTIC_WITH_CAUTION"
    return "KEEP_D13B_LOCKED"


def _write_md_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if df.empty:
        return "No data."
    use = df.head(max_rows).copy() if max_rows else df.copy()
    for col in use.columns:
        if pd.api.types.is_numeric_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        else:
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare D13A visual pooling audit outputs")
    parser.add_argument("--audit_a", required=True)
    parser.add_argument("--name_a", required=True)
    parser.add_argument("--audit_b", required=True)
    parser.add_argument("--name_b", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    a = _load_audit(Path(args.audit_a), args.name_a)
    b = _load_audit(Path(args.audit_b), args.name_b)

    summary_cols = [
        "run_name", "decision", "total_samples", "reviewed_samples", "pass_count", "partial_count",
        "fail_count", "pass_partial_rate", "avg_face_coverage_score", "avg_region_traceability_score",
        "avg_assignment_interpretability_score", "avg_region_softness_issue", "avg_hair_glasses_risk",
        "avg_background_border_risk", "avg_center_shortcut_risk", "dominant_area", "dominant_area_rate",
        "risk_case_count",
    ]
    summary = pd.DataFrame([a["row"], b["row"]])
    summary = summary[[c for c in summary_cols if c in summary.columns]]
    summary.to_csv(output_dir / "d13a_visual_pooling_compare_summary.csv", index=False)

    area = pd.concat(
        [
            a["area"].assign(run_name=args.name_a) if not a["area"].empty else pd.DataFrame(),
            b["area"].assign(run_name=args.name_b) if not b["area"].empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    area.to_csv(output_dir / "d13a_visual_pooling_compare_area.csv", index=False)

    risk_rows = []
    for name, audit in [(args.name_a, a), (args.name_b, b)]:
        row = {"run_name": name, **_risk_counts(audit["sheet"])}
        risk_rows.append(row)
    risk_compare = pd.DataFrame(risk_rows)
    risk_compare.to_csv(output_dir / "d13a_visual_pooling_compare_risks.csv", index=False)

    paired = _paired_compare(a["sheet"], b["sheet"], args.name_a, args.name_b)
    paired_path = output_dir / "d13a_visual_pooling_paired_sample_compare.csv"
    if not paired.empty:
        paired.to_csv(paired_path, index=False)

    recommendation = _recommend(summary, paired)
    report = [
        "# D13A Visual Pooling Audit Comparison: K256 vs K144",
        "",
        "## 1. Context",
        "- K256 has the strongest D13A score but soft assignment.",
        "- K144 is the fair baseline reference with lower entropy.",
        "- Goal: choose a D13A base for D13B diagnostic, not claim motif.",
        "- Region nodes are soft learnable bottleneck nodes, not semantic regions.",
        "",
        "## 2. Score Context",
        f"- K256: test_macro_f1 = {K256_SCORE_CONTEXT['test_macro_f1']:.4f}; test_acc = {K256_SCORE_CONTEXT['test_acc']:.4f}; assignment_entropy ~= {K256_SCORE_CONTEXT['assignment_entropy']:.4f}.",
        f"- K144: test_macro_f1 = {K144_SCORE_CONTEXT['test_macro_f1']:.4f}; test_acc = {K144_SCORE_CONTEXT['test_acc']:.4f}; assignment_entropy ~= {K144_SCORE_CONTEXT['assignment_entropy']:.4f}.",
        "- Score gap K256 vs K144 is small, around +0.0037 macro-F1, so visual reliability matters.",
        "",
        "## 3. Audit Summary Table",
        _write_md_table(summary),
        "",
        "## 4. Area Distribution",
        _write_md_table(area, max_rows=30),
        "",
        "## 5. Risk Analysis",
        _write_md_table(risk_compare),
        "",
        "## 6. Paired Sample Comparison",
        f"Paired sample comparison rows: {len(paired)}.",
        "Written to `d13a_visual_pooling_paired_sample_compare.csv`." if not paired.empty else "Sample sets do not overlap enough for paired comparison.",
        "",
        "## 7. Decision Rules",
        "- Use K144 if it clears assignment interpretability, softness, center-shortcut, and pass-partial gates.",
        "- Audit anneal before D13B if K144 only partly improves interpretability or both runs remain too soft.",
        "- Use K256 only as a D13B diagnostic base if K144 does not improve visual reliability clearly.",
        "- Never output OPEN_D13B_FULL from this comparison.",
        "",
        "## 8. Final Recommendation",
        recommendation,
        "",
        "No motif, semantic-region, or causal-evidence claim is made.",
        "",
    ]
    (output_dir / "d13a_visual_pooling_compare_report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"recommendation={recommendation}")


if __name__ == "__main__":
    main()
