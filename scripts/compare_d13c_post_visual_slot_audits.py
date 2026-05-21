"""Compare post-D13C visual slot audit outputs across runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


SCORE_CONTEXT = {
    "d13b_m16_reference": {"macro_f1": 0.6187, "acc": 0.6328},
    "d13c_m16_ce_continue": {"macro_f1": 0.6222, "acc": 0.6358},
    "d13c_m16_supcon_l005": {"macro_f1": 0.6277, "acc": 0.6420},
    "d13c_m8_supcon_l002_control": {"macro_f1": 0.6364, "acc": 0.6481},
}


def _sanitize(value: str | None, fallback: str = "run") -> str:
    raw = str(value or fallback or "run").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    return safe.strip("_") or "run"


def _read_first(paths: List[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return pd.DataFrame()


def _find_summary(audit_dir: Path) -> pd.DataFrame:
    return _read_first(sorted(audit_dir.glob("*_post_visual_slot_audit_summary.csv")) + sorted(audit_dir.glob("*_audit_summary.csv")))


def _find_area(audit_dir: Path) -> pd.DataFrame:
    return _read_first(sorted(audit_dir.glob("*_post_visual_slot_area_summary.csv")) + sorted(audit_dir.glob("*_area_summary.csv")))


def _find_risk(audit_dir: Path) -> pd.DataFrame:
    return _read_first(sorted(audit_dir.glob("*_post_visual_slot_risk_cases.csv")) + sorted(audit_dir.glob("*_risk_cases.csv")))


def _find_sheet(audit_dir: Path) -> pd.DataFrame:
    candidates: List[Path] = []
    candidates.extend(sorted(audit_dir.glob("*_filled.csv")))
    candidates.extend(sorted(audit_dir.glob("*audit_sheet.csv")))
    return _read_first(candidates)


def _num(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _quality_score(row: pd.Series) -> float:
    good = [
        _num(row.get("slot_traceability_score"), 0.0),
        _num(row.get("slot_diversity_visual_score"), 0.0),
        _num(row.get("slot_face_coverage_score"), 0.0),
        _num(row.get("slot_assignment_readability_score"), 0.0),
        _num(row.get("multi_region_support"), 0.0),
    ]
    bad = [
        _num(row.get("mouth_only_risk"), 0.0),
        _num(row.get("center_shortcut_risk"), 0.0),
        _num(row.get("hair_glasses_risk"), 0.0),
        _num(row.get("background_border_risk"), 0.0),
        _num(row.get("slot_collapse_visual"), 0.0),
    ]
    status = str(row.get("visual_status", "")).upper()
    bonus = 1.0 if status == "PASS" else (0.5 if status == "PARTIAL" else (-1.0 if status == "FAIL" else 0.0))
    return float(sum(good) - sum(bad) + bonus)


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "No data."
    use = df.head(max_rows).copy()
    for col in use.columns:
        if pd.api.types.is_numeric_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        else:
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def _summary_for_run(audit_dir: Path, name: str) -> Dict[str, Any]:
    summary = _find_summary(audit_dir)
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    row["run_name"] = name
    row["audit_dir"] = str(audit_dir)
    ctx = SCORE_CONTEXT.get(name, {})
    row["score_macro_f1"] = ctx.get("macro_f1", np.nan)
    row["score_acc"] = ctx.get("acc", np.nan)
    return row


def _paired_compare(names: List[str], sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if len(names) < 2:
        return pd.DataFrame()
    left_name, right_name = names[0], names[1]
    left = sheets.get(left_name, pd.DataFrame())
    right = sheets.get(right_name, pd.DataFrame())
    if left.empty or right.empty or "sample_id" not in left.columns or "sample_id" not in right.columns:
        return pd.DataFrame()
    right_by_id = {str(row["sample_id"]): row for _, row in right.iterrows()}
    rows = []
    for _, lrow in left.iterrows():
        rid = str(lrow["sample_id"])
        rrow = right_by_id.get(rid)
        if rrow is None:
            continue
        lq = _quality_score(lrow)
        rq = _quality_score(rrow)
        diff = lq - rq
        if diff >= 1.0:
            change = "improved_vs_ce"
        elif diff <= -1.0:
            change = "worse_than_ce"
        else:
            change = "similar_to_ce"
        rows.append(
            {
                "sample_id": rid,
                "left_run": left_name,
                "right_run": right_name,
                "left_visual_status": lrow.get("visual_status", ""),
                "right_visual_status": rrow.get("visual_status", ""),
                "left_quality_score": lq,
                "right_quality_score": rq,
                "delta_quality_score": diff,
                "paired_visual_change": change,
                "left_figure_path": lrow.get("figure_path", ""),
                "right_figure_path": rrow.get("figure_path", ""),
            }
        )
    return pd.DataFrame(rows)


def _recommend(summary_df: pd.DataFrame, paired_df: pd.DataFrame, names: List[str]) -> str:
    if summary_df.empty:
        return "NEED_MORE_D13C_TUNING"
    if summary_df.get("audit_incomplete", pd.Series(dtype=bool)).astype(bool).any():
        return "NEED_MORE_D13C_TUNING"
    sup = summary_df[summary_df["run_name"].astype(str) == "d13c_m16_supcon_l005"]
    ce = summary_df[summary_df["run_name"].astype(str) == "d13c_m16_ce_continue"]
    if not sup.empty:
        s = sup.iloc[0]
        hard_fail = (
            _num(s.get("fail_rate"), 1.0) > 0.40
            or _num(s.get("avg_slot_traceability_score"), 0.0) < 1.0
            or _num(s.get("avg_slot_collapse_visual"), 9.0) >= 1.0
        )
        if hard_fail:
            return "D13C_VISUAL_UNRELIABLE_STOP"
        if not ce.empty:
            worse_count = int((paired_df.get("paired_visual_change", pd.Series(dtype=str)).astype(str) == "worse_than_ce").sum())
            improved_count = int((paired_df.get("paired_visual_change", pd.Series(dtype=str)).astype(str) == "improved_vs_ce").sum())
            if worse_count > improved_count:
                return "KEEP_D13B_M16_FINAL_SUPCON_NOT_VISUALLY_BETTER"
        if _num(s.get("pass_partial_rate"), 0.0) >= 0.60:
            return "USE_D13C_M16_SUPCON_L005_AS_DIAGNOSTIC_CANDIDATE"
    if "d13c_m8_supcon_l002_control" in names:
        m8 = summary_df[summary_df["run_name"].astype(str) == "d13c_m8_supcon_l002_control"]
        if not m8.empty and _num(m8.iloc[0].get("pass_partial_rate"), 0.0) >= 0.60:
            return "USE_D13C_M8_CONTROL_AS_COMPACT_BRANCH_NEEDS_SEPARATE_VALIDATION"
    return "NEED_MORE_D13C_TUNING"


def _row(summary_df: pd.DataFrame, run_name: str) -> Optional[pd.Series]:
    rows = summary_df[summary_df["run_name"].astype(str) == run_name]
    return rows.iloc[0] if not rows.empty else None


def _delta(left: Optional[pd.Series], right: Optional[pd.Series], col: str) -> float:
    if left is None or right is None:
        return np.nan
    return _num(left.get(col)) - _num(right.get(col))


def _paired_counts(paired_df: pd.DataFrame) -> Dict[str, int]:
    if paired_df.empty or "paired_visual_change" not in paired_df.columns:
        return {"improved_vs_ce": 0, "similar_to_ce": 0, "worse_than_ce": 0}
    counts = paired_df["paired_visual_change"].astype(str).value_counts().to_dict()
    return {
        "improved_vs_ce": int(counts.get("improved_vs_ce", 0)),
        "similar_to_ce": int(counts.get("similar_to_ce", 0)),
        "worse_than_ce": int(counts.get("worse_than_ce", 0)),
    }


def _visual_relation(counts: Dict[str, int]) -> str:
    if counts["improved_vs_ce"] > counts["worse_than_ce"]:
        return "better_overall_than_ce"
    if counts["worse_than_ce"] > counts["improved_vs_ce"]:
        return "worse_overall_than_ce"
    if counts["similar_to_ce"] > 0:
        return "similar_to_ce"
    return "unknown"


def run(args: argparse.Namespace) -> None:
    audits = [Path(p) for p in args.audits]
    names = [_sanitize(n) for n in args.names]
    if len(audits) != len(names):
        raise ValueError("--audits and --names must have the same length")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([_summary_for_run(path, name) for path, name in zip(audits, names)])
    area_rows = []
    risk_rows = []
    sheets: Dict[str, pd.DataFrame] = {}
    for path, name in zip(audits, names):
        area = _find_area(path)
        if not area.empty:
            area = area.copy()
            area.insert(0, "run_name", name)
            area_rows.append(area)
        risk = _find_risk(path)
        if not risk.empty:
            risk = risk.copy()
            risk.insert(0, "run_name", name)
            risk_rows.append(risk)
        sheets[name] = _find_sheet(path)
    area_df = pd.concat(area_rows, ignore_index=True) if area_rows else pd.DataFrame()
    risk_df = pd.concat(risk_rows, ignore_index=True) if risk_rows else pd.DataFrame()
    paired_df = _paired_compare(names, sheets)
    recommendation = _recommend(summary_df, paired_df, names)
    sup = _row(summary_df, "d13c_m16_supcon_l005")
    ce = _row(summary_df, "d13c_m16_ce_continue")
    m8 = _row(summary_df, "d13c_m8_supcon_l002_control")
    counts = _paired_counts(paired_df)
    relation = _visual_relation(counts)
    score_diff = SCORE_CONTEXT["d13c_m16_supcon_l005"]["macro_f1"] - SCORE_CONTEXT["d13c_m16_ce_continue"]["macro_f1"]
    mouth_delta = _delta(sup, ce, "avg_mouth_only_risk")
    center_delta = _delta(sup, ce, "avg_center_shortcut_risk")
    diversity_delta = _delta(sup, ce, "avg_slot_diversity_visual_score")
    readability_delta = _delta(sup, ce, "avg_slot_assignment_readability_score")
    collapse_delta = _delta(sup, ce, "avg_slot_collapse_visual")
    degradation = relation == "worse_overall_than_ce" or (
        np.isfinite(mouth_delta)
        and np.isfinite(center_delta)
        and (mouth_delta > 0.20 or center_delta > 0.20 or collapse_delta > 0.20)
    )

    summary_df.to_csv(output_dir / "d13c_post_visual_slot_compare_summary.csv", index=False)
    area_df.to_csv(output_dir / "d13c_post_visual_slot_compare_area.csv", index=False)
    risk_df.to_csv(output_dir / "d13c_post_visual_slot_compare_risks.csv", index=False)
    paired_df.to_csv(output_dir / "d13c_post_visual_slot_paired_sample_compare.csv", index=False)

    lines = [
        "# D13C Post Visual Slot Audit Comparison",
        "",
        "## 1. Context",
        "- Compare D13C SupCon l005 vs CE-only continuation when both are provided.",
        "- Optional M8 control can be included as a compact-control branch.",
        "- Goal: decide whether D13C candidate is visually acceptable.",
        "- No motif claim, no semantic-region claim, no causal-evidence claim.",
        "",
        "## 2. Score Context",
        f"- D13B M16 reference macro-F1 = {SCORE_CONTEXT['d13b_m16_reference']['macro_f1']:.4f}",
        f"- CE-only macro-F1 = {SCORE_CONTEXT['d13c_m16_ce_continue']['macro_f1']:.4f}",
        f"- SupCon l005 macro-F1 = {SCORE_CONTEXT['d13c_m16_supcon_l005']['macro_f1']:.4f}",
        f"- M8 control macro-F1 = {SCORE_CONTEXT['d13c_m8_supcon_l002_control']['macro_f1']:.4f}",
        "",
        "## 3. Visual Audit Summary",
        _md_table(summary_df),
        "",
        "## 4. SupCon vs CE-only",
        _md_table(paired_df["paired_visual_change"].value_counts().rename_axis("paired_visual_change").reset_index(name="count") if not paired_df.empty else pd.DataFrame()),
        "",
        f"- Visual reliability relation: `{relation}`.",
        f"- Paired samples: improved={counts['improved_vs_ce']}, similar={counts['similar_to_ce']}, worse={counts['worse_than_ce']}.",
        f"- Score difference l005 minus CE-only macro-F1: {score_diff:+.4f}.",
        f"- Visual degradation flag: {bool(degradation)}.",
        f"- Mouth-only risk delta l005 minus CE-only: {mouth_delta:+.4f}.",
        f"- Center-shortcut risk delta l005 minus CE-only: {center_delta:+.4f}.",
        f"- Slot diversity score delta l005 minus CE-only: {diversity_delta:+.4f}.",
        f"- Assignment readability score delta l005 minus CE-only: {readability_delta:+.4f}.",
        f"- Slot collapse visual delta l005 minus CE-only: {collapse_delta:+.4f}.",
        "",
        "Answer: l005 is acceptable only if score gain does not come with visual degradation. In this comparison the decision below uses paired visual counts plus risk deltas.",
        "",
        "## 5. Optional M8 Control",
        (
            "M8 control is present in this comparison. Treat it as a separate compact-control branch; "
            f"pass_partial_rate={_num(m8.get('pass_partial_rate')):.3f}, fail_count={int(_num(m8.get('fail_count'), 0))}."
            if m8 is not None
            else "M8 control was not included in this comparison."
        ),
        "",
        "M8 control should not silently replace the M16 candidate without a separate compactness-validation criterion.",
        "",
        "## 6. Final Recommendation",
        recommendation,
        "",
        "Forbidden outputs remain forbidden: OPEN_D13C_FULL, OPEN_SUPCON_FULL, MOTIF_DISCOVERED, SEMANTIC_REGION_DISCOVERED, CAUSAL_EVIDENCE_CONFIRMED.",
        "",
    ]
    (output_dir / "d13c_post_visual_slot_compare_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "recommendation": recommendation, "runs": names}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare post-D13C visual slot audit outputs.")
    parser.add_argument("--audits", nargs="+", required=True, help="Audit directories to compare.")
    parser.add_argument("--names", nargs="+", required=True, help="Run names matching --audits.")
    parser.add_argument("--output_dir", required=True, help="Comparison output directory.")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
