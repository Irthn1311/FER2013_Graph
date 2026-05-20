"""Compare D13B visual slot audit summaries.

This comparison is a diagnostic gate only. It must not claim motifs, semantic
regions, causal evidence, SupCon readiness, or full D13C readiness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


SCORE_CONTEXT = {
    "d13b_k144_m16_deep_readout": {"test_macro_f1": 0.6187, "test_acc": 0.6328},
    "d13b_k144_m8_deep_region": {"test_macro_f1": 0.6171, "test_acc": 0.6344},
    "d13b_k256_m8_score_control": {"test_macro_f1": 0.6135, "test_acc": 0.6386},
}


def _sanitize(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))
    return safe.strip("_") or "run"


def _prefix(name: str) -> str:
    return f"d13b_{_sanitize(name)}_visual_slot"


def _read_first_existing(paths: List[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _load_audit(audit_dir: Path, name: str) -> Dict[str, Any]:
    prefix = _prefix(name)
    summary = _read_first_existing(
        [
            audit_dir / f"{prefix}_audit_summary.csv",
            *sorted(audit_dir.glob("d13b_*_visual_slot_audit_summary.csv")),
        ]
    )
    area = _read_first_existing(
        [
            audit_dir / f"{prefix}_area_summary.csv",
            *sorted(audit_dir.glob("d13b_*_visual_slot_area_summary.csv")),
            *sorted(audit_dir.glob("d13b_*_visual_slot_area_summary.csv")),
        ]
    )
    risk = _read_first_existing(
        [
            audit_dir / f"{prefix}_risk_cases.csv",
            *sorted(audit_dir.glob("d13b_*_visual_slot_risk_cases.csv")),
            *sorted(audit_dir.glob("d13b_*_risk_cases.csv")),
        ]
    )
    sheet = _read_first_existing(
        [
            audit_dir / f"{prefix}_audit_sheet_filled.csv",
            audit_dir / f"{prefix}_audit_sheet.csv",
            *sorted(audit_dir.glob("d13b_*_visual_slot_audit_sheet_filled.csv")),
            *sorted(audit_dir.glob("d13b_*_visual_slot_audit_sheet.csv")),
        ]
    )
    return {"name": name, "audit_dir": audit_dir, "summary": summary, "area": area, "risk": risk, "sheet": sheet}


def _summary_row(item: Dict[str, Any]) -> Dict[str, Any]:
    df = item["summary"]
    row = df.iloc[0].to_dict() if not df.empty else {}
    score = SCORE_CONTEXT.get(item["name"], {})
    return {
        "run_name": item["name"],
        "audit_dir": str(item["audit_dir"]),
        "test_macro_f1": score.get("test_macro_f1", row.get("test_macro_f1", "")),
        "test_acc": score.get("test_acc", row.get("test_acc", "")),
        "total_samples": row.get("total_samples", 0),
        "reviewed_samples": row.get("reviewed_samples", 0),
        "pass_count": row.get("pass_count", 0),
        "partial_count": row.get("partial_count", 0),
        "fail_count": row.get("fail_count", 0),
        "pass_partial_rate": row.get("pass_partial_rate", ""),
        "avg_slot_traceability_score": row.get("avg_slot_traceability_score", ""),
        "avg_slot_diversity_visual_score": row.get("avg_slot_diversity_visual_score", ""),
        "avg_slot_face_coverage_score": row.get("avg_slot_face_coverage_score", ""),
        "avg_slot_assignment_readability_score": row.get("avg_slot_assignment_readability_score", ""),
        "avg_mouth_only_risk": row.get("avg_mouth_only_risk", ""),
        "avg_center_shortcut_risk": row.get("avg_center_shortcut_risk", ""),
        "avg_hair_glasses_risk": row.get("avg_hair_glasses_risk", ""),
        "avg_background_border_risk": row.get("avg_background_border_risk", ""),
        "avg_slot_collapse_visual": row.get("avg_slot_collapse_visual", ""),
        "decision": row.get("decision", "VISUAL_SLOT_AUDIT_MISSING"),
    }


def _concat_with_name(items: List[Dict[str, Any]], key: str) -> pd.DataFrame:
    frames = []
    for item in items:
        df = item[key]
        if not df.empty:
            work = df.copy()
            work.insert(0, "run_name", item["name"])
            frames.append(work)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _paired(items: List[Dict[str, Any]]) -> pd.DataFrame:
    if len(items) < 2:
        return pd.DataFrame()
    sheets = []
    for item in items:
        df = item["sheet"]
        if df.empty or "sample_id" not in df:
            return pd.DataFrame()
        cols = [
            "sample_id",
            "visual_status",
            "slot_traceability_score",
            "slot_diversity_visual_score",
            "mouth_only_risk",
            "center_shortcut_risk",
            "slot_collapse_visual",
            "figure_path",
        ]
        work = df[[c for c in cols if c in df.columns]].copy()
        work = work.rename(columns={c: f"{item['name']}__{c}" for c in work.columns if c != "sample_id"})
        sheets.append(work)
    out = sheets[0]
    for other in sheets[1:]:
        out = out.merge(other, on="sample_id", how="inner")
    return out


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _recommend(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "KEEP_D13C_LOCKED_SLOT_VISUAL_UNRELIABLE"
    if summary["decision"].astype(str).str.contains("INCOMPLETE|MISSING", case=False, regex=True).any():
        return "KEEP_D13C_LOCKED_SLOT_VISUAL_UNRELIABLE"
    work = summary.copy()
    work["visual_score"] = (
        pd.to_numeric(work["pass_partial_rate"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["avg_slot_traceability_score"], errors="coerce").fillna(0.0) / 2.0
        + pd.to_numeric(work["avg_slot_diversity_visual_score"], errors="coerce").fillna(0.0) / 2.0
        - pd.to_numeric(work["avg_mouth_only_risk"], errors="coerce").fillna(2.0) / 2.0
        - pd.to_numeric(work["avg_center_shortcut_risk"], errors="coerce").fillna(2.0) / 2.0
        - pd.to_numeric(work["avg_slot_collapse_visual"], errors="coerce").fillna(2.0) / 2.0
    )
    best = work.sort_values(["visual_score", "test_macro_f1"], ascending=False).iloc[0]
    name = str(best["run_name"])
    if _num(best.get("visual_score"), -99) < 0.5:
        return "NEED_MORE_D13B_SLOT_TUNING"
    if name == "d13b_k144_m16_deep_readout":
        return "USE_M16_DEEP_READOUT_FOR_D13C_DIAGNOSTIC"
    if name == "d13b_k144_m8_deep_region":
        score_gap = SCORE_CONTEXT["d13b_k144_m16_deep_readout"]["test_macro_f1"] - SCORE_CONTEXT["d13b_k144_m8_deep_region"]["test_macro_f1"]
        if abs(score_gap) <= 0.003:
            return "USE_M8_DEEP_REGION_FOR_VISUAL_INTERPRETABILITY_DESPITE_SCORE"
        return "USE_M8_DEEP_REGION_FOR_D13C_DIAGNOSTIC"
    return "NEED_MORE_D13B_SLOT_TUNING"


def _md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare D13B visual slot audit outputs.")
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if len(args.audits) != len(args.names):
        parser.error("--audits and --names must have the same length")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = [_load_audit(Path(path), name) for path, name in zip(args.audits, args.names)]
    summary = pd.DataFrame([_summary_row(item) for item in items])
    area = _concat_with_name(items, "area")
    risks = _concat_with_name(items, "risk")
    paired = _paired(items)
    recommendation = _recommend(summary)

    summary.to_csv(output_dir / "d13b_visual_slot_compare_summary.csv", index=False)
    area.to_csv(output_dir / "d13b_visual_slot_compare_area.csv", index=False)
    risks.to_csv(output_dir / "d13b_visual_slot_compare_risks.csv", index=False)
    if not paired.empty:
        paired.to_csv(output_dir / "d13b_visual_slot_paired_sample_compare.csv", index=False)

    lines = [
        "# D13B Visual Slot Audit Comparison",
        "",
        "## 1. Context",
        "- Compare best-score candidate vs balanced candidate.",
        "- Goal: select D13B candidate for D13C diagnostic or stop.",
        "- No SupCon, no full D13C, no motif claim, no semantic-region claim.",
        "",
        "## 2. Score Context",
        _md_table(summary[["run_name", "test_macro_f1", "test_acc", "decision"]]),
        "",
        "## 3. Visual Audit Summary",
        _md_table(summary),
        "",
        "## 4. M8 vs M16",
        "- M16 has more slots; prefer it only if visual diversity and traceability improve, not from score alone.",
        "- M8 is easier to read if it reaches similar pass/partial and lower shortcut/collapse risks.",
        "- Deep readout score gain must be supported by visual slot quality before using it downstream.",
        "- Deep region is favored for interpretability if it is cleaner at near-equal score.",
        "",
        "## 5. Optional K256 control",
        "- K256 score-control should not be selected for a motif path if visual slots are worse than K144 visual-base candidates.",
        "",
        "## 6. Final Recommendation",
        recommendation,
        "",
        "Do not output OPEN_SUPCON or OPEN_D13C_FULL from this comparison.",
        "",
    ]
    (output_dir / "d13b_visual_slot_compare_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "recommendation": recommendation}, indent=2))


if __name__ == "__main__":
    main()
