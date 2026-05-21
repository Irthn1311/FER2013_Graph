"""AI-assisted heuristic review for post-D13C visual slot audits.

The helper fills audit sheets from saved masks and metadata. It is diagnostic
only and requires human confirmation for any final interpretation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from review_d13b_visual_slot_audit import _has_review_status, _review_row


REVIEW_STATUS = "AI_ASSISTED_REVIEW"
REVIEW_COLUMNS = [
    "review_status",
    "visual_status",
    "slot_traceability_score",
    "slot_diversity_visual_score",
    "slot_face_coverage_score",
    "slot_assignment_readability_score",
    "mouth_only_risk",
    "center_shortcut_risk",
    "hair_glasses_risk",
    "background_border_risk",
    "dominant_slot_area",
    "multi_region_support",
    "slot_collapse_visual",
    "supcon_visual_change",
    "notes",
]

SHEET_COLUMNS = [
    "sample_id",
    "split",
    "label",
    "pred",
    "confidence",
    "correct",
    "run_name",
    "num_slots",
    "lambda_supcon",
    "projection_dim",
    "figure_path",
    "metadata_path",
    *REVIEW_COLUMNS,
]


def _sanitize(value: str | None, fallback: str = "run") -> str:
    raw = str(value or fallback or "run").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    return safe.strip("_") or "run"


def _prefix(run_name: str) -> str:
    return f"d13c_{_sanitize(run_name)}_post_visual_slot"


def _find_sheet(audit_dir: Path, run_name: str, output_sheet: Optional[Path]) -> Path:
    out_resolved = output_sheet.resolve() if output_sheet is not None else None
    candidates = [
        audit_dir / f"{_prefix(run_name)}_audit_sheet.csv",
        audit_dir / f"{_prefix(run_name)}_audit_sheet_filled.csv",
    ]
    candidates.extend(sorted(audit_dir.glob("d13c_*_post_visual_slot_audit_sheet.csv")))
    candidates.extend(sorted(audit_dir.glob("*audit_sheet.csv")))
    for path in candidates:
        if not path.exists():
            continue
        if out_resolved is not None and path.resolve() == out_resolved:
            continue
        return path
    raise FileNotFoundError(f"Could not find base audit sheet in {audit_dir}")


def _find_reference_sheet(audit_dir: Optional[Path]) -> pd.DataFrame:
    if audit_dir is None:
        return pd.DataFrame()
    candidates: List[Path] = []
    candidates.extend(sorted(audit_dir.glob("*_filled.csv")))
    candidates.extend(sorted(audit_dir.glob("*audit_sheet.csv")))
    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return pd.DataFrame()


def _numeric(row: pd.Series, col: str, default: float = np.nan) -> float:
    try:
        value = row.get(col, default)
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _visual_bonus(status: Any) -> float:
    text = str(status).strip().upper()
    if text == "PASS":
        return 1.0
    if text == "PARTIAL":
        return 0.5
    if text == "FAIL":
        return -1.0
    return 0.0


def _quality_score(row: pd.Series) -> float:
    good = [
        _numeric(row, "slot_traceability_score", 0.0),
        _numeric(row, "slot_diversity_visual_score", 0.0),
        _numeric(row, "slot_face_coverage_score", 0.0),
        _numeric(row, "slot_assignment_readability_score", 0.0),
        _numeric(row, "multi_region_support", 0.0),
        _visual_bonus(row.get("visual_status")),
    ]
    bad = [
        _numeric(row, "mouth_only_risk", 0.0),
        _numeric(row, "center_shortcut_risk", 0.0),
        _numeric(row, "hair_glasses_risk", 0.0),
        _numeric(row, "background_border_risk", 0.0),
        _numeric(row, "slot_collapse_visual", 0.0),
    ]
    return float(sum(good) - sum(bad))


def _supcon_visual_change(row: pd.Series, ce_by_sample: Dict[str, pd.Series]) -> str:
    ce_row = ce_by_sample.get(str(row.get("sample_id")))
    if ce_row is None or not _has_review_status(ce_row.get("review_status")):
        return "unknown"
    diff = _quality_score(row) - _quality_score(ce_row)
    if diff >= 1.0:
        return "improved_vs_ce"
    if diff <= -1.0:
        return "worse_than_ce"
    return "similar_to_ce"


def _merge_existing(df: pd.DataFrame, existing: pd.DataFrame, overwrite: bool) -> pd.DataFrame:
    if overwrite or existing.empty or "sample_id" not in df.columns or "sample_id" not in existing.columns:
        return df
    existing_by_id = {str(row["sample_id"]): row for _, row in existing.iterrows()}
    out = df.copy()
    for idx, row in out.iterrows():
        old = existing_by_id.get(str(row["sample_id"]))
        if old is None or not _has_review_status(old.get("review_status")):
            continue
        for col in REVIEW_COLUMNS:
            if col in old.index:
                out.at[idx, col] = old[col]
    return out


def _high_risk_mask(df: pd.DataFrame) -> pd.Series:
    status = df.get("visual_status", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
    risk = status.eq("FAIL") | df.get("supcon_visual_change", pd.Series(index=df.index, dtype=str)).astype(str).eq("worse_than_ce")
    for col in ["mouth_only_risk", "center_shortcut_risk", "hair_glasses_risk", "background_border_risk", "slot_collapse_visual"]:
        vals = pd.to_numeric(df.get(col, pd.Series(index=df.index, dtype=float)), errors="coerce")
        risk = risk | vals.ge(2)
    for col in ["slot_traceability_score", "slot_diversity_visual_score", "slot_assignment_readability_score"]:
        vals = pd.to_numeric(df.get(col, pd.Series(index=df.index, dtype=float)), errors="coerce")
        risk = risk | vals.eq(0)
    return risk


def _write_todo(output_sheet: Path, df: pd.DataFrame, run_name: str) -> Optional[Path]:
    todo_path = output_sheet.with_name(output_sheet.stem + "_review_todo.md")
    risk_df = df.loc[_high_risk_mask(df)].copy()
    lines = [
        f"# D13C Post Visual Slot Review TODO: {run_name}",
        "",
        "These cases were flagged by AI-assisted heuristic review and need human confirmation before any final interpretation.",
        "",
        "| sample_id | visual_status | supcon_visual_change | dominant_slot_area | risks | figure_path | notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in risk_df.iterrows():
        risks = []
        for col in ["mouth_only_risk", "center_shortcut_risk", "hair_glasses_risk", "background_border_risk", "slot_collapse_visual"]:
            try:
                if float(row.get(col, 0)) >= 2:
                    risks.append(col)
            except Exception:
                pass
        if str(row.get("supcon_visual_change", "")) == "worse_than_ce":
            risks.append("worse_than_ce")
        notes = str(row.get("notes", "")).replace("|", "/")[:220]
        lines.append(
            f"| {row.get('sample_id', '')} | {row.get('visual_status', '')} | {row.get('supcon_visual_change', '')} | "
            f"{row.get('dominant_slot_area', '')} | {', '.join(risks) or 'score/readability gate'} | "
            f"{row.get('figure_path', '')} | {notes} |"
        )
    if risk_df.empty:
        lines.append("| none | PASS/PARTIAL only |  |  |  |  |  |")
    todo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return todo_path


def run(args: argparse.Namespace) -> None:
    audit_dir = Path(args.audit_dir)
    run_name = _sanitize(args.run_name, audit_dir.name)
    output_sheet = Path(args.output_sheet)
    base_sheet = _find_sheet(audit_dir, run_name, output_sheet)
    df = pd.read_csv(base_sheet)
    for col in SHEET_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[SHEET_COLUMNS + [c for c in df.columns if c not in SHEET_COLUMNS]]
    for col in REVIEW_COLUMNS:
        df[col] = df[col].astype("object")
    if args.limit is not None:
        df = df.head(int(args.limit)).copy()

    existing = pd.read_csv(output_sheet) if output_sheet.exists() else pd.DataFrame()
    ce_ref = _find_reference_sheet(Path(args.ce_reference_audit_dir) if args.ce_reference_audit_dir else None)
    ce_by_sample = {str(row["sample_id"]): row for _, row in ce_ref.iterrows()} if "sample_id" in ce_ref.columns else {}

    reviewed_rows = 0
    preserved_rows = 0
    for idx, row in df.iterrows():
        if args.preserve_existing_reviews and not args.overwrite and not existing.empty and "sample_id" in existing.columns:
            old = existing.loc[existing["sample_id"].astype(str) == str(row.get("sample_id"))]
            if not old.empty and _has_review_status(old.iloc[0].get("review_status")):
                for col in REVIEW_COLUMNS:
                    if col in old.columns:
                        df.at[idx, col] = old.iloc[0][col]
                preserved_rows += 1
                continue
        review: Dict[str, Any] = _review_row(audit_dir, row)
        review["review_status"] = REVIEW_STATUS
        for col, value in review.items():
            if col in df.columns:
                df.at[idx, col] = value
        reviewed_rows += 1

    if not args.overwrite and not args.preserve_existing_reviews and output_sheet.exists():
        df = _merge_existing(df, existing, overwrite=False)

    for idx, row in df.iterrows():
        if args.ce_reference_audit_dir:
            df.at[idx, "supcon_visual_change"] = _supcon_visual_change(row, ce_by_sample)
        elif not str(row.get("supcon_visual_change", "")).strip():
            df.at[idx, "supcon_visual_change"] = "unknown"
        note = str(df.at[idx, "notes"] or "")
        if "AI-assisted heuristic review" not in note:
            note = (note + "; " if note else "") + "AI-assisted heuristic review; requires human confirmation for final motif interpretation."
        if "not a motif claim" not in note:
            note += " This is not a motif claim, semantic-region claim, or causal-evidence claim."
        df.at[idx, "notes"] = note

    output_sheet.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_sheet, index=False)
    todo_path = _write_todo(output_sheet, df, run_name)
    counts = df["visual_status"].astype(str).str.upper().value_counts().to_dict()
    result = {
        "audit_dir": str(audit_dir),
        "base_sheet": str(base_sheet),
        "output_sheet": str(output_sheet),
        "run_name": run_name,
        "rows_written": int(len(df)),
        "rows_reviewed_by_heuristic": int(reviewed_rows),
        "rows_preserved": int(preserved_rows),
        "ce_reference_audit_dir": str(args.ce_reference_audit_dir) if args.ce_reference_audit_dir else None,
        "review_status": REVIEW_STATUS,
        "visual_status_counts": counts,
        "todo_path": str(todo_path) if todo_path else None,
        "no_motif_claim": True,
        "no_semantic_region_claim": True,
        "no_causal_claim": True,
    }
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill D13C post visual slot audit sheets with AI-assisted heuristic reviews.")
    parser.add_argument("--audit_dir", required=True, help="Run-specific post visual slot audit directory.")
    parser.add_argument("--run_name", required=True, help="D13C run name.")
    parser.add_argument("--output_sheet", required=True, help="Filled review sheet path to write.")
    parser.add_argument("--preserve_existing_reviews", action="store_true", help="Preserve existing reviewed rows unless --overwrite is set.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing review rows.")
    parser.add_argument("--limit", type=int, default=None, help="Only review the first N rows for smoke testing.")
    parser.add_argument("--ce_reference_audit_dir", default=None, help="Optional CE-only audit dir for paired SupCon visual comparison.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
