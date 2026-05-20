"""Multi-run comparison for D13A visual pooling / assignment audits.

The comparison is about reduction traceability and visual reliability only. It
does not make motif, semantic-region, or causal-evidence claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from compare_d13a_visual_pooling_audits import (
    _dominant_area,
    _load_audit,
    _risk_counts,
    _write_md_table,
)


SCORE_CONTEXT = {
    "k256": {"test_macro_f1": 0.5866, "test_acc": 0.6227, "assignment_entropy_global": 1.0000},
    "k144": {"test_macro_f1": 0.5829, "test_acc": 0.6166, "assignment_entropy_global": 0.8589},
    "baseline": {"test_macro_f1": 0.5829, "test_acc": 0.6166, "assignment_entropy_global": 0.8589},
    "anneal": {"test_macro_f1": 0.5813, "test_acc": 0.6141, "assignment_entropy_global": 0.8765},
}


def _score_for_name(name: str) -> Dict[str, float]:
    low = name.lower()
    for key, value in SCORE_CONTEXT.items():
        if key in low:
            return value
    return {"test_macro_f1": float("nan"), "test_acc": float("nan"), "assignment_entropy_global": float("nan")}


def _alias(name: str) -> str:
    low = name.lower()
    if "k256" in low:
        return "k256"
    if "anneal" in low:
        return "anneal"
    if "k144" in low or "baseline" in low:
        return "k144"
    safe = "".join(ch if ch.isalnum() else "_" for ch in low).strip("_")
    return safe or "run"


def _risk_summary(name: str, sheet: pd.DataFrame) -> Dict[str, int | str]:
    return {"run_name": name, **_risk_counts(sheet)}


def _numeric(row: pd.Series, col: str, default: float = 999.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    return float(default if pd.isna(value) else value)


def _paired_multi(audits: List[Dict[str, object]], names: List[str]) -> pd.DataFrame:
    prepared = []
    for audit, name in zip(audits, names):
        sheet = audit["sheet"]
        if sheet.empty or "sample_id" not in sheet:
            return pd.DataFrame()
        alias = _alias(name)
        keep = [
            "sample_id", "label", "visual_status", "region_softness_issue",
            "center_shortcut_risk", "assignment_interpretability_score",
            "region_traceability_score", "main_visible_area",
        ]
        work = sheet[[c for c in keep if c in sheet.columns]].copy()
        rename = {c: f"{alias}_{c}" for c in work.columns if c not in {"sample_id", "label"}}
        if "label" in work.columns:
            rename["label"] = f"{alias}_label"
        prepared.append(work.rename(columns=rename))

    merged = prepared[0]
    for work in prepared[1:]:
        merged = merged.merge(work, on="sample_id", how="inner")
    if merged.empty:
        return merged

    aliases = [_alias(name) for name in names]
    rows = []
    for _, row in merged.iterrows():
        best = None
        best_tuple = None
        for alias in aliases:
            candidate = (
                _numeric(row, f"{alias}_region_softness_issue"),
                _numeric(row, f"{alias}_center_shortcut_risk"),
                -_numeric(row, f"{alias}_assignment_interpretability_score", default=-999.0),
                -_numeric(row, f"{alias}_region_traceability_score", default=-999.0),
            )
            if best_tuple is None or candidate < best_tuple:
                best_tuple = candidate
                best = alias
            elif candidate == best_tuple:
                best = "TIE"
        label = ""
        for alias in aliases:
            label = row.get(f"{alias}_label", label)
            if label:
                break
        out = {"sample_id": row["sample_id"], "label": label}
        for alias in aliases:
            out[f"{alias}_status"] = row.get(f"{alias}_visual_status", "")
            out[f"{alias}_softness"] = row.get(f"{alias}_region_softness_issue", "")
            out[f"{alias}_center"] = row.get(f"{alias}_center_shortcut_risk", "")
            out[f"{alias}_assignment_interpretability"] = row.get(f"{alias}_assignment_interpretability_score", "")
            out[f"{alias}_main_area"] = row.get(f"{alias}_main_visible_area", "")
        out["best_visual_run"] = best or "TIE"
        out["notes"] = "Best is ranked by lower softness, lower center shortcut, higher assignment interpretability, then higher traceability. No motif claim."
        rows.append(out)
    return pd.DataFrame(rows)


def _recommend(summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    rows = {_alias(str(row["run_name"])): row for _, row in summary.iterrows()}
    anneal = rows.get("anneal")
    k144 = rows.get("k144")
    k256 = rows.get("k256")

    def pass_partial(row) -> float:
        return float(row.get("pass_partial_rate", 0.0) or 0.0) if row is not None else 0.0

    def interp(row) -> float:
        return float(row.get("avg_assignment_interpretability_score", 0.0) or 0.0) if row is not None else 0.0

    def soft(row) -> float:
        return float(row.get("avg_region_softness_issue", 9.0) or 9.0) if row is not None else 9.0

    def center(row) -> float:
        return float(row.get("avg_center_shortcut_risk", 9.0) or 9.0) if row is not None else 9.0

    def dom(row) -> float:
        return float(row.get("dominant_area_rate", 9.0) or 9.0) if row is not None else 9.0

    if anneal is not None:
        anneal_gate = pass_partial(anneal) >= 0.60 and interp(anneal) >= 1.2 and soft(anneal) < 1.2 and center(anneal) < 1.0
        if anneal_gate:
            return "USE_ANNEAL_FOR_D13B_DIAGNOSTIC"
        if k144 is not None:
            anneal_better = (
                pass_partial(anneal) >= 0.60
                and (
                    soft(anneal) < soft(k144)
                    or center(anneal) < center(k144) - 0.05
                    or dom(anneal) < dom(k144) - 0.05
                )
                and interp(anneal) >= interp(k144) - 0.05
            )
            if anneal_better:
                return "USE_ANNEAL_FOR_D13B_DIAGNOSTIC_WITH_CAUTION"
            if pass_partial(k144) >= 0.60 and soft(k144) < 1.5 and center(k144) < 1.2:
                return "USE_K144_FOR_D13B_DIAGNOSTIC_WITH_CAUTION"
    if k256 is not None and pass_partial(k256) >= 0.60 and k144 is None:
        return "USE_K256_FOR_D13B_DIAGNOSTIC_WITH_CAUTION"
    return "KEEP_D13B_LOCKED_IMPROVE_REDUCTION_BEFORE_MOTIF"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple D13A visual pooling audit outputs")
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if len(args.audits) != len(args.names):
        parser.error("--audits and --names must have the same length")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audits = [_load_audit(Path(path), name) for path, name in zip(args.audits, args.names)]
    rows = []
    for audit, name in zip(audits, args.names):
        row = dict(audit["row"])
        row["run_name"] = name
        row.update(_score_for_name(name))
        rows.append(row)

    summary_cols = [
        "run_name", "decision", "total_samples", "reviewed_samples", "pass_count", "partial_count",
        "fail_count", "pass_partial_rate", "avg_face_coverage_score", "avg_region_traceability_score",
        "avg_assignment_interpretability_score", "avg_region_softness_issue", "avg_hair_glasses_risk",
        "avg_background_border_risk", "avg_center_shortcut_risk", "dominant_area", "dominant_area_rate",
        "risk_case_count", "test_macro_f1", "test_acc", "assignment_entropy_global",
    ]
    summary = pd.DataFrame(rows)
    summary = summary[[c for c in summary_cols if c in summary.columns]]
    summary.to_csv(output_dir / "d13a_visual_pooling_multi_compare_summary.csv", index=False)

    area = pd.concat(
        [
            audit["area"].assign(run_name=name) if not audit["area"].empty else pd.DataFrame()
            for audit, name in zip(audits, args.names)
        ],
        ignore_index=True,
    )
    area.to_csv(output_dir / "d13a_visual_pooling_multi_compare_area.csv", index=False)

    risks = pd.DataFrame([_risk_summary(name, audit["sheet"]) for audit, name in zip(audits, args.names)])
    risks.to_csv(output_dir / "d13a_visual_pooling_multi_compare_risks.csv", index=False)

    paired = _paired_multi(audits, args.names)
    paired.to_csv(output_dir / "d13a_visual_pooling_multi_paired_sample_compare.csv", index=False)

    recommendation = _recommend(summary, paired)
    best_counts = paired["best_visual_run"].value_counts().to_dict() if not paired.empty and "best_visual_run" in paired else {}
    lines = [
        "# D13A Visual Pooling Audit Multi-Comparison: K256 vs K144 vs Anneal",
        "",
        "## 1. Context",
        "- D13A pure GNN hierarchical reduction.",
        "- K256 wins score but is too soft.",
        "- K144 improves softness/center risk but remains partial.",
        "- Anneal is tested because it should harden assignment gradually.",
        "- Goal is to choose a D13A base for D13B diagnostic, not claim motif.",
        "- Region nodes are soft learnable bottleneck nodes, not semantic regions.",
        "",
        "## 2. Score Context",
        "- K256: test_macro_f1 = 0.5866; test_acc = 0.6227; assignment_entropy ~= 1.0000.",
        "- K144: test_macro_f1 = 0.5829; test_acc = 0.6166; assignment_entropy ~= 0.8589.",
        "- Anneal: test_macro_f1 = 0.5813; test_acc = 0.6141; assignment_entropy ~= 0.8765.",
        "- Score gap between the three runs is small, so visual reliability decides the D13B diagnostic base.",
        "",
        "## 3. Audit Summary",
        _write_md_table(summary),
        "",
        "## 4. Area Distribution",
        _write_md_table(area, max_rows=40),
        "",
        "## 5. Risk Analysis",
        _write_md_table(risks),
        "",
        "## 6. Paired Sample Analysis",
        f"Paired sample rows: {len(paired)}.",
        f"Best visual run counts: {best_counts}.",
        "Written to `d13a_visual_pooling_multi_paired_sample_compare.csv`.",
        "",
        "## 7. Decision Rules",
        "- Use Anneal if it clears assignment interpretability, softness, center shortcut, and pass-partial gates.",
        "- Use Anneal with caution if it is clearly better than K144 on softness/center/area diversity while keeping traceability.",
        "- Use K144 with caution if Anneal is not better and K144 remains the cleaner non-K256 reduction.",
        "- K256 is not preferred for D13B if it only wins score but loses visual reliability.",
        "- Never output OPEN_D13B_FULL from this comparison.",
        "",
        "## 8. Final Recommendation",
        recommendation,
        "",
        "No motif, semantic-region, or causal-evidence claim is made.",
        "",
    ]
    (output_dir / "d13a_visual_pooling_multi_compare_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"best_visual_run_counts={best_counts}")
    print(f"recommendation={recommendation}")


if __name__ == "__main__":
    main()
