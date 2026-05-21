"""Compare D13D evidence diagnostics between D13C runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _read_one(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return pd.read_csv(path).iloc[0].to_dict()
    except Exception:
        return {}


def _num(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data."
    use = df.copy()
    for col in use.columns:
        if pd.api.types.is_numeric_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def _decision(rows: pd.DataFrame) -> str:
    if len(rows) < 2:
        return "SUPCON_NO_CLEAR_EVIDENCE_GAIN"
    left = rows.iloc[0]
    right = rows.iloc[1]
    gap_delta = _num(left.get("top1_vs_random1_gap")) - _num(right.get("top1_vs_random1_gap"))
    stability_delta = _num(left.get("avg_slot_map_similarity")) - _num(right.get("avg_slot_map_similarity"))
    if gap_delta > 0.005 and stability_delta >= -0.03:
        return "SUPCON_IMPROVES_EVIDENCE_DIAGNOSTIC"
    if _num(left.get("score_macro_f1")) > _num(right.get("score_macro_f1")) and gap_delta <= 0.005:
        return "SUPCON_IMPROVES_SCORE_NOT_EVIDENCE"
    return "SUPCON_NO_CLEAR_EVIDENCE_GAIN"


def run(args: argparse.Namespace) -> None:
    audits = [Path(p) for p in args.audits]
    names = list(args.names)
    if len(audits) != len(names):
        raise ValueError("--audits and --names must match")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_map = {
        "d13c_m16_supcon_l005": 0.6277,
        "d13c_m16_ce_continue": 0.6222,
    }
    rows: List[Dict[str, Any]] = []
    for audit, name in zip(audits, names):
        d = _read_one(audit / "d13d_deletion_summary.csv")
        s = _read_one(audit / "d13d_stability_summary.csv")
        row = {"run_name": name, "audit_dir": str(audit), "score_macro_f1": score_map.get(name, np.nan), **d, **s}
        rows.append(row)
    df = pd.DataFrame(rows)
    decision = _decision(df)
    df.to_csv(output_dir / "d13d_compare_summary.csv", index=False)
    lines = [
        "# D13D Evidence Diagnostic Comparison",
        "",
        "## 1. Context",
        "- Compare l005 vs CE-only under the same D13D deletion/stability protocol.",
        "- No motif claim and no causal-evidence claim.",
        "",
        "## 2. Summary",
        _md_table(df),
        "",
        "## 3. Decision",
        decision,
        "",
        "Decision options: SUPCON_IMPROVES_EVIDENCE_DIAGNOSTIC, SUPCON_IMPROVES_SCORE_NOT_EVIDENCE, SUPCON_NO_CLEAR_EVIDENCE_GAIN.",
        "",
    ]
    (output_dir / "d13d_compare_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": decision}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare D13D evidence diagnostic outputs.")
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
