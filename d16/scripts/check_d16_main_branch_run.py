"""Check D16 main-branch run artifacts.

This checker is accuracy-first and keeps part-attention diagnostics optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


HARD_CLASSES = {0: "Angry", 2: "Fear", 4: "Sad", 6: "Neutral"}


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def as_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def latest(rows: List[Dict[str, str]]) -> Dict[str, str]:
    return rows[-1] if rows else {}


def finite(value: Any) -> bool:
    value = as_float(value)
    return bool(math.isfinite(value))


def attention_status(run_dir: Path) -> Dict[str, Any]:
    summary = read_rows(run_dir / "part_attention_summary.csv")
    by_class = read_rows(run_dir / "part_attention_by_class.csv")
    if not summary and not by_class:
        return {"present": False, "status": "NOT_AVAILABLE", "failures": []}
    failures: List[str] = []
    if len(summary) != 5:
        failures.append(f"part_attention_summary.csv row_count={len(summary)} expected=5")
    if by_class and len(by_class) != 35:
        failures.append(f"part_attention_by_class.csv row_count={len(by_class)} expected=35")
    for name, rows in (("part_attention_summary.csv", summary), ("part_attention_by_class.csv", by_class)):
        for idx, row in enumerate(rows):
            val = as_float(row.get("attention_mean"))
            samples = as_int(row.get("samples"))
            if samples > 0 and not math.isfinite(val):
                failures.append(f"{name} row {idx} attention_mean is non-finite")
            if math.isfinite(val) and not (0.0 <= val <= 1.0):
                failures.append(f"{name} row {idx} attention_mean out of [0,1]: {val}")
    return {
        "present": True,
        "status": "PASS" if not failures else "FAIL",
        "summary_rows": len(summary),
        "by_class_rows": len(by_class),
        "failures": failures,
    }


def part_token_status(run_dir: Path) -> Dict[str, Any]:
    summary = read_rows(run_dir / "part_token_transformer_summary.csv")
    by_class = read_rows(run_dir / "part_token_transformer_by_class.csv")
    if not summary and not by_class:
        return {"present": False, "status": "NOT_AVAILABLE", "failures": []}
    failures: List[str] = []
    if len(summary) != 5:
        failures.append(f"part_token_transformer_summary.csv row_count={len(summary)} expected=5")
    if by_class and len(by_class) != 35:
        failures.append(f"part_token_transformer_by_class.csv row_count={len(by_class)} expected=35")
    for name, rows in (
        ("part_token_transformer_summary.csv", summary),
        ("part_token_transformer_by_class.csv", by_class),
    ):
        for idx, row in enumerate(rows):
            samples = as_int(row.get("valid_samples"))
            for field in ("token_norm_mean", "transformed_token_norm_mean"):
                val = as_float(row.get(field))
                if samples > 0 and not math.isfinite(val):
                    failures.append(f"{name} row {idx} {field} is non-finite")
                if math.isfinite(val) and val < 0.0:
                    failures.append(f"{name} row {idx} {field} is negative: {val}")
    return {
        "present": True,
        "status": "PASS" if not failures else "FAIL",
        "summary_rows": len(summary),
        "by_class_rows": len(by_class),
        "failures": failures,
    }


def part_motif_status(run_dir: Path) -> Dict[str, Any]:
    summary = read_rows(run_dir / "part_motif_summary.csv")
    by_class = read_rows(run_dir / "part_motif_by_class.csv")
    similarity = read_rows(run_dir / "part_motif_similarity.csv")
    if not summary and not by_class and not similarity:
        return {"present": False, "status": "NOT_AVAILABLE", "failures": [], "warnings": []}
    failures: List[str] = []
    warnings: List[str] = []
    if len(summary) != 12:
        warnings.append(f"part_motif_summary.csv row_count={len(summary)} expected=12 for A3 default")
    if by_class and len(by_class) != 84:
        warnings.append(f"part_motif_by_class.csv row_count={len(by_class)} expected=84 for A3 default")
    if similarity and len(similarity) != 144:
        warnings.append(f"part_motif_similarity.csv row_count={len(similarity)} expected=144 for A3 default")

    for name, rows in (
        ("part_motif_summary.csv", summary),
        ("part_motif_by_class.csv", by_class),
    ):
        for idx, row in enumerate(rows):
            samples = as_int(row.get("samples"))
            for field in (
                "motif_usage_mean",
                "motif_attention_entropy_mean",
                "motif_attention_peak_mean",
                "motif_part_mass_mean",
            ):
                val = as_float(row.get(field))
                if samples > 0 and not math.isfinite(val):
                    failures.append(f"{name} row {idx} {field} is non-finite")
                if field in {"motif_usage_mean", "motif_attention_peak_mean", "motif_part_mass_mean"} and math.isfinite(val):
                    if not (0.0 <= val <= 1.0 + 1e-6):
                        failures.append(f"{name} row {idx} {field} out of [0,1]: {val}")
            for field in ("motif_token_norm_mean", "motif_transformed_token_norm_mean"):
                if field in row:
                    val = as_float(row.get(field))
                    if samples > 0 and not math.isfinite(val):
                        failures.append(f"{name} row {idx} {field} is non-finite")
    for idx, row in enumerate(similarity):
        val = as_float(row.get("cosine_mean"))
        if not math.isfinite(val):
            failures.append(f"part_motif_similarity.csv row {idx} cosine_mean is non-finite")
        elif not (-1.0 - 1e-6 <= val <= 1.0 + 1e-6):
            failures.append(f"part_motif_similarity.csv row {idx} cosine_mean out of [-1,1]: {val}")

    if summary:
        offdiag = as_float(summary[0].get("avg_offdiag_similarity_mean"))
        effective = as_float(summary[0].get("effective_motif_count_mean"))
        if math.isfinite(offdiag) and offdiag > 0.90:
            warnings.append(f"average off-diagonal motif similarity is high: {offdiag:.6f}")
        if math.isfinite(effective) and effective < 2.0:
            warnings.append(f"effective motif count is very low: {effective:.6f}")
        usage_by_part: Dict[str, List[float]] = {}
        for row in summary:
            usage_by_part.setdefault(str(row.get("part_name")), []).append(as_float(row.get("motif_usage_mean")))
            mass = as_float(row.get("motif_part_mass_mean"))
            part_name = str(row.get("part_name"))
            if part_name != "global" and math.isfinite(mass) and mass < 0.20:
                warnings.append(f"{row.get('motif_name')} motif_part_mass is low: {mass:.6f}")
            peak = as_float(row.get("motif_attention_peak_mean"))
            if math.isfinite(peak) and peak > 0.90:
                warnings.append(f"{row.get('motif_name')} attention peak is very high: {peak:.6f}")
        for part_name, values in usage_by_part.items():
            finite_values = [value for value in values if math.isfinite(value)]
            total = sum(finite_values)
            if total > 0.0 and max(finite_values) / total > 0.80 and len(finite_values) > 1:
                warnings.append(f"one motif dominates usage within {part_name}")
    return {
        "present": True,
        "status": "PASS" if not failures else "FAIL",
        "summary_rows": len(summary),
        "by_class_rows": len(by_class),
        "similarity_rows": len(similarity),
        "failures": failures,
        "warnings": warnings,
    }


def micro_motif_status(run_dir: Path) -> Dict[str, Any]:
    summary = read_rows(run_dir / "micro_motif_summary.csv")
    by_class = read_rows(run_dir / "micro_motif_by_class.csv")
    similarity = read_rows(run_dir / "micro_motif_similarity.csv")
    if not summary and not by_class and not similarity:
        return {"present": False, "status": "NOT_AVAILABLE", "failures": [], "warnings": []}
    failures: List[str] = []
    warnings: List[str] = []
    is_a4b = "no_global_micro" in str(run_dir)
    expected_major = 12
    expected_micro = 7 if is_a4b else 8
    expected_summary = expected_major + expected_micro
    expected_by_class = expected_summary * 7
    expected_similarity = expected_major * expected_major + expected_micro * expected_micro
    if len(summary) != expected_summary:
        message = f"micro_motif_summary.csv row_count={len(summary)} expected={expected_summary}"
        (failures if is_a4b else warnings).append(message)
    if by_class and len(by_class) != expected_by_class:
        message = f"micro_motif_by_class.csv row_count={len(by_class)} expected={expected_by_class}"
        (failures if is_a4b else warnings).append(message)
    if similarity and len(similarity) != expected_similarity:
        message = f"micro_motif_similarity.csv row_count={len(similarity)} expected={expected_similarity}"
        (failures if is_a4b else warnings).append(message)

    branch_counts: Dict[str, int] = {}
    for name, rows in (
        ("micro_motif_summary.csv", summary),
        ("micro_motif_by_class.csv", by_class),
    ):
        for idx, row in enumerate(rows):
            samples = as_int(row.get("samples"))
            branch = str(row.get("branch"))
            if name == "micro_motif_summary.csv":
                branch_counts[branch] = branch_counts.get(branch, 0) + 1
            for field in (
                "motif_usage_mean",
                "motif_attention_entropy_mean",
                "motif_attention_peak_mean",
                "motif_part_mass_mean",
            ):
                val = as_float(row.get(field))
                if samples > 0 and not math.isfinite(val):
                    failures.append(f"{name} row {idx} {field} is non-finite")
                if field in {"motif_usage_mean", "motif_attention_peak_mean", "motif_part_mass_mean"} and math.isfinite(val):
                    if not (0.0 <= val <= 1.0 + 1e-6):
                        failures.append(f"{name} row {idx} {field} out of [0,1]: {val}")
            for field in ("motif_token_norm_mean", "motif_transformed_token_norm_mean"):
                if field in row:
                    val = as_float(row.get(field))
                    if samples > 0 and not math.isfinite(val):
                        failures.append(f"{name} row {idx} {field} is non-finite")
            if branch == "micro":
                detail = as_float(row.get("micro_detail_score_mean"))
                if samples > 0 and not math.isfinite(detail):
                    failures.append(f"{name} row {idx} micro_detail_score_mean is non-finite")
                if math.isfinite(detail) and abs(detail) > 5.0:
                    warnings.append(f"{row.get('motif_name')} micro_detail_score_mean is unusually large: {detail:.6f}")
            gate = as_float(row.get("micro_gate_mean"))
            if math.isfinite(gate) and not (0.0 <= gate <= 1.0 + 1e-6):
                failures.append(f"{name} row {idx} micro_gate_mean out of [0,1]: {gate}")
    if summary and (branch_counts.get("major", 0) != 12 or branch_counts.get("micro", 0) != 8):
        message = (
            f"micro_motif_summary branch counts expected major={expected_major} "
            f"micro={expected_micro} got {branch_counts}"
        )
        (failures if is_a4b else warnings).append(message)
    if is_a4b:
        global_micro = [
            row.get("motif_name")
            for row in summary
            if str(row.get("branch")) == "micro" and str(row.get("motif_name", "")).startswith("global_micro")
        ]
        if global_micro:
            failures.append(f"A4b diagnostics must not include global micro motifs: {global_micro}")

    for idx, row in enumerate(similarity):
        val = as_float(row.get("cosine_mean"))
        if not math.isfinite(val):
            failures.append(f"micro_motif_similarity.csv row {idx} cosine_mean is non-finite")
        elif not (-1.0 - 1e-6 <= val <= 1.0 + 1e-6):
            failures.append(f"micro_motif_similarity.csv row {idx} cosine_mean out of [-1,1]: {val}")

    micro_rows = [row for row in summary if str(row.get("branch")) == "micro"]
    if micro_rows:
        offdiag = as_float(micro_rows[0].get("avg_offdiag_similarity_mean"))
        effective = as_float(micro_rows[0].get("effective_motif_count_mean"))
        gate = as_float(micro_rows[0].get("micro_gate_mean"))
        if math.isfinite(offdiag) and offdiag > 0.90:
            warnings.append(f"average micro off-diagonal similarity is high: {offdiag:.6f}")
        if math.isfinite(effective) and effective < 2.0:
            warnings.append(f"effective micro motif count is very low: {effective:.6f}")
        if math.isfinite(gate) and gate > 0.90:
            warnings.append(f"micro_gate_mean near 1; micro support may dominate: {gate:.6f}")
        if math.isfinite(gate) and gate < 0.05:
            warnings.append(f"micro_gate_mean near 0; micro support may be unused: {gate:.6f}")
        usage_by_part: Dict[str, List[float]] = {}
        for row in micro_rows:
            part_name = str(row.get("part_name"))
            usage_by_part.setdefault(part_name, []).append(as_float(row.get("motif_usage_mean")))
            mass = as_float(row.get("motif_part_mass_mean"))
            if part_name != "global" and math.isfinite(mass) and mass < 0.20:
                warnings.append(f"{row.get('motif_name')} micro motif_part_mass is low: {mass:.6f}")
            peak = as_float(row.get("motif_attention_peak_mean"))
            entropy = as_float(row.get("motif_attention_entropy_mean"))
            if math.isfinite(peak) and peak > 0.90:
                warnings.append(f"{row.get('motif_name')} micro attention peak is very high: {peak:.6f}")
            if math.isfinite(entropy) and entropy > 7.5:
                warnings.append(f"{row.get('motif_name')} micro attention entropy is very high: {entropy:.6f}")
        for part_name, values in usage_by_part.items():
            finite_values = [value for value in values if math.isfinite(value)]
            total = sum(finite_values)
            if total > 0.0 and max(finite_values) / total > 0.80 and len(finite_values) > 1:
                warnings.append(f"one micro motif dominates usage within {part_name}")
    return {
        "present": True,
        "status": "PASS" if not failures else "FAIL",
        "summary_rows": len(summary),
        "by_class_rows": len(by_class),
        "similarity_rows": len(similarity),
        "expected_micro_tokens": expected_micro,
        "micro_tokens": branch_counts.get("micro", 0),
        "failures": failures,
        "warnings": warnings,
    }


def check_run(run_dir: Path) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    required = [
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "test_metrics.csv",
        "last_test_metrics.csv",
        "per_class_metrics.csv",
        "detected_vs_fallback_metrics.csv",
        "detected_fallback_per_class_metrics.csv",
        "confusion_matrix.csv",
        "predictions.csv",
        "d16_train_summary.json",
    ]
    for rel in required:
        if not (run_dir / rel).exists():
            failures.append(f"missing {rel}")

    test = latest(read_rows(run_dir / "test_metrics.csv"))
    pred_count = read_rows(run_dir / "pred_count.csv")
    per_class = read_rows(run_dir / "per_class_metrics.csv")
    groups = read_rows(run_dir / "detected_vs_fallback_metrics.csv")
    group_per_class = read_rows(run_dir / "detected_fallback_per_class_metrics.csv")

    test_accuracy = as_float(test.get("accuracy"))
    test_macro_f1 = as_float(test.get("macro_f1"))
    if not finite(test_accuracy):
        failures.append("test_accuracy missing or non-finite")
    if not finite(test_macro_f1):
        failures.append("test_macro_f1 missing or non-finite")

    predicted_classes = as_int(test.get("predicted_classes"))
    if pred_count:
        predicted_classes = sum(1 for row in pred_count if as_int(row.get("pred_count")) > 0)
    if predicted_classes < 7:
        failures.append(f"predicted_classes={predicted_classes} expected=7")

    if not groups:
        failures.append("missing detected/fallback group metrics")
    else:
        group_names = {str(row.get("group")) for row in groups}
        if "detected" not in group_names or "fallback" not in group_names:
            failures.append(f"detected/fallback groups incomplete: {sorted(group_names)}")
    if not group_per_class:
        failures.append("missing detected_fallback_per_class_metrics.csv rows")

    hard_seen = {as_int(row.get("class_id")) for row in per_class if as_int(row.get("class_id")) in HARD_CLASSES}
    missing_hard = sorted(set(HARD_CLASSES) - hard_seen)
    if missing_hard:
        failures.append(f"missing hard class metrics: {missing_hard}")
    for row in per_class:
        if as_int(row.get("class_id")) in HARD_CLASSES and not finite(row.get("f1")):
            failures.append(f"hard class {row.get('class_id')} f1 non-finite")

    attention = attention_status(run_dir)
    part_token = part_token_status(run_dir)
    part_motif = part_motif_status(run_dir)
    micro_motif = micro_motif_status(run_dir)
    if attention["status"] == "FAIL":
        failures.extend(attention["failures"])
    if part_token["status"] == "FAIL":
        failures.extend(part_token["failures"])
    if part_motif["status"] == "FAIL":
        failures.extend(part_motif["failures"])
    if micro_motif["status"] == "FAIL":
        failures.extend(micro_motif["failures"])
    warnings.extend(part_motif.get("warnings", []))
    warnings.extend(micro_motif.get("warnings", []))
    if not attention["present"] and not part_token["present"] and not part_motif["present"] and not micro_motif["present"]:
        warnings.append("readout diagnostics not found")

    if predicted_classes < 7:
        decision = "REJECT_RUN_COLLAPSE"
    elif failures:
        decision = "D16_MAIN_BRANCH_CHECK_FAIL"
    else:
        decision = "D16_MAIN_BRANCH_CHECK_PASS"

    return {
        "run_dir": str(run_dir),
        "decision": decision,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "predicted_classes": predicted_classes,
        "attention": attention,
        "part_token_transformer": part_token,
        "part_motif_query": part_motif,
        "micro_motif_support": micro_motif,
        "failures": failures,
        "warnings": warnings,
    }


def write_report(output_dir: Path, summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "d16_main_branch_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# D16 Main Branch Run Check",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- decision: `{summary['decision']}`",
        f"- test_accuracy: `{as_float(summary.get('test_accuracy')):.6f}`",
        f"- test_macro_f1: `{as_float(summary.get('test_macro_f1')):.6f}`",
        f"- predicted_classes: `{summary.get('predicted_classes')}`",
        f"- attention_diagnostics: `{summary.get('attention', {}).get('status')}`",
        f"- part_token_transformer_diagnostics: `{summary.get('part_token_transformer', {}).get('status')}`",
        f"- part_motif_query_diagnostics: `{summary.get('part_motif_query', {}).get('status')}`",
        f"- micro_motif_support_diagnostics: `{summary.get('micro_motif_support', {}).get('status')}`",
        "",
        "## Failures",
    ]
    failures = summary.get("failures") or []
    lines.extend([f"- {item}" for item in failures] if failures else ["- none"])
    warnings = summary.get("warnings") or []
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {item}" for item in warnings] if warnings else ["- none"])
    output_dir.joinpath("D16_MAIN_BRANCH_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_dir.joinpath("CHECK_D16R_A2_PART_TOKEN_TRANSFORMER.md").write_text(
        "\n".join(lines).replace("# D16 Main Branch Run Check", "# D16R-A2 Part-token Transformer Check") + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("CHECK_D16R_A3_PART_MOTIF_QUERY.md").write_text(
        "\n".join(lines).replace("# D16 Main Branch Run Check", "# D16R-A3 Part-Motif Query Check") + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("CHECK_D16R_A4_MICRO_MOTIF_SUPPORT.md").write_text(
        "\n".join(lines).replace("# D16 Main Branch Run Check", "# D16R-A4 Micro-Motif Support Check") + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("CHECK_D16R_A4B_NO_GLOBAL_MICRO.md").write_text(
        "\n".join(lines).replace("# D16 Main Branch Run Check", "# D16R-A4b No-Global-Micro Check") + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summary = check_run(Path(args.run_dir))
    write_report(Path(args.output_dir), summary)
    print(json.dumps(summary, indent=2))
    if summary["decision"] in {"D16_MAIN_BRANCH_CHECK_FAIL", "REJECT_RUN_COLLAPSE"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
