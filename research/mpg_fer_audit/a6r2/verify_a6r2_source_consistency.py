"""Programmatic Source Consistency Checker for MPG-FER A6-R2.

Verifies:
1. Composer numbers in final Markdown == Composer raw JSON.
2. Readout numbers in final Markdown == corrected readout sample CSV aggregation.
3. Routing numbers in final Markdown == per-sample routing CSV aggregation.
4. Hypothesis JSON == master result JSON.
5. Target JSON == final report decision.
6. Zero stale values across all artifacts.

Saves:
- a6r2_source_consistency_check.json
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import pandas as pd

from regenerate_a6r2_derived import (
    build_hypothesis_document,
    build_target_document,
)

AUDIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AUDIT_DIR.parents[2]


def main():
    print("Running A6-R2 Source Consistency Verification...", flush=True)

    master = json.load(open(AUDIT_DIR / "a6r2_master_results.json"))
    hyp = json.load(open(AUDIT_DIR / "a6r2_hypothesis_decisions.json"))
    target = json.load(open(AUDIT_DIR / "a6r2_final_target.json"))
    rd_res = json.load(open(AUDIT_DIR / "a6r2_readout_factorial_results.json"))
    rd_samples = pd.read_csv(AUDIT_DIR / "a6r2_readout_factorial_samples.csv")
    routing_samples = pd.read_csv(AUDIT_DIR / "a6r2_routing_per_sample.csv")
    routing_assoc = json.load(open(AUDIT_DIR / "a6r2_routing_association.json"))

    md_report = (AUDIT_DIR / "A6R2_READOUT_ROUTING_SOURCELOCK.md").read_text(encoding="utf-8")
    md_ledger = (AUDIT_DIR / "A6R2_CORRECTION_LEDGER.md").read_text(encoding="utf-8")

    checks = []

    # Check 1: Composer numbers in Master == Markdown
    comp_a_what = master["composer_functional_swaps"]["v21_to_v22"]["what_only_rescue_rate"]
    comp_b_what = master["composer_functional_swaps"]["v22_to_v21"]["what_only_rescue_rate"]
    c1 = f"{comp_a_what*100:.2f}%" in md_report or f"{comp_a_what*100:.1f}%" in md_report
    c2 = f"{comp_b_what*100:.2f}%" in md_report or f"{comp_b_what*100:.1f}%" in md_report
    checks.append({"check": "composer_what_in_markdown", "pass": bool(c1 and c2), "details": f"Dir A: {comp_a_what*100:.2f}%, Dir B: {comp_b_what*100:.2f}%"})

    # Check 2: Readout numbers from CSV aggregation == Master == Markdown
    dir_a_csv = rd_samples[rd_samples["direction"] == "dir_A"]
    dir_b_csv = rd_samples[rd_samples["direction"] == "dir_B"]

    f00_a_cnt = int(dir_a_csv["f00_correct"].sum())
    f01_a_cnt = int(dir_a_csv["f01_correct"].sum())
    f10_a_cnt = int(dir_a_csv["f10_correct"].sum())
    f11_a_cnt = int(dir_a_csv["f11_correct"].sum())

    f00_b_cnt = int(dir_b_csv["f00_correct"].sum())
    f01_b_cnt = int(dir_b_csv["f01_correct"].sum())
    f10_b_cnt = int(dir_b_csv["f10_correct"].sum())
    f11_b_cnt = int(dir_b_csv["f11_correct"].sum())

    rd_match_master = (
        f00_a_cnt == rd_res["dir_A"]["all"]["f00_baseline"]["correct_count"] == 0 and
        f01_a_cnt == rd_res["dir_A"]["all"]["f01_readout_operator_only"]["correct_count"] == 129 and
        f10_a_cnt == rd_res["dir_A"]["all"]["f10_node_states_only"]["correct_count"] == 423 and
        f11_a_cnt == rd_res["dir_A"]["all"]["f11_both_swapped_s7"]["correct_count"] == 530 and
        f00_b_cnt == rd_res["dir_B"]["all"]["f00_baseline"]["correct_count"] == 0 and
        f01_b_cnt == rd_res["dir_B"]["all"]["f01_readout_operator_only"]["correct_count"] == 132 and
        f10_b_cnt == rd_res["dir_B"]["all"]["f10_node_states_only"]["correct_count"] == 448 and
        f11_b_cnt == rd_res["dir_B"]["all"]["f11_both_swapped_s7"]["correct_count"] == 538
    )
    checks.append({"check": "readout_csv_equals_master_json", "pass": bool(rd_match_master), "details": "All F00/F01/F10/F11 sample counts match exact CSV aggregation"})

    # Check 3: Readout numbers in Markdown
    rd_md_pass = (
        ("0.00%" in md_report and "(0 / 593)" in md_report and "(0 / 604)" in md_report) and
        ("21.75%" in md_report and "(129 / 593)" in md_report) and
        ("21.85%" in md_report and "(132 / 604)" in md_report) and
        ("71.33%" in md_report and "(423 / 593)" in md_report) and
        ("74.17%" in md_report and "(448 / 604)" in md_report) and
        ("89.38%" in md_report and "(530 / 593)" in md_report) and
        ("89.07%" in md_report and "(538 / 604)" in md_report)
    )
    checks.append({"check": "readout_numbers_in_markdown", "pass": bool(rd_md_pass), "details": "Markdown tables exactly match master results"})

    # Check 4: Routing numbers from CSV == JSON == Markdown
    r_j1_mean = float(routing_samples["jaccard_layer_1"].mean())
    r_j4_mean = float(routing_samples["jaccard_layer_4"].mean())
    r_match_json = (
        abs(r_j1_mean - routing_assoc["layer_statistics"]["layer_1"]["per_sample_jaccard_distribution"]["mean"]) < 1e-6 and
        abs(r_j4_mean - routing_assoc["layer_statistics"]["layer_4"]["per_sample_jaccard_distribution"]["mean"]) < 1e-6
    )
    checks.append({"check": "routing_csv_equals_json", "pass": bool(r_match_json), "details": f"L1 mean: {r_j1_mean:.4f}, L4 mean: {r_j4_mean:.4f}"})

    # Check 5: Standalone raw result JSONs are exact master projections.
    raw_projection_match = (
        rd_res == master["readout_factorial_repaired"]
        and routing_assoc == master["routing_sample_associations"]
    )
    checks.append({
        "check": "standalone_numeric_results_equal_master",
        "pass": bool(raw_projection_match),
        "details": "Readout factorial and routing association JSONs exactly equal their master sections",
    })

    # Check 6: Entire hypothesis artifact, including every numerical evidence
    # field and generated finding, equals a fresh projection from the master.
    expected_hyp = build_hypothesis_document(master)
    hyp_match = hyp == expected_hyp
    checks.append({
        "check": "hypothesis_json_exactly_regenerated_from_master",
        "pass": bool(hyp_match),
        "details": "Statuses, structured numerical evidence, and findings equal the master-derived document",
    })

    # Check 7: Entire target artifact is an exact master projection, and its
    # categorical decision remains consistent with the source-lock report.
    expected_target = build_target_document(master)
    target_match = (
        target == expected_target and
        target["final_mechanistic_decision"] == master["master_conclusions"]["final_target_class"] == "EARLY_DEPTH_GENERALIZATION_TARGET" and
        target["verdict"] == master["master_conclusions"]["final_verdict"] == "A6R2_COMPLETE_V23_TARGET_READY" and
        "EARLY_DEPTH_GENERALIZATION_TARGET" in md_report and
        "A6R2_COMPLETE_V23_TARGET_READY" in md_report
    )
    checks.append({
        "check": "target_json_exactly_regenerated_from_master",
        "pass": bool(target_match),
        "details": "Target numerical evidence, decision, and verdict equal the master-derived document",
    })

    # Check 8: Every routing-association value displayed in the source-lock
    # table comes from the authoritative master, rather than an older summary.
    routing_md_tokens = []
    for layer_number in range(1, 6):
        layer = master["routing_sample_associations"]["layer_statistics"][f"layer_{layer_number}"]
        distribution = layer["per_sample_jaccard_distribution"]
        routing_md_tokens.extend([
            f"{distribution['mean']:.3f}",
            f"{distribution['std']:.3f}",
            f"{distribution['p10']:.3f}",
            f"{distribution['median']:.3f}",
            f"{distribution['p90']:.3f}",
        ])
        for direction in ("v21_correct_v22_wrong", "v22_correct_v21_wrong"):
            values = layer["directions"][direction]
            representation = values[
                "spearman_routing_divergence_vs_representation_cosine_dist"
            ]
            rescue = values["point_biserial_routing_divergence_vs_rescue"]
            routing_md_tokens.extend([
                f"{representation['estimate']:+.3f}",
                f"{rescue['estimate']:+.3f}",
                f"p = {rescue['p_value']:.3f}",
            ])
    routing_markdown_match = all(token in md_report for token in routing_md_tokens)
    checks.append({
        "check": "routing_markdown_contains_master_numeric_evidence",
        "pass": bool(routing_markdown_match),
        "details": f"Checked {len(routing_md_tokens)} formatted routing values from all five layers and both directions",
    })

    # Check 9: No known stale claims remain in current A6-R2 derived files.
    stale_patterns = [
        r"\b15\.01\b", r"\b13\.91\b", r"\b34\.74\b", r"\b34\.27\b",
        r"\b20\.07\b", r"\b20\.70\b", r"\b71\.13\b", r"\b72\.85\b",
        r"\b75\.04%", r"74\.2%-75\.0%", r"\+0\.12 to \+0\.22",
        r"-0\.05\$ to \$\+0\.02", r"p < 1e-7 across Layers 1-5"
    ]
    stale_hits = []
    for f_p in AUDIT_DIR.glob("*"):
        if not f_p.is_file():
            continue
        if f_p.name in ["A6R2_CORRECTION_LEDGER.md", "verify_a6r2_source_consistency.py", "a6r2_source_consistency_check.json"]:
            continue  # Ledger documents historical stale numbers for audit trail
        text = f_p.read_text(encoding="utf-8", errors="ignore")
        for pat in stale_patterns:
            if re.search(pat, text):
                stale_hits.append(f"{f_p.name} matched {pat}")

    no_stale_pass = (len(stale_hits) == 0)
    checks.append({"check": "zero_stale_numbers_in_derived_artifacts", "pass": bool(no_stale_pass), "details": f"Stale hits: {stale_hits}"})

    all_passed = all(c["pass"] for c in checks)
    overall_status = "CONSISTENCY_PASS" if all_passed else "CONSISTENCY_FAIL"

    summary_doc = {
        "overall_status": overall_status,
        "checks_total": len(checks),
        "checks_passed": sum(1 for c in checks if c["pass"]),
        "all_checks_passed": all_passed,
        "checks": checks,
    }

    (AUDIT_DIR / "a6r2_source_consistency_check.json").write_text(json.dumps(summary_doc, indent=2), encoding="utf-8")
    print(f"\nFinal Consistency Result: {overall_status} ({summary_doc['checks_passed']}/{len(checks)} passed)")
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['details']}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
