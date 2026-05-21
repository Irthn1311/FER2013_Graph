"""Build the final D13 pipeline synthesis report.

This report is a synthesis artifact only. It does not train, modify models, or
upgrade diagnostic slot candidates into motif, semantic-region, or causal
claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FINAL_DECISION = "D13_FINAL_CLASSIFICATION_DIAGNOSTIC_SUCCESS_BUT_EVIDENCE_CLAIM_REJECTED"


def _read_csv_row(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception:
        return {}


def _float(value: Any, fallback: float) -> float:
    try:
        out = float(value)
        return out
    except Exception:
        return fallback


def _format(value: float | str) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _md_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "No data."
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _metrics(project_root: Path) -> Dict[str, Any]:
    d13d_l005 = _read_csv_row(project_root / "outputs/d13d_evidence/d13c_m16_supcon_l005/d13d_deletion_summary.csv")
    d13d_ce = _read_csv_row(project_root / "outputs/d13d_evidence/d13c_m16_ce_continue/d13d_deletion_summary.csv")
    d13d_l005_stability = _read_csv_row(project_root / "outputs/d13d_evidence/d13c_m16_supcon_l005/d13d_stability_summary.csv")
    d13d_ce_stability = _read_csv_row(project_root / "outputs/d13d_evidence/d13c_m16_ce_continue/d13d_stability_summary.csv")
    return {
        "stage49_decision": "DOCUMENT_AS_VISUALLY_UNRELIABLE_NEAR_MISS_AND_STOP_STAGE5_PATH",
        "d13a_k144_macro": 0.5829,
        "d13a_k144_acc": 0.6166,
        "d13a_k256_macro": 0.5866,
        "d13a_k256_acc": 0.6227,
        "d13b_m16_macro": 0.6187,
        "d13b_m16_acc": 0.6328,
        "d13c_ce_macro": 0.6222,
        "d13c_ce_acc": 0.6358,
        "d13c_l005_macro": 0.6277,
        "d13c_l005_acc": 0.6420,
        "d13c_m8_macro": 0.6364,
        "d13c_m8_acc": 0.6481,
        "d13d_l005_top1_gap": _float(d13d_l005.get("top1_vs_random1_gap"), -0.0036),
        "d13d_l005_top3_gap": _float(d13d_l005.get("top3_vs_random3_gap"), -0.0033),
        "d13d_l005_top1_low_gap": _float(d13d_l005.get("top1_vs_low_gap"), -0.0051),
        "d13d_l005_pred_consistency": _float(d13d_l005_stability.get("avg_prediction_consistency"), 0.8794),
        "d13d_l005_slot_similarity": _float(d13d_l005_stability.get("avg_slot_map_similarity"), 0.8236),
        "d13d_l005_decision": str(d13d_l005.get("decision", "D13D_EVIDENCE_FAIL_SHORTCUT_DOMINATED")),
        "d13d_ce_top1_gap": _float(d13d_ce.get("top1_vs_random1_gap"), 0.0029),
        "d13d_ce_top3_gap": _float(d13d_ce.get("top3_vs_random3_gap"), 0.0143),
        "d13d_ce_pred_consistency": _float(d13d_ce_stability.get("avg_prediction_consistency"), 0.9111),
        "d13d_ce_decision": str(d13d_ce.get("decision", "D13D_EVIDENCE_FAIL_SHORTCUT_DOMINATED")),
    }


def _summary_rows(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "stage": "Stage 4.9",
            "run_name": "hybrid_slic_near_miss",
            "role": "low-level selector near-miss",
            "test_macro_f1": "",
            "test_acc": "",
            "main_gain": "region-like hints but gap below gate",
            "visual_status": "unreliable",
            "evidence_status": "failed visual/evidence gate",
            "decision": m["stage49_decision"],
            "claim_level": "negative finding only",
        },
        {
            "stage": "D13A K144",
            "run_name": "d13a_k144_ep100",
            "role": "visual diagnostic base",
            "test_macro_f1": m["d13a_k144_macro"],
            "test_acc": m["d13a_k144_acc"],
            "main_gain": "pure GNN reduction works",
            "visual_status": "caution",
            "evidence_status": "not tested as evidence",
            "decision": "USE_AS_D13B_VISUAL_BASE_WITH_CAUTION",
            "claim_level": "classification/reduction diagnostic",
        },
        {
            "stage": "D13A K256",
            "run_name": "d13a_k256_ep100",
            "role": "score candidate",
            "test_macro_f1": m["d13a_k256_macro"],
            "test_acc": m["d13a_k256_acc"],
            "main_gain": "+0.0037 macro-F1 vs K144",
            "visual_status": "weak visual pooling",
            "evidence_status": "not evidence-supported",
            "decision": "KEEP_AS_SCORE_REFERENCE_NOT_VISUAL_BASE",
            "claim_level": "score reference",
        },
        {
            "stage": "D13B",
            "run_name": "d13b_k144_m16_deep_readout",
            "role": "slot bottleneck diagnostic",
            "test_macro_f1": m["d13b_m16_macro"],
            "test_acc": m["d13b_m16_acc"],
            "main_gain": "+0.0358 macro-F1 vs D13A K144",
            "visual_status": "visual audit pass 56/49/0",
            "evidence_status": "not evidence-tested",
            "decision": "USE_M16_DEEP_READOUT_FOR_D13C_DIAGNOSTIC",
            "claim_level": "slot-candidate diagnostic",
        },
        {
            "stage": "D13C CE",
            "run_name": "d13c_m16_ce_continue",
            "role": "CE continuation control",
            "test_macro_f1": m["d13c_ce_macro"],
            "test_acc": m["d13c_ce_acc"],
            "main_gain": "+0.0035 macro-F1 vs D13B",
            "visual_status": "post visual audit pass",
            "evidence_status": "D13D gate failed",
            "decision": "CE_CONTINUATION_CONTROL",
            "claim_level": "classification control",
        },
        {
            "stage": "D13C",
            "run_name": "d13c_m16_supcon_l005",
            "role": "selected D13C diagnostic candidate",
            "test_macro_f1": m["d13c_l005_macro"],
            "test_acc": m["d13c_l005_acc"],
            "main_gain": "+0.0055 macro-F1 vs CE-only",
            "visual_status": "post visual audit pass 58/47/0",
            "evidence_status": "D13D weak",
            "decision": "USE_D13C_M16_SUPCON_L005_AS_DIAGNOSTIC_CANDIDATE",
            "claim_level": "classification with diagnostic attention",
        },
        {
            "stage": "D13D",
            "run_name": "d13c_m16_supcon_l005_evidence",
            "role": "evidence diagnostic",
            "test_macro_f1": "",
            "test_acc": "",
            "main_gain": "top deletion not above random/low",
            "visual_status": "visual audit remained acceptable",
            "evidence_status": "failed shortcut/evidence gate",
            "decision": "D13D_EVIDENCE_WEAK_KEEP_AS_ATTENTION_DIAGNOSTIC_ONLY",
            "claim_level": "attention diagnostic only",
        },
    ]


def _stage_decisions(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "stage": "Stage 4.9",
            "input_question": "Can low-level selector support motif path?",
            "decision": m["stage49_decision"],
            "reason": "Near-miss region-like signal, but visual/evidence gate failed.",
            "next_stage_allowed": "No old Stage 5 motif path.",
            "forbidden_claims": "motif discovery; semantic facial region; causal evidence",
        },
        {
            "stage": "D13A",
            "input_question": "Can pure GNN hierarchical reduction train?",
            "decision": "PASS_USE_K144_AS_VISUAL_BASE_WITH_CAUTION",
            "reason": "K256 scored higher, but K144 was more appropriate as visual diagnostic base.",
            "next_stage_allowed": "D13B slot bottleneck diagnostic.",
            "forbidden_claims": "semantic region claim",
        },
        {
            "stage": "D13B",
            "input_question": "Does slot bottleneck improve classification and remain visually auditable?",
            "decision": "USE_M16_DEEP_READOUT_FOR_D13C_DIAGNOSTIC",
            "reason": "Best D13B score and visual audit passed.",
            "next_stage_allowed": "D13C image-level SupCon diagnostic.",
            "forbidden_claims": "motif claim; causal evidence",
        },
        {
            "stage": "D13C",
            "input_question": "Does image-level SupCon improve beyond CE-only without visual degradation?",
            "decision": "USE_D13C_M16_SUPCON_L005_AS_DIAGNOSTIC_CANDIDATE",
            "reason": "l005 improved macro-F1 by +0.0055 vs CE-only and post visual audit passed.",
            "next_stage_allowed": "D13D evidence diagnostics.",
            "forbidden_claims": "evidence claim before deletion/control/stability",
        },
        {
            "stage": "D13D",
            "input_question": "Do top slots behave as evidence under deletion/control/stability?",
            "decision": "D13D_EVIDENCE_WEAK_KEEP_AS_ATTENTION_DIAGNOSTIC_ONLY",
            "reason": "l005 improved score, not evidence; top deletion failed against random/low controls.",
            "next_stage_allowed": "Evidence-aware training or stronger diagnostics only.",
            "forbidden_claims": "motif discovery; semantic region discovery; causal confirmation; full interpretability",
        },
    ]


def _key_metrics(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    metrics = [
        ("accuracy", "D13A K144 macro-F1", "d13a_k144_ep100", m["d13a_k144_macro"], "visual base score"),
        ("accuracy", "D13A K256 macro-F1", "d13a_k256_ep100", m["d13a_k256_macro"], "best D13A score but weak visual pooling"),
        ("accuracy", "D13B M16 macro-F1", "d13b_k144_m16_deep_readout", m["d13b_m16_macro"], "slot bottleneck gain"),
        ("accuracy", "D13C CE macro-F1", "d13c_m16_ce_continue", m["d13c_ce_macro"], "fine-tune control"),
        ("accuracy", "D13C l005 macro-F1", "d13c_m16_supcon_l005", m["d13c_l005_macro"], "selected classification candidate"),
        ("accuracy", "M8 control macro-F1", "d13c_m8_supcon_l002_control", m["d13c_m8_macro"], "separate compact control, not replacement"),
        ("visual", "D13B visual audit", "d13b_k144_m16_deep_readout", "56/49/0", "PASS/PARTIAL/FAIL"),
        ("visual", "D13C l005 post visual audit", "d13c_m16_supcon_l005", "58/47/0", "PASS/PARTIAL/FAIL"),
        ("evidence", "l005 top1 vs random1 gap", "d13c_m16_supcon_l005", m["d13d_l005_top1_gap"], "negative evidence diagnostic gap"),
        ("evidence", "l005 top3 vs random3 gap", "d13c_m16_supcon_l005", m["d13d_l005_top3_gap"], "negative evidence diagnostic gap"),
        ("evidence", "CE top1 vs random1 gap", "d13c_m16_ce_continue", m["d13d_ce_top1_gap"], "CE stronger than l005 in D13D"),
        ("stability", "l005 prediction consistency", "d13c_m16_supcon_l005", m["d13d_l005_pred_consistency"], "weak augmentation consistency"),
        ("stability", "l005 slot map similarity", "d13c_m16_supcon_l005", m["d13d_l005_slot_similarity"], "weak augmentation slot stability"),
    ]
    return [
        {
            "metric_group": group,
            "metric_name": name,
            "run_name": run,
            "value": value,
            "interpretation": interp,
        }
        for group, name, run, value, interp in metrics
    ]


def _write_report(output_dir: Path, m: Dict[str, Any], summary_rows: List[Dict[str, Any]]) -> None:
    table_columns = ["Stage", "selected run", "macro-F1", "acc", "visual gate", "evidence gate", "decision"]
    main_rows = [
        {
            "Stage": "D13A K144",
            "selected run": "d13a_k144_ep100",
            "macro-F1": m["d13a_k144_macro"],
            "acc": m["d13a_k144_acc"],
            "visual gate": "caution",
            "evidence gate": "not tested",
            "decision": "visual base for D13B",
        },
        {
            "Stage": "D13A K256",
            "selected run": "d13a_k256_ep100",
            "macro-F1": m["d13a_k256_macro"],
            "acc": m["d13a_k256_acc"],
            "visual gate": "weak",
            "evidence gate": "not accepted",
            "decision": "score reference only",
        },
        {
            "Stage": "D13B M16",
            "selected run": "d13b_k144_m16_deep_readout",
            "macro-F1": m["d13b_m16_macro"],
            "acc": m["d13b_m16_acc"],
            "visual gate": "pass",
            "evidence gate": "not tested",
            "decision": "D13C diagnostic base",
        },
        {
            "Stage": "D13C l005",
            "selected run": "d13c_m16_supcon_l005",
            "macro-F1": m["d13c_l005_macro"],
            "acc": m["d13c_l005_acc"],
            "visual gate": "pass",
            "evidence gate": "weak",
            "decision": "classification diagnostic candidate",
        },
        {
            "Stage": "D13D l005 evidence",
            "selected run": "d13c_m16_supcon_l005",
            "macro-F1": "",
            "acc": "",
            "visual gate": "previous pass",
            "evidence gate": "fail",
            "decision": "attention diagnostic only",
        },
    ]
    lines = [
        "# D13 Pipeline Final Analysis Report",
        "",
        "## 1. Research Goal",
        "The D13 branch tests a pure GNN hierarchical pipeline for FER-2013: pixel graph to region reduction, slot bottleneck, and image-level SupCon representation. The branch evaluates classification accuracy alongside visual and evidence reliability. It does not use a CNN teacher, and diagnostic slots are not upgraded into motif claims unless evidence diagnostics support that step.",
        "",
        "## 2. Stage 4.9 Negative Finding",
        "The hybrid SLIC near-miss selector showed some region-like signals, but the gap against random controls stayed below gate and visual audit did not provide enough facial evidence. The old Stage 5 path remains locked. This shows that heuristic low-level region selection is insufficient for motif-level claims.",
        "",
        "## 3. D13A: Hierarchical Reduction",
        f"D13A showed that pure GNN hierarchical reduction can train. K256 ep100 had the best D13A score (macro-F1 {m['d13a_k256_macro']:.4f}, acc {m['d13a_k256_acc']:.4f}), but visual pooling was too soft and mouth-heavy. K144 ep100 (macro-F1 {m['d13a_k144_macro']:.4f}, acc {m['d13a_k144_acc']:.4f}) was selected as the visual diagnostic base for D13B with caution. K256 is a score candidate; K144 is the visual diagnostic base. The reduced regions are not semantic regions.",
        "",
        "## 4. D13B: Slot Bottleneck Diagnostic",
        f"D13B slot bottleneck diagnostics passed without collapse. The selected candidate was `d13b_k144_m16_deep_readout` with macro-F1 {m['d13b_m16_macro']:.4f} and acc {m['d13b_m16_acc']:.4f}. The visual slot audit passed with 56 PASS, 49 PARTIAL, and 0 FAIL, so it was allowed into D13C diagnostic work. These slots remain slot candidates, not motifs.",
        "",
        "## 5. D13C: Image-level SupCon Diagnostic",
        f"D13C was judged against CE-only continuation. CE-only reached macro-F1 {m['d13c_ce_macro']:.4f}; l005 reached macro-F1 {m['d13c_l005_macro']:.4f}, a +0.0055 gain over CE-only. Post-D13C visual audit passed with 58 PASS, 47 PARTIAL, and 0 FAIL, with no visual degradation against CE-only. M8 control scored strongly but remains a separate compact-control branch, not an automatic replacement for M16.",
        "",
        "## 6. D13D: Evidence Diagnostic",
        f"D13D deletion/control/stability diagnostics did not support strong evidence claims. For l005, top1 vs random1 gap was {m['d13d_l005_top1_gap']:.4f}, top3 vs random3 gap was {m['d13d_l005_top3_gap']:.4f}, and top1 vs low-importance gap was {m['d13d_l005_top1_low_gap']:.4f}. CE-only had stronger deletion gaps but still failed the shortcut/evidence gate. The comparison decision was `SUPCON_IMPROVES_SCORE_NOT_EVIDENCE`. Therefore slots remain attention/slot-candidate diagnostics.",
        "",
        "## 7. Main Results Table",
        _md_table(main_rows, table_columns),
        "",
        "## 8. Main Scientific Conclusions",
        "1. Pure GNN hierarchical reduction is feasible for FER-2013 classification.",
        "2. A D13B slot bottleneck improves classification over the D13A visual base.",
        "3. Image-level SupCon l005 improves diagnostic classification beyond CE-only continuation.",
        "4. Visual slot reliability can remain acceptable after SupCon.",
        "5. Deletion/control/stability diagnostics fail to support evidence-like slot behavior.",
        "6. Accuracy and visually traceable attention are not equivalent to evidence or motif validity.",
        "7. The final claim must stay limited to classification and diagnostic attention.",
        "",
        "## 9. What Can Be Claimed",
        "Allowed claims:",
        "- Pure GNN hierarchical reduction works as a trainable classification pipeline.",
        "- Slot bottleneck diagnostics improve classification.",
        "- Image-level SupCon l005 improves diagnostic performance over CE-only.",
        "- Visual slot maps are traceable enough for diagnostic inspection.",
        "- Evidence diagnostics do not support motif or causal claims.",
        "",
        "Forbidden claims:",
        "- A motif was discovered.",
        "- A semantic facial region was discovered.",
        "- Causal evidence was confirmed.",
        "- Top slots are true facial expression motifs.",
        "- SupCon improves evidence reliability.",
        "",
        "## 10. Limitations",
        "- FER-2013 is low resolution.",
        "- AI-assisted visual audit is heuristic, not human landmark annotation.",
        "- Slot deletion is an imperfect intervention.",
        "- Slot importance proxy is imperfect.",
        "- Center and mouth shortcut risks remain.",
        "- D13D audit/evidence diagnostics used 105 samples.",
        "- No external dataset validation has been run.",
        "- No landmark or face-aligned ground truth is available.",
        "",
        "## 11. Recommended Next Work",
        "- Evidence-aware training: deletion-consistency objectives, shortcut decorrelation, and slot sparsity/orthogonality with an evidence objective.",
        "- Stronger diagnostics: image-region deletion, counterfactual masking, and landmark-aware evaluation if labels are available.",
        "- External validation on RAF-DB or an AffectNet subset if feasible.",
        "- Human visual audit with multiple annotators.",
        "- Final packaging as `D13Final = K144 + M16 deep readout + SupCon l005`, labeled as a classification model with diagnostic attention, not a motif model.",
        "",
        "## 12. Final Decision",
        FINAL_DECISION,
        "",
    ]
    (output_dir / "D13_PIPELINE_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_negative_findings(output_dir: Path) -> None:
    lines = [
        "# D13 Negative Findings",
        "",
        "1. Stage 4.9 selector near-miss failed the visual/evidence gate despite region-like hints.",
        "2. D13A K256 had the best D13A score but weak visual pooling interpretability.",
        "3. Anneal-style area diversity improvements did not solve assignment interpretability.",
        "4. D13D showed that SupCon improves score, not evidence diagnostics.",
        "5. CE-only had stronger deletion gaps than l005, but still failed the D13D evidence gate.",
        "6. Attention and slot maps must not be upgraded into motif claims.",
        "",
    ]
    (output_dir / "D13_NEGATIVE_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def _write_limitations(output_dir: Path) -> None:
    lines = [
        "# D13 Limitations and Next Work",
        "",
        "## What Is Solved",
        "- A pure GNN hierarchical FER-2013 pipeline was trained and improved through D13B/D13C.",
        "- The selected D13C l005 candidate improves classification over CE-only.",
        "- Visual diagnostic auditing is reproducible and paired against controls.",
        "",
        "## What Remains Unsolved",
        "- D13D evidence diagnostics do not support evidence-like slot behavior.",
        "- Shortcut risk remains, especially around center/mouth controls.",
        "- Slot deletion is not a perfect intervention.",
        "- No external dataset or landmark-grounded validation has been performed.",
        "",
        "## Recommended Experiments",
        "- Evidence-aware training with deletion-consistency and shortcut decorrelation.",
        "- Region-level deletion and counterfactual masking diagnostics.",
        "- Landmark-aware or human-reviewed evaluation.",
        "- External validation on a second FER dataset.",
        "",
        "## Forbidden Claims",
        "- Motif discovery.",
        "- Semantic facial-region discovery.",
        "- Causal evidence confirmation.",
        "- Full interpretability.",
        "",
    ]
    (output_dir / "D13_LIMITATIONS_AND_NEXT_WORK.md").write_text("\n".join(lines), encoding="utf-8")


def _write_figures(output_dir: Path, m: Dict[str, Any]) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = ["D13A K144", "D13B M16", "D13C CE", "D13C l005", "M8 control"]
    macro = [m["d13a_k144_macro"], m["d13b_m16_macro"], m["d13c_ce_macro"], m["d13c_l005_macro"], m["d13c_m8_macro"]]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, macro)
    plt.ylabel("test macro-F1")
    plt.title("D13 macro-F1 progression")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "d13_macro_f1_progression.png", dpi=150)
    plt.close()

    stages = ["D13A visual", "D13B visual", "D13C visual", "D13D evidence"]
    values = [0.5, 1.0, 1.0, 0.0]
    plt.figure(figsize=(8, 4))
    plt.bar(stages, values)
    plt.ylim(0, 1.1)
    plt.ylabel("gate status score")
    plt.title("Accuracy path vs evidence gate summary")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "d13_accuracy_vs_evidence_summary.png", dpi=150)
    plt.close()

    flow_labels = ["Stage 4.9\nstop", "D13A\npass", "D13B\npass", "D13C\nscore pass", "D13D\nevidence fail"]
    x = list(range(len(flow_labels)))
    y = [0, 1, 2, 3, 2]
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker="o")
    for xi, yi, label in zip(x, y, flow_labels):
        plt.text(xi, yi + 0.08, label, ha="center", va="bottom", fontsize=9)
    plt.xticks([])
    plt.yticks([])
    plt.title("D13 stage decision flow")
    plt.tight_layout()
    plt.savefig(fig_dir / "d13_stage_decision_flow.png", dpi=150)
    plt.close()


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    m = _metrics(project_root)
    summary_rows = _summary_rows(m)
    decisions = _stage_decisions(m)
    key_metrics = _key_metrics(m)

    pd.DataFrame(summary_rows).to_csv(output_dir / "D13_PIPELINE_FINAL_SUMMARY.csv", index=False)
    pd.DataFrame(decisions).to_csv(output_dir / "D13_STAGE_DECISIONS.csv", index=False)
    pd.DataFrame(key_metrics).to_csv(output_dir / "D13_KEY_METRICS.csv", index=False)
    _write_report(output_dir, m, summary_rows)
    _write_negative_findings(output_dir)
    _write_limitations(output_dir)
    _write_figures(output_dir, m)
    manifest = {
        "output_dir": str(output_dir),
        "final_decision": FINAL_DECISION,
        "files": [
            "D13_PIPELINE_FINAL_REPORT.md",
            "D13_PIPELINE_FINAL_SUMMARY.csv",
            "D13_STAGE_DECISIONS.csv",
            "D13_KEY_METRICS.csv",
            "D13_NEGATIVE_FINDINGS.md",
            "D13_LIMITATIONS_AND_NEXT_WORK.md",
            "figures/d13_macro_f1_progression.png",
            "figures/d13_stage_decision_flow.png",
            "figures/d13_accuracy_vs_evidence_summary.png",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final D13 pipeline synthesis artifacts.")
    parser.add_argument("--output_dir", default="outputs/d13_final_analysis")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
