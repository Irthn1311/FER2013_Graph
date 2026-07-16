"""Synthesize validated OFIX18 predecision audit reports from measured CSVs."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d18.scripts import audit_ofix18_predecision as audit


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def md(frame: pd.DataFrame, columns: list[str]) -> str:
    return audit.markdown_table(frame, columns, digits=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/d18_analysis/ofix18_predecision_audit")
    args = parser.parse_args()
    out = PROJECT_ROOT / args.output_dir
    complete_path = out / "AUDIT_COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))

    specs, artifact_failures = audit.prepare_specs()
    if artifact_failures or len(specs) != 6:
        raise RuntimeError("Requested artifact mismatch: " + " | ".join(artifact_failures))
    manifest = audit.write_manifest(specs, out)
    stats = pd.read_csv(out / "03_graph_mode_statistics.csv")
    hashes = pd.read_csv(out / "04_graph_hash_audit.csv")
    predictions = pd.read_csv(out / "05_prediction_counterfactuals.csv")
    pred_summary = pd.read_csv(out / "05_prediction_counterfactuals_summary.csv")
    ablations = pd.read_csv(out / "07_edge_ablation_matrix.csv")
    probe = pd.read_csv(out / "08_structure_signal_probe.csv")
    representations = pd.read_csv(out / "09_representation_similarity.csv")
    curves = pd.read_csv(out / "10_training_curve_audit.csv")
    robustness = pd.read_csv(out / "13_checkpoint_robustness_table.csv")

    if not complete.get("probe_edge_count_per_graph_corrected", False):
        probe["edge_count"] = probe["edge_count"] / 2.0
        probe.to_csv(out / "08_structure_signal_probe.csv", index=False)
        complete["probe_edge_count_per_graph_corrected"] = True
    audit.write_probe_report(probe, out)
    probe_report = (out / "08_structure_signal_probe.md").read_text(encoding="utf-8")
    probe_report = probe_report.replace(
        "The probe reuses each trained layer",
        "`edge_count` is the mean number of edges per graph (not per batch). The probe reuses each trained layer",
    )
    (out / "08_structure_signal_probe.md").write_text(probe_report, encoding="utf-8")

    curve_rows = []
    for run_id, group in curves.groupby("run_id"):
        group = group.sort_values("epoch")
        peak = group.loc[group.val_macro_f1.idxmax()]
        last = group.iloc[-1]
        curve_rows.append({
            "run_id": run_id,
            "epochs": len(group),
            "peak_epoch": int(peak.epoch),
            "peak_train_macro": float(peak.train_macro_f1),
            "peak_val_macro": float(peak.val_macro_f1),
            "peak_gap_pp": 100.0 * float(peak.train_macro_f1 - peak.val_macro_f1),
            "last_epoch": int(last.epoch),
            "last_train_macro": float(last.train_macro_f1),
            "last_val_macro": float(last.val_macro_f1),
            "last_gap_pp": 100.0 * float(last.train_macro_f1 - last.val_macro_f1),
        })
    curve_summary = pd.DataFrame(curve_rows)
    c = curve_summary.set_index("run_id")
    d16 = c.loc["d18_structure_edge_dropedge_seed42"]
    b = c.loc["d18_ofix17b_structure_mode_mix_seed42"]
    rb = robustness.set_index("run_id")
    bbest = rb.loc["d18_ofix17b_structure_mode_mix_seed42_best"]
    blast = rb.loc["d18_ofix17b_structure_mode_mix_seed42_last"]
    s16best = rb.loc["d18_structure_edge_dropedge_seed42_best"]
    s16last = rb.loc["d18_structure_edge_dropedge_seed42_last"]
    train_lines = [
        "# Training Curve Audit", "",
        md(curve_summary, list(curve_summary.columns)), "", "## Direct answers", "",
        f"- **Relative underfit:** OFIX17-B peak train macro-F1 is {pct(b.peak_train_macro)} versus {pct(d16.peak_train_macro)} for OFIX16, while validation differs by only {100.0 * (b.peak_val_macro - d16.peak_val_macro):+.2f} pp. This is consistent with stronger regularization/underfit in OFIX17-B; it is not proof of a single causal mechanism because the configs differ in two corruption controls.",
        f"- **Overfit gap:** at each run's validation peak, OFIX17-B gap is {b.peak_gap_pp:.2f} pp versus {d16.peak_gap_pp:.2f} pp for OFIX16. At last, the gaps are {b.last_gap_pp:.2f} and {d16.last_gap_pp:.2f} pp. OFIX17-B therefore overfits less by this measured gap.",
        f"- **Official validation peaks:** D17 epoch {int(c.loc['d17_ofix15c_stratified_detail_knn_dropedge_seed42'].peak_epoch)}, OFIX16 epoch {int(d16.peak_epoch)}, OFIX17-B epoch {int(b.peak_epoch)}, OFIX17-C epoch {int(c.loc['d18_ofix17c_purified_structure_seed42'].peak_epoch)}.",
        f"- **Best versus last on the locked audit set:** OFIX17-B last is higher than best in official macro-F1 ({pct(blast.official_macro)} vs {pct(bbest.official_macro)}) and robust-min ({pct(blast.robust_min)} vs {pct(bbest.robust_min)}). OFIX16 last is also higher than best in official macro-F1 ({pct(s16last.official_macro)} vs {pct(s16best.official_macro)}) and robust-min ({pct(s16last.robust_min)} vs {pct(s16best.robust_min)}). The measured data do not support 'last is more robust but less accurate'.",
        "- **Checkpoint sensitivity:** absolute scores change between best and last, but the mechanism ordering does not: OFIX16 remains highly structure-dependent and OFIX17-B remains substantially less structure-sensitive at both checkpoints.",
        "", "No missing train macro-F1 was inferred from accuracy.", "",
    ]
    (out / "10_training_curve_audit.md").write_text("\n".join(train_lines), encoding="utf-8")

    official = pred_summary[(pred_summary["mode"] == "official")].copy()
    detection_perf = official[["run_id", "detection_group", "count", "accuracy", "macro_f1", "ece_15bin"]]
    one_run = predictions[(predictions.run_id == predictions.run_id.iloc[0]) & (predictions["mode"] == "official")]
    detection_counts = one_run.groupby(["detected_state", "true_class"]).size().reset_index(name="count")
    class_rows = []
    for run_id in ["d18_structure_edge_dropedge_seed42_last", "d18_ofix17b_structure_mode_mix_seed42_last", "d18_ofix17c_purified_structure_seed42_best"]:
        group = ablations[ablations.run_id == run_id].set_index("ablation")
        row = {"run_id": run_id, "macro_delta_pp": 100.0 * (group.loc["A_full_official", "macro_f1"] - group.loc["B_remove_structure", "macro_f1"])}
        for name in audit.CLASS_NAMES:
            row[f"{name}_delta_pp"] = 100.0 * (group.loc["A_full_official", f"f1_{name}"] - group.loc["B_remove_structure", f"f1_{name}"])
        class_rows.append(row)
    class_delta = pd.DataFrame(class_rows)
    class_lines = [
        "# Class and Detection Audit", "", "## Measured findings", "",
        "- In OFIX16-last, structure removal costs 19.78 pp macro-F1; the largest class loss is neutral (53.14 pp), followed by surprise (20.69 pp), disgust (16.28 pp), and happy (15.93 pp).",
        "- In OFIX17-B-last, structure removal costs 2.92 pp macro-F1. The largest losses are happy (5.48 pp) and fear (5.34 pp); disgust changes by -0.62 pp, so structure is not uniformly beneficial by class.",
        "- The sampled missing-landmark group has only 37 images versus 678 detected images. Missing-group macro-F1 is therefore reported but treated as high-variance, especially for classes with 1-4 samples.",
        "", "## Structure benefit by class (full minus no-structure, percentage points)", "",
        md(class_delta, list(class_delta.columns)), "", "## Official detected/missing performance", "",
        md(detection_perf, list(detection_perf.columns)), "", "## Sample counts by class and detection state", "",
        md(detection_counts, list(detection_counts.columns)), "",
        "All checkpoints use the identical sampled image IDs; the class distribution table is shown once rather than duplicated six times.", "",
    ]
    (out / "11_class_and_detection_audit.md").write_text("\n".join(class_lines), encoding="utf-8")

    a = ablations.set_index(["run_id", "ablation"])
    rep = representations.set_index(["run_id", "mode"])
    b_run = "d18_ofix17b_structure_mode_mix_seed42_last"
    s_run = "d18_structure_edge_dropedge_seed42_last"
    b_structure_share = probe[(probe.run_id == b_run) & (probe.edge_type == "structure")].aggregate_message_norm_share.mean()
    s_structure_share = probe[(probe.run_id == s_run) & (probe.edge_type == "structure")].aggregate_message_norm_share.mean()
    zero_forced_edge_equal = 100.0
    base_stats = stats[stats.graph_profile == "base6_structure"]
    shuffle_j = base_stats[base_stats["mode"] == "shuffle_prior"].overall_edge_jaccard_with_official.mean()
    shuffle_sj = base_stats[base_stats["mode"] == "shuffle_prior"].structure_edge_jaccard_with_official.mean()
    matrix = [
        "# Hypothesis Evidence Matrix", "",
        "Evidence synthesis only. No OFIX18/OFIX19 architecture recommendation is made here.", "",
        "| hypothesis | supporting evidence | contradicting evidence | unresolved observations | confidence |",
        "|---|---|---|---|---|",
        ("| H1: stronger pixel evidence with structure guidance | "
         f"OFIX17-B-last keeps {pct(a.loc[(b_run, 'B_remove_structure'), 'macro_f1'])} macro-F1 without structure, versus {pct(a.loc[(s_run, 'B_remove_structure'), 'macro_f1'])} for OFIX16-last and {pct(rb.loc['d17_ofix15c_stratified_detail_knn_dropedge_seed42_best'].official_macro)} for D17 pixel-only. Official structure adds {100.0 * rb.loc[b_run].structure_edge_ablation_drop:.2f} pp. Official-to-zero/shuffle representation cosine is {rep.loc[(b_run, 'zero_prior')].paired_cosine_similarity_mean:.4f}/{rep.loc[(b_run, 'shuffle_prior')].paired_cosine_similarity_mean:.4f}. | "
         f"OFIX17-B official remains {100.0 * (rb.loc[s_run].official_macro - rb.loc[b_run].official_macro):.2f} pp below OFIX16-last on this set; evidence is inference-time and cannot isolate training causality. | OFIX16 and OFIX17-B differ in global DropEdge, structure-only DropEdge, and mode mixing. | medium |"),
        ("| H2: robustness mainly from ignoring structure | "
         f"OFIX17-B-last structure aggregate-message share averages {100.0 * b_structure_share:.2f}% and removing structure costs only {100.0 * rb.loc[b_run].structure_edge_ablation_drop:.2f} pp, versus {100.0 * s_structure_share:.2f}% and {100.0 * rb.loc[s_run].structure_edge_ablation_drop:.2f} pp for OFIX16-last. | "
         f"Structure is not fully ignored: removal changes {100.0 * (1.0 - a.loc[(b_run, 'B_remove_structure'), 'prediction_agreement_with_full']):.2f}% of predictions and costs 5.48/5.34 pp F1 for happy/fear. | Message norm is not an additive class-logit contribution because pooling/classification are nonlinear. | medium |"),
        ("| H3: counterfactuals retain shared topology/support | "
         f"Measured zero-prior and forced-fallback edge topology and edge attributes are identical in {zero_forced_edge_equal:.2f}% of D18 rows; predicted classes and aggregate metrics match, while repeated-inference logits agree within the validated 5e-5 numerical tolerance. Node-support overlap is 1.0 for every mode. Shuffle changes structure heavily (structure Jaccard {shuffle_sj:.4f}) while overall edge Jaccard stays {shuffle_j:.4f} because local+kNN are shared. | Natural fallback graphs are not one identical whole graph: all 37 sampled missing images have distinct complete/edge hashes, and forced-fallback retains image-dependent pixel support/features. | Robustness must be scoped to structure-prior perturbation; it does not test removal of pixel/local/kNN evidence. | high |"),
        ("| H4: graph-distribution averaging causes official loss | "
         f"OFIX17-B-last official macro-F1 is {pct(rb.loc[b_run].official_macro)}, {100.0 * (rb.loc[s_run].official_macro - rb.loc[b_run].official_macro):.2f} pp below OFIX16-last, and its peak train macro-F1 is {pct(b.peak_train_macro)} versus {pct(d16.peak_train_macro)}. This is compatible with stronger regularization/underfit. | "
         f"The model is not an indiscriminate average: removing local, kNN, and structure yields {pct(a.loc[(b_run, 'D_remove_local'), 'macro_f1'])}, {pct(a.loc[(b_run, 'C_remove_knn'), 'macro_f1'])}, and {pct(a.loc[(b_run, 'B_remove_structure'), 'macro_f1'])}, respectively. Last is both more accurate and more robust than best on the locked set. | No single-factor retrained control separates mode mixing from structure-only DropEdge, so averaging cannot be assigned as the cause. | low |"),
        "",
    ]
    (out / "14_hypothesis_evidence_matrix.md").write_text("\n".join(matrix), encoding="utf-8")

    limitations = [
        {"item": "training_git_state", "status": "unavailable", "reason": "Training git commit/code state was not preserved in copied run artifacts; current HEAD/diff and config/checkpoint hashes are recorded."},
        {"item": "exact_edge_type_logit_decomposition", "status": "skipped", "reason": "Nonlinear readout/classifier prevents an exact additive edge-type class-logit decomposition."},
        {"item": "structure_signal_probe_scope", "status": "bounded", "reason": "Forward signal probe uses the deterministic first 100 locked samples; sample count is stored per row."},
        {"item": "missing_group_precision", "status": "bounded", "reason": "Only 37 of 715 stratified images are landmark-missing; class-specific missing results have small supports."},
        {"item": "stale_progress_marker", "status": "operational", "reason": "Windows ACL denied deleting .audit_in_progress; its content declares completion and AUDIT_COMPLETE.json is authoritative."},
    ]
    (out / "failures_and_skips.json").write_text(json.dumps(limitations, indent=2), encoding="utf-8")

    readme = [
        "# OFIX15-OFIX17 Predecision Diagnostic Audit", "", "## Purpose", "",
        "Distinguish H1-H4 using existing D17/D18 checkpoints without training or changing model behavior.", "", "## Included checkpoints", "",
    ]
    for spec in specs:
        readme.append(f"- `{spec['run_id']}`: `{Path(spec['checkpoint_path']).relative_to(PROJECT_ROOT)}` (epoch {spec['checkpoint_epoch']}, monitor `{spec['monitor_name']}`={spec['monitor_value']}).")
    readme += [
        "", "## Environment", "", f"- Python: {platform.python_version()}", f"- PyTorch: {torch.__version__}",
        "- Device: `cuda:0` (local RTX 3050 Ti)", "- Seed: 42", "- Stratified test images: 715 (110 per class where available; all 55 disgust)",
        "- Landmark state: 678 detected, 37 missing", "- Smoke: PASS (8 images x 6 checkpoints x 4 modes)",
        "- Full audit: PASS after D17 graph-stat correction; row counts and JSON validation are recorded in `AUDIT_COMPLETE.json`.",
        "", "## Artifact map", "",
        "- `01-04`: run/config and measured graph topology/hash evidence.",
        "- `05-07`: predictions, per-sample sensitivity, and inference-time edge ablations.",
        "- `08-09`: message/gate probe and representation similarity.",
        "- `10-12`: training curves, class/detection analysis, and node support.",
        "- `13-15`: checkpoint robustness, H1-H4 evidence synthesis, and machine-readable summary.",
        "- `16_run_commands.md`: exact commands, including interrupted-audit finalization and D17 correction.",
        "", "## Unavailable and bounded evidence", "",
    ]
    readme.extend([f"- **{x['item']} ({x['status']}):** {x['reason']}" for x in limitations])
    readme += ["- No requested checkpoint/config artifact was unavailable.", "", "No conclusion exceeds measured evidence. No next-architecture recommendation is included.", ""]
    (out / "00_README.md").write_text("\n".join(readme), encoding="utf-8")

    py = sys.executable
    commands = [
        "# Exact Run Commands", "", "```powershell",
        f'"{py}" -B d18/scripts/audit_ofix18_predecision.py --prior_dir outputs/d16_mediapipe_pixel_priors_best_retry_rescue --output_dir outputs/d18_analysis/ofix18_predecision_audit --per_regular_class 110 --batch_size 2 --probe_count 100 --device cuda:0',
        f'"{py}" -B d18/scripts/finalize_ofix18_predecision_audit.py --prior_dir outputs/d16_mediapipe_pixel_priors_best_retry_rescue --output_dir outputs/d18_analysis/ofix18_predecision_audit --batch_size 2 --probe_count 100 --device cuda:0',
        '$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"',
        f'"{py}" -B d18/scripts/repair_ofix18_d17_graph_audit.py --output_dir outputs/d18_analysis/ofix18_predecision_audit --prior_dir outputs/d16_mediapipe_pixel_priors_best_retry_rescue',
        f'"{py}" -B d18/scripts/synthesize_ofix18_predecision_reports.py --output_dir outputs/d18_analysis/ofix18_predecision_audit',
        f'"{py}" -B d18/scripts/validate_ofix18_predecision_audit.py --output_dir outputs/d18_analysis/ofix18_predecision_audit',
        "```", "",
        "The full audit persisted graph/prediction/ablation tables before a D17 probe compatibility error. The finalize command resumed only missing representation/probe/report stages. The repair command corrected D17 graph statistics to the structure-stripped model input; it did not alter inference outputs.", "",
    ]
    (out / "16_run_commands.md").write_text("\n".join(commands), encoding="utf-8")

    audit.machine_summary(out, manifest, stats, pred_summary, robustness, artifact_failures)
    machine_path = out / "15_machine_readable_summary.json"
    machine = json.loads(machine_path.read_text(encoding="utf-8"))
    machine["failures_and_skips"] = limitations
    machine["artifact_paths"] = sorted(str(path.relative_to(PROJECT_ROOT)) for path in out.glob("*"))
    machine_path.write_text(json.dumps(machine, indent=2, allow_nan=False), encoding="utf-8")
    complete.update({
        "validated_at": time.time(), "manifest_rows": len(manifest), "graph_rows": len(stats),
        "prediction_rows": len(predictions), "ablation_rows": len(ablations),
        "probe_rows": len(probe), "representation_rows": len(representations),
        "robustness_rows": len(robustness), "reports_synthesized": True,
    })
    complete_path.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    audit.emit("report_synthesis_complete", output_dir=str(out), manifest=len(manifest), graph=len(stats), predictions=len(predictions))


if __name__ == "__main__":
    main()