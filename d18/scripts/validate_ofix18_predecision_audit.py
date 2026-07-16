"""Validate the completed OFIX18 predecision audit package."""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/d18_analysis/ofix18_predecision_audit")
    args = parser.parse_args()
    out = PROJECT_ROOT / args.output_dir
    required = [
        "00_README.md", "01_run_manifest.csv", "02_exact_config_diff.md",
        "03_graph_mode_statistics.csv", "03_graph_mode_statistics_summary.md",
        "04_graph_hash_audit.csv", "05_prediction_counterfactuals.csv",
        "05_prediction_counterfactuals_summary.csv", "06_per_sample_sensitivity.csv",
        "07_edge_ablation_matrix.csv", "08_structure_signal_probe.md",
        "08_structure_signal_probe.csv", "09_representation_similarity.csv",
        "10_training_curve_audit.csv", "10_training_curve_audit.md",
        "11_class_and_detection_audit.md", "12_node_support_information.md",
        "12_node_support_information.csv", "13_checkpoint_robustness_table.csv",
        "14_hypothesis_evidence_matrix.md", "15_machine_readable_summary.json",
        "16_run_commands.md", "AUDIT_COMPLETE.json", "failures_and_skips.json",
    ]
    missing = [name for name in required if not (out / name).is_file() or (out / name).stat().st_size == 0]
    if missing:
        raise AssertionError(f"Missing/empty artifacts: {missing}")

    manifest = pd.read_csv(out / "01_run_manifest.csv")
    stats = pd.read_csv(out / "03_graph_mode_statistics.csv")
    hashes = pd.read_csv(out / "04_graph_hash_audit.csv")
    predictions = pd.read_csv(out / "05_prediction_counterfactuals.csv")
    pred_summary = pd.read_csv(out / "05_prediction_counterfactuals_summary.csv")
    sensitivity = pd.read_csv(out / "06_per_sample_sensitivity.csv")
    ablations = pd.read_csv(out / "07_edge_ablation_matrix.csv")
    probe = pd.read_csv(out / "08_structure_signal_probe.csv")
    representations = pd.read_csv(out / "09_representation_similarity.csv")
    curves = pd.read_csv(out / "10_training_curve_audit.csv")
    supports = pd.read_csv(out / "12_node_support_information.csv")
    robustness = pd.read_csv(out / "13_checkpoint_robustness_table.csv")
    frames = {
        "manifest": manifest, "graph": stats, "hash": hashes,
        "pred": predictions, "pred_summary": pred_summary,
        "sensitivity": sensitivity, "ablation": ablations, "probe": probe,
        "representation": representations, "curves": curves,
        "support": supports, "robustness": robustness,
    }
    expected = {
        "manifest": 6, "graph": 8580, "hash": 8580, "pred": 17160,
        "pred_summary": 72, "sensitivity": 4290, "ablation": 54,
        "probe": 51, "representation": 24, "curves": 305,
        "support": 8580, "robustness": 6,
    }
    actual = {name: len(frame) for name, frame in frames.items()}
    if actual != expected:
        raise AssertionError((actual, expected))
    for frame, keys in [
        (stats, ["graph_profile", "sample_index", "mode"]),
        (hashes, ["graph_profile", "sample_index", "mode"]),
        (predictions, ["run_id", "sample_index", "mode"]),
        (sensitivity, ["run_id", "sample_index"]),
        (ablations, ["run_id", "ablation"]),
        (representations, ["run_id", "mode"]),
    ]:
        if frame.duplicated(keys).any():
            raise AssertionError(f"Duplicate key: {keys}")
    if manifest.monitor_value.isna().any():
        raise AssertionError("Missing monitor_value")
    for path in manifest.config_path.tolist() + manifest.checkpoint_path.tolist():
        if not Path(path).exists():
            raise AssertionError(f"Missing locked artifact: {path}")

    logit_cols = [f"logit_{name}" for name in NAMES]
    prob_cols = [f"prob_{name}" for name in NAMES]
    if not np.isfinite(predictions[logit_cols + prob_cols].to_numpy(float)).all():
        raise AssertionError("Non-finite prediction value")
    if not np.allclose(predictions[prob_cols].sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("Probability rows do not sum to one")
    for run_id, group in predictions.groupby("run_id"):
        official = group[group["mode"] == "official"].sort_values("sample_index")
        identity = official[["sample_index", "true_class", "image_id"]].reset_index(drop=True)
        for mode in ("zero_prior", "shuffle_prior", "forced_fallback"):
            current = group[group["mode"] == mode].sort_values("sample_index")
            if not identity.equals(current[["sample_index", "true_class", "image_id"]].reset_index(drop=True)):
                raise AssertionError(f"Identity/order mismatch: {run_id}/{mode}")
        zero = group[group["mode"] == "zero_prior"].sort_values("sample_index")
        forced = group[group["mode"] == "forced_fallback"].sort_values("sample_index")
        if not np.allclose(zero[logit_cols], forced[logit_cols], atol=5e-5):
            raise AssertionError(f"Unexpected zero/forced numerical difference: {run_id}")
        if not np.array_equal(zero.predicted_class.to_numpy(), forced.predicted_class.to_numpy()):
            raise AssertionError(f"Zero/forced prediction mismatch: {run_id}")

    for profile in ("base6_structure", "purified_structure", "d17_pixel_only"):
        zero = hashes[(hashes.graph_profile == profile) & (hashes["mode"] == "zero_prior")].sort_values("sample_index")
        forced = hashes[(hashes.graph_profile == profile) & (hashes["mode"] == "forced_fallback")].sort_values("sample_index")
        if not np.array_equal(zero.edge_index_hash.to_numpy(), forced.edge_index_hash.to_numpy()):
            raise AssertionError(f"Zero/forced topology mismatch: {profile}")
        if not np.array_equal(zero.edge_attr_hash.to_numpy(), forced.edge_attr_hash.to_numpy()):
            raise AssertionError(f"Zero/forced edge_attr mismatch: {profile}")
    d17 = stats[stats.graph_profile == "d17_pixel_only"]
    if not (d17.number_of_structure_edges == 0).all() or not np.allclose(d17.overall_edge_jaccard_with_official, 1.0):
        raise AssertionError("D17 model-input graph correction failed")
    if set(probe.probe_sample_count) != {100} or probe.edge_count.max() >= 13000:
        raise AssertionError("Probe count/edge normalization failed")
    if not np.isfinite(representations.select_dtypes(include=[np.number]).to_numpy()).all():
        raise AssertionError("Non-finite representation metric")
    rank = sensitivity[sensitivity.sensitivity_rank_group.notna()].groupby(["run_id", "sensitivity_rank_group"]).size()
    if not (rank.xs("most_50", level=1) == 50).all() or not (rank.xs("least_50", level=1) == 50).all():
        raise AssertionError("Sensitivity top/bottom ranks incomplete")

    json.loads((out / "15_machine_readable_summary.json").read_text(encoding="utf-8"))
    complete_path = out / "AUDIT_COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("smoke") != "PASS" or not complete.get("reports_synthesized") or not complete.get("d17_graph_rows_repaired"):
        raise AssertionError("Completion flags invalid")
    stale_marker = out / ".audit_in_progress"
    if stale_marker.exists() and json.loads(stale_marker.read_text(encoding="utf-8")).get("status") != "completed":
        raise AssertionError("Progress marker is not explicitly finalized")
    for source in [
        "d18/scripts/audit_ofix18_predecision.py",
        "d18/scripts/finalize_ofix18_predecision_audit.py",
        "d18/scripts/repair_ofix18_d17_graph_audit.py",
        "d18/scripts/synthesize_ofix18_predecision_reports.py",
        "d18/scripts/validate_ofix18_predecision_audit.py",
    ]:
        ast.parse((PROJECT_ROOT / source).read_text(encoding="utf-8"), filename=source)

    sample_manifest = pd.read_csv(out / "sample_manifest.csv")
    summary = {
        "status": "PASS", "validated_at": time.time(), "row_counts": actual,
        "class_counts": sample_manifest.true_class.value_counts().sort_index().to_dict(),
        "landmark_missing_flag_counts": sample_manifest.landmark_missing_flag.value_counts().sort_index().to_dict(),
        "zero_forced_graph_identity": "100% for edge_index and edge_attr in all three graph profiles",
        "repeated_inference_logit_tolerance": 5e-5,
        "finite_predictions": True, "mode_identity_and_order": True,
        "stale_progress_marker_present": stale_marker.exists(),
    }
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    complete.update({"validation_status": "PASS", "validation_summary": str((out / "validation_summary.json").relative_to(PROJECT_ROOT)), "files_created": len(list(out.iterdir()))})
    complete_path.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()