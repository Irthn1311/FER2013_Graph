"""Master script to construct the complete, authentic MPG_FER_HANDOFF package.

Builds staging directory structure:
MPG_FER_HANDOFF/
├── 00_MASTER_SUMMARY_MPG_FER.docx
├── 00_HANDOFF_MANIFEST.md
├── 00_ENVIRONMENT.md
├── 01_CODE/
│   ├── model/
│   ├── losses/
│   ├── dataset/
│   ├── train/
│   ├── evaluation/
│   └── CODE_TRACEABILITY.md
├── 02_CONFIGS/
│   ├── v2.1/
│   ├── v2.2/
│   ├── v2.3/
│   └── CONFIG_INDEX.md
├── 03_RESULTS/
│   ├── MASTER_RUN_REGISTRY.csv
│   ├── v2_2_best_raw.csv
│   ├── version_comparison.csv
│   ├── ablation_results.csv
│   ├── mechanistic_audits/
│   └── predictions/
├── 04_LOGS/
│   ├── v2.1/
│   ├── v2.2/
│   └── v2.3/
├── 05_DATA_PROTOCOL/
│   ├── FER2013_PROTOCOL.md
│   ├── split_manifest.json / .csv
│   └── label_mapping.json / .csv
├── 06_CHECKPOINT_METADATA/
│   ├── checkpoint_manifest.csv
│   └── hashes.txt
├── 07_FIGURES/
│   ├── architecture/
│   ├── confusion_matrix/
│   ├── training_curves/
│   ├── motif_visualization/
│   └── FIGURE_MANIFEST.md
├── 08_AUDITS/
│   ├── A1/
│   ├── A5/
│   ├── A6/
│   ├── A6_R2/
│   └── AUDIT_INDEX.md
└── 09_OPEN_ISSUES/
    └── OPEN_ISSUES.md

Then zips it into MPG_FER_HANDOFF.zip in the repository root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PROJECT_ROOT / "research" / "mpg_fer_handoff_stage"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    print(f"Building MPG-FER handoff package from {PROJECT_ROOT}...", flush=True)

    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 01_CODE
    # -------------------------------------------------------------
    print("Copying 01_CODE...", flush=True)
    v22_src = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "src" / "mpg_fer_v2_2"
    v23_src = PROJECT_ROOT / "research" / "mpg_fer_v2_3" / "src" / "mpg_fer_v2_3"
    
    # Use v2.2 as primary frozen code, plus v2.3 model.py for comparison
    copy_file(v22_src / "model.py", STAGE_DIR / "01_CODE" / "model" / "model_v2_2.py")
    copy_file(v22_src / "model.py", STAGE_DIR / "01_CODE" / "model" / "model.py") # main model file
    if (v23_src / "model.py").exists():
        copy_file(v23_src / "model.py", STAGE_DIR / "01_CODE" / "model" / "model_v2_3.py")

    copy_file(v22_src / "motif.py", STAGE_DIR / "01_CODE" / "model" / "motif.py")
    copy_file(v22_src / "graph.py", STAGE_DIR / "01_CODE" / "model" / "graph.py")
    copy_file(v22_src / "features.py", STAGE_DIR / "01_CODE" / "model" / "features.py")
    copy_file(v22_src / "ema.py", STAGE_DIR / "01_CODE" / "model" / "ema.py")
    copy_file(v22_src / "losses.py", STAGE_DIR / "01_CODE" / "losses" / "losses.py")
    copy_file(v22_src / "data.py", STAGE_DIR / "01_CODE" / "dataset" / "data.py")
    copy_file(v22_src / "train.py", STAGE_DIR / "01_CODE" / "train" / "train.py")
    copy_file(v22_src / "evaluate.py", STAGE_DIR / "01_CODE" / "evaluation" / "evaluate.py")
    copy_file(v22_src / "checkpoint.py", STAGE_DIR / "01_CODE" / "train" / "checkpoint.py")
    copy_file(v22_src / "utils.py", STAGE_DIR / "01_CODE" / "train" / "utils.py")

    # -------------------------------------------------------------
    # 02_CONFIGS
    # -------------------------------------------------------------
    print("Copying 02_CONFIGS...", flush=True)
    # v2.1
    v21_root = PROJECT_ROOT / "research" / "mpg_fer_v2_1"
    copy_file(v21_root / "src" / "mpg_fer_v2_1" / "config.py", STAGE_DIR / "02_CONFIGS" / "v2.1" / "config.py")
    copy_file(v21_root / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "config.json", STAGE_DIR / "02_CONFIGS" / "v2.1" / "config.json")
    
    # v2.2
    v22_root = PROJECT_ROOT / "research" / "mpg_fer_v2_2"
    copy_file(v22_root / "src" / "mpg_fer_v2_2" / "config.py", STAGE_DIR / "02_CONFIGS" / "v2.2" / "config.py")
    copy_file(v22_root / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "config.json", STAGE_DIR / "02_CONFIGS" / "v2.2" / "config.json")
    copy_file(v22_root / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "execution_manifest.json", STAGE_DIR / "02_CONFIGS" / "v2.2" / "execution_manifest.json")
    copy_file(v22_root / "notebooks" / "MPG_FER_v2_2_Kaggle_T4.ipynb", STAGE_DIR / "02_CONFIGS" / "v2.2" / "MPG_FER_v2_2_Kaggle_T4.ipynb")

    # v2.3
    v23_root = PROJECT_ROOT / "research" / "mpg_fer_v2_3"
    if (v23_root / "src" / "mpg_fer_v2_3" / "config.py").exists():
        copy_file(v23_root / "src" / "mpg_fer_v2_3" / "config.py", STAGE_DIR / "02_CONFIGS" / "v2.3" / "config.py")
    copy_file(v23_root / "official_runs" / "segment_02" / "mpg_fer_v2_3_run" / "config.json", STAGE_DIR / "02_CONFIGS" / "v2.3" / "config.json")
    copy_file(v23_root / "official_runs" / "segment_02" / "mpg_fer_v2_3_run" / "execution_manifest.json", STAGE_DIR / "02_CONFIGS" / "v2.3" / "execution_manifest.json")
    copy_file(v23_root / "notebooks" / "MPG_FER_v2_3_Kaggle_T4.ipynb", STAGE_DIR / "02_CONFIGS" / "v2.3" / "MPG_FER_v2_3_Kaggle_T4.ipynb")

    # -------------------------------------------------------------
    # 04_LOGS
    # -------------------------------------------------------------
    print("Copying 04_LOGS...", flush=True)
    # v2.1
    copy_file(v21_root / "official_runs" / "segment_01" / "mpg-fer-v2-1-official-t4-segment-01.log", STAGE_DIR / "04_LOGS" / "v2.1" / "segment-01.log")
    copy_file(v21_root / "official_runs" / "segment_02" / "mpg-fer-v2-1-official-t4-segment-02.log", STAGE_DIR / "04_LOGS" / "v2.1" / "segment-02.log")
    copy_file(v21_root / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "history.csv", STAGE_DIR / "04_LOGS" / "v2.1" / "history.csv")
    copy_file(v21_root / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "history.json", STAGE_DIR / "04_LOGS" / "v2.1" / "history.json")

    # v2.2
    copy_file(v22_root / "official_runs" / "segment_01" / "mpg-fer-v2-2-official-t4-segment-01.log", STAGE_DIR / "04_LOGS" / "v2.2" / "segment-01.log")
    copy_file(v22_root / "official_runs" / "segment_02" / "mpg-fer-v2-2-official-t4-segment-02.log", STAGE_DIR / "04_LOGS" / "v2.2" / "segment-02.log")
    copy_file(v22_root / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "history.csv", STAGE_DIR / "04_LOGS" / "v2.2" / "history.csv")
    copy_file(v22_root / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "history.json", STAGE_DIR / "04_LOGS" / "v2.2" / "history.json")

    # v2.3
    copy_file(v23_root / "official_runs" / "segment_01" / "mpg-fer-v2-3-official-t4-segment-01.log", STAGE_DIR / "04_LOGS" / "v2.3" / "segment-01.log")
    copy_file(v23_root / "official_runs" / "segment_02" / "mpg-fer-v2-3-official-t4-segment-02.log", STAGE_DIR / "04_LOGS" / "v2.3" / "segment-02.log")
    copy_file(v23_root / "official_runs" / "segment_02" / "mpg_fer_v2_3_run" / "history.csv", STAGE_DIR / "04_LOGS" / "v2.3" / "history.csv")
    copy_file(v23_root / "official_runs" / "segment_02" / "mpg_fer_v2_3_run" / "history.json", STAGE_DIR / "04_LOGS" / "v2.3" / "history.json")

    # -------------------------------------------------------------
    # 07_FIGURES
    # -------------------------------------------------------------
    print("Copying 07_FIGURES...", flush=True)
    fig_dir = STAGE_DIR / "07_FIGURES"
    # confusion matrices
    copy_file(v21_root / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "public_confusion_matrix.png", fig_dir / "confusion_matrix" / "v21_public_confusion_matrix.png")
    copy_file(v21_root / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "private_confusion_matrix.png", fig_dir / "confusion_matrix" / "v21_private_confusion_matrix.png")
    copy_file(v22_root / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "public_confusion_matrix.png", fig_dir / "confusion_matrix" / "v22_public_confusion_matrix.png")
    copy_file(v22_root / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "private_confusion_matrix.png", fig_dir / "confusion_matrix" / "v22_private_confusion_matrix.png")
    copy_file(v23_root / "official_runs" / "segment_02" / "mpg_fer_v2_3_run" / "public_confusion_matrix.png", fig_dir / "confusion_matrix" / "v23_public_confusion_matrix.png")
    copy_file(v23_root / "official_runs" / "segment_02" / "mpg_fer_v2_3_run" / "private_confusion_matrix.png", fig_dir / "confusion_matrix" / "v23_private_confusion_matrix.png")

    # training curves
    copy_file(v21_root / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "training_curves.png", fig_dir / "training_curves" / "v21_training_curves.png")
    copy_file(v22_root / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "training_curves.png", fig_dir / "training_curves" / "v22_training_curves.png")
    copy_file(v23_root / "official_runs" / "segment_02" / "mpg_fer_v2_3_run" / "training_curves.png", fig_dir / "training_curves" / "v23_training_curves.png")

    # motif_visualization / audit plots
    audit_plots = [
        ("research/mpg_fer_audit/a4/a4_cka_pooled.png", "a4_cka_pooled.png"),
        ("research/mpg_fer_audit/a4/a4_cka_cross_layer_public.png", "a4_cka_cross_layer_public.png"),
        ("research/mpg_fer_audit/a4/a4_routing_overlap_by_layer.png", "a4_routing_overlap_by_layer.png"),
        ("research/mpg_fer_audit/a4/a4_calibration_public.png", "a4_calibration_public.png"),
        ("research/mpg_fer_audit/a4/a4_calibration_private.png", "a4_calibration_private.png"),
        ("research/mpg_fer_audit/a4/a4_flip_support_jaccard.png", "a4_flip_support_jaccard.png"),
        ("research/mpg_fer_audit/a6/a6_margin_by_stage_v21_correct.png", "a6_margin_by_stage_v21_correct.png"),
        ("research/mpg_fer_audit/a6/a6_margin_by_stage_v22_correct.png", "a6_margin_by_stage_v22_correct.png"),
        ("research/mpg_fer_audit/a6/a6_correct_wrong_delta_by_stage.png", "a6_correct_wrong_delta_by_stage.png"),
        ("research/mpg_fer_audit/a6/a6_swap_rescue_by_boundary.png", "a6_swap_rescue_by_boundary.png"),
        ("research/mpg_fer_audit/a6r/a6r_composer_component_rescue.png", "a6r_composer_component_rescue.png"),
        ("research/mpg_fer_audit/a6r2/a6r2_readout_factorial_corrected.png", "a6r2_readout_factorial_corrected.png"),
        ("research/mpg_fer_audit/a6r2/a6r2_routing_divergence_vs_margin.png", "a6r2_routing_divergence_vs_margin.png"),
    ]
    for src_rel, dst_name in audit_plots:
        src_p = PROJECT_ROOT / src_rel
        if src_p.exists():
            copy_file(src_p, fig_dir / "motif_visualization" / dst_name)

    # -------------------------------------------------------------
    # 08_AUDITS
    # -------------------------------------------------------------
    print("Copying 08_AUDITS...", flush=True)
    aud_dir = STAGE_DIR / "08_AUDITS"
    # A1
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/A1_BRANCH_BOTTLENECK_AUDIT.md", aud_dir / "A1" / "A1_BRANCH_BOTTLENECK_AUDIT.md")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a1_linear_probes.csv", aud_dir / "A1" / "a1_linear_probes.csv")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a1_probe_metrics.json", aud_dir / "A1" / "a1_probe_metrics.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a1_primary_comparison_deltas.csv", aud_dir / "A1" / "a1_primary_comparison_deltas.csv")

    # A5 & A5-R & A5-H
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5/A5_HARD_EXAMPLE_DATA_AUDIT.md", aud_dir / "A5" / "A5_HARD_EXAMPLE_DATA_AUDIT.md")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5/A5R_AUTOMATED_CORRECTION_AND_LEAKAGE_AUDIT.md", aud_dir / "A5" / "A5R_AUTOMATED_CORRECTION_AND_LEAKAGE_AUDIT.md")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5/a5r_duplicate_leakage.json", aud_dir / "A5" / "a5r_duplicate_leakage.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5/a5r_duplicate_conditioned_metrics.json", aud_dir / "A5" / "a5r_duplicate_conditioned_metrics.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5/a5_exact_duplicate_summary.json", aud_dir / "A5" / "a5_exact_duplicate_summary.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5/a5_exact_duplicates.csv", aud_dir / "A5" / "a5_exact_duplicates.csv")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5/a5r_near_duplicate_corrected.json", aud_dir / "A5" / "a5r_near_duplicate_corrected.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5/a5r_deduplicated_neighborhood.json", aud_dir / "A5" / "a5r_deduplicated_neighborhood.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5h/A5H_FINAL_DIAGNOSTIC_CLOSURE.md", aud_dir / "A5" / "A5H_FINAL_DIAGNOSTIC_CLOSURE.md")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5h/a5h_reviewer_agreement.json", aud_dir / "A5" / "a5h_reviewer_agreement.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5h/a5h_label_mismatch_candidates.csv", aud_dir / "A5" / "a5h_label_mismatch_candidates.csv")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5h/a5h_common_representation_failure_candidates.csv", aud_dir / "A5" / "a5h_common_representation_failure_candidates.csv")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5h/a5h_benchmark_conflict_candidates.csv", aud_dir / "A5" / "a5h_benchmark_conflict_candidates.csv")

    # A6
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6/A6_MODEL_RESOLVABLE_REPRESENTATION_AUDIT.md", aud_dir / "A6" / "A6_MODEL_RESOLVABLE_REPRESENTATION_AUDIT.md")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6/a6_swap_results.json", aud_dir / "A6" / "a6_swap_results.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6/a6_stage_margin_summary.json", aud_dir / "A6" / "a6_stage_margin_summary.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6/a6_clean_sample_groups.csv", aud_dir / "A6" / "a6_clean_sample_groups.csv")

    # A6_R2
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6r2/A6R2_READOUT_ROUTING_SOURCELOCK.md", aud_dir / "A6_R2" / "A6R2_READOUT_ROUTING_SOURCELOCK.md")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6r2/A6R2_CORRECTION_LEDGER.md", aud_dir / "A6_R2" / "A6R2_CORRECTION_LEDGER.md")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6r2/a6r2_master_results.json", aud_dir / "A6_R2" / "a6r2_master_results.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6r2/a6r2_readout_factorial_results.json", aud_dir / "A6_R2" / "a6r2_readout_factorial_results.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6r2/a6r2_routing_association.json", aud_dir / "A6_R2" / "a6r2_routing_association.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6r2/a6r2_final_target.json", aud_dir / "A6_R2" / "a6r2_final_target.json")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6r2/a6r2_source_consistency_check.json", aud_dir / "A6_R2" / "a6r2_source_consistency_check.json")

    # -------------------------------------------------------------
    # 03_RESULTS
    # -------------------------------------------------------------
    print("Copying 03_RESULTS...", flush=True)
    res_dir = STAGE_DIR / "03_RESULTS"
    # mechanistic audits copies
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a4/a4_error_overlap.csv", res_dir / "mechanistic_audits" / "a4_error_overlap.csv")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a4/a4_shared_high_confidence_errors.csv", res_dir / "mechanistic_audits" / "a4_shared_high_confidence_errors.csv")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a4/a4_cross_layer_cka.csv", res_dir / "mechanistic_audits" / "a4_cross_layer_cka.csv")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5/a5_exact_duplicates.csv", res_dir / "mechanistic_audits" / "a5_exact_duplicates.csv")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6/a6_stage_margin_samples.csv", res_dir / "mechanistic_audits" / "a6_stage_margin_samples.csv")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6/a6_swap_sample_results.csv", res_dir / "mechanistic_audits" / "a6_swap_sample_results.csv")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6r2/a6r2_readout_factorial_samples.csv", res_dir / "mechanistic_audits" / "a6r2_readout_factorial_samples.csv")
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a6r2/a6r2_routing_per_sample.csv", res_dir / "mechanistic_audits" / "a6r2_routing_per_sample.csv")

    # predictions
    copy_file(PROJECT_ROOT / "research/mpg_fer_audit/a5/a5_model_consensus.csv", res_dir / "predictions" / "multi_model_test_consensus_fp32.csv")

    print("Base file copies completed.", flush=True)


if __name__ == "__main__":
    main()
