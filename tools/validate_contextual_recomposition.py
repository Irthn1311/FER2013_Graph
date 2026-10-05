"""Independent validator for the Contextual Scale Recomposition (CSR) Diagnostic."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "mpg_fer_v2_3" / "src"))

from mpg_fer_cumulative_ablation7.protocol import sha256_file

CSR_DIR = ROOT / "analysis" / "mpg_fer_contextual_recomposition"
EXPECTED_CKPT_SHA = "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"


def validate_checksums() -> dict:
    chk_file = CSR_DIR / "checksums.sha256"
    if not chk_file.is_file():
        raise FileNotFoundError(f"Missing {chk_file}")
    lines = chk_file.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        expected_sha = parts[0]
        fname = parts[1]
        fpath = CSR_DIR / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing file: {fname}")
        actual_sha = sha256_file(fpath)
        if actual_sha != expected_sha:
            raise ValueError(f"Checksum mismatch for {fname}: expected {expected_sha}, got {actual_sha}")
    return {"status": "PASS", "verified_files": len(lines)}


def validate_source_and_features() -> dict:
    # 1. Source Audit
    source_audit_path = CSR_DIR / "SOURCE_AUDIT.json"
    if not source_audit_path.is_file():
        raise FileNotFoundError("SOURCE_AUDIT.json is missing")
    sa = json.load(open(source_audit_path, "r", encoding="utf-8"))
    if sa["checkpoint_sha256"] != EXPECTED_CKPT_SHA:
        raise ValueError(f"Checkpoint SHA mismatch: {sa['checkpoint_sha256']}")
    
    dims = sa["canonical_dimensions"]
    if dims["num_occurrences_M"] != 49:
        raise ValueError(f"Occurrence count mismatch: {dims['num_occurrences_M']}")
    if dims["num_scales_S"] != 3 or dims["scale_window_sizes"] != [8, 12, 16]:
        raise ValueError(f"Scales mismatch: {dims['scale_window_sizes']}")
    if dims["d_motif"] != 192:
        raise ValueError(f"d_motif mismatch: {dims['d_motif']}")

    # 2. Feature Manifest
    feat_manifest_path = CSR_DIR / "FEATURE_MANIFEST.json"
    if not feat_manifest_path.is_file():
        raise FileNotFoundError("FEATURE_MANIFEST.json is missing")
    fm = json.load(open(feat_manifest_path, "r", encoding="utf-8"))
    if fm["source_checkpoint_sha256"] != EXPECTED_CKPT_SHA:
        raise ValueError(f"Feature source checkpoint SHA mismatch: {fm['source_checkpoint_sha256']}")
    if fm["train_samples"] != 28709:
        raise ValueError(f"Train sample count mismatch: {fm['train_samples']}")
    if fm["val_samples"] != 3589:
        raise ValueError(f"Val sample count mismatch: {fm['val_samples']}")
    
    shapes = fm["shapes"]
    if shapes["z_m_s"] != [49, 3, 192]:
        raise ValueError(f"z_m_s shape mismatch: {shapes['z_m_s']}")
    if shapes["alpha_m_s"] != [49, 3]:
        raise ValueError(f"alpha_m_s shape mismatch: {shapes['alpha_m_s']}")
    if shapes["h_m_0"] != [49, 192]:
        raise ValueError(f"h_m_0 shape mismatch: {shapes['h_m_0']}")
    if shapes["h_m_L"] != [49, 192]:
        raise ValueError(f"h_m_L shape mismatch: {shapes['h_m_L']}")

    return {"status": "PASS", "features_verified": True}


def validate_probes_and_decision() -> dict:
    summary_path = CSR_DIR / "CSR_DIAGNOSTIC_SUMMARY.json"
    if not summary_path.is_file():
        raise FileNotFoundError("CSR_DIAGNOSTIC_SUMMARY.json is missing")
    summary = json.load(open(summary_path, "r", encoding="utf-8"))

    c0_res = summary["c0_baseline"]
    c1_res = summary["c1_early_skip"]
    c2_res = summary["c2_csr"]

    for name, res in [("C0", c0_res), ("C1", c1_res), ("C2", c2_res)]:
        acc = res["val_accuracy"]
        f1 = res["val_macro_f1"]
        if not math.isfinite(acc) or not math.isfinite(f1):
            raise ValueError(f"Non-finite metric in {name}: acc={acc}, f1={f1}")

    # Check delta arithmetic
    deltas = summary["pairwise_deltas_pp"]
    c0_acc = c0_res["val_accuracy"] * 100.0
    c1_acc = c1_res["val_accuracy"] * 100.0
    c2_acc = c2_res["val_accuracy"] * 100.0

    exp_d_c1_c0 = c1_acc - c0_acc
    exp_d_c2_c0 = c2_acc - c0_acc
    exp_d_c2_c1 = c2_acc - c1_acc

    if abs(deltas["c1_minus_c0"] - exp_d_c1_c0) > 1e-4:
        raise ValueError(f"Delta C1 - C0 mismatch: {deltas['c1_minus_c0']} vs {exp_d_c1_c0}")
    if abs(deltas["c2_minus_c0"] - exp_d_c2_c0) > 1e-4:
        raise ValueError(f"Delta C2 - C0 mismatch: {deltas['c2_minus_c0']} vs {exp_d_c2_c0}")
    if abs(deltas["c2_minus_c1"] - exp_d_c2_c1) > 1e-4:
        raise ValueError(f"Delta C2 - C1 mismatch: {deltas['c2_minus_c1']} vs {exp_d_c2_c1}")

    # Check decision logic
    f1_c0 = c0_res["val_macro_f1"] * 100.0
    f1_c2 = c2_res["val_macro_f1"] * 100.0
    
    if exp_d_c2_c0 >= 0.30 and exp_d_c2_c1 >= 0.20 and (f1_c2 >= f1_c0 - 0.10):
        expected_status = "CSR_DIAGNOSTIC_GO_RECOMPOSITION"
    elif exp_d_c1_c0 >= 0.30 and abs(c2_acc - c1_acc) < 0.20:
        expected_status = "CSR_DIAGNOSTIC_GO_SKIP"
    elif exp_d_c1_c0 <= 0.20 and exp_d_c2_c0 <= 0.20:
        expected_status = "CSR_DIAGNOSTIC_NO_GO"
    else:
        expected_status = "CSR_DIAGNOSTIC_AMBIGUOUS"

    if summary["decision_status_string"] != expected_status:
        raise ValueError(f"Decision mismatch: expected {expected_status}, got {summary['decision_status_string']}")

    # Check C2 recomposition dynamics
    c2_diag = summary["c2_diagnostics"]
    if not (0.0 <= c2_diag["argmax_scale_change_rate"] <= 1.0):
        raise ValueError("Invalid argmax scale change rate")
    if len(c2_diag["per_occurrence_change_frequency"]) != 49:
        raise ValueError("Occurrence change frequency list must have length 49")

    return {"status": "PASS", "decision_verified": expected_status}


def main():
    print("Running independent Contextual Scale Recomposition (CSR) Diagnostic validation...")
    v1 = validate_source_and_features()
    print("1. Source audit & feature extraction points:", v1["status"])

    v2 = validate_probes_and_decision()
    print(f"2. Probe metrics & decision logic ({v2['decision_verified']}):", v2["status"])

    v3 = validate_checksums()
    print("3. Artifact checksums:", v3["status"])

    overall = "PASS" if v1["status"] == "PASS" and v2["status"] == "PASS" and v3["status"] == "PASS" else "FAIL"
    print(f"\nFINAL VALIDATOR STATUS: {overall}")
    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
