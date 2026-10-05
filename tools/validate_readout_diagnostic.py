"""Independent validator for the MPG-FER Readout Diagnostic."""

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

DIAG_DIR = ROOT / "analysis" / "mpg_fer_readout_diagnostic"
EXPECTED_CKPT_SHA = "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"


def validate_checksums() -> dict:
    chk_file = DIAG_DIR / "checksums.sha256"
    if not chk_file.is_file():
        raise FileNotFoundError(f"Missing {chk_file}")
    lines = chk_file.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        expected_sha = parts[0]
        fname = parts[1]
        fpath = DIAG_DIR / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing file: {fname}")
        actual_sha = sha256_file(fpath)
        if actual_sha != expected_sha:
            raise ValueError(f"Checksum mismatch for {fname}: expected {expected_sha}, got {actual_sha}")
    return {"status": "PASS", "verified_files": len(lines)}


def validate_source_and_features() -> dict:
    # 1. Source Audit
    source_audit_path = DIAG_DIR / "SOURCE_AUDIT.json"
    if not source_audit_path.is_file():
        raise FileNotFoundError("SOURCE_AUDIT.json is missing")
    sa = json.load(open(source_audit_path, "r", encoding="utf-8"))
    if sa["checkpoint_sha256"] != EXPECTED_CKPT_SHA:
        raise ValueError(f"Checkpoint SHA mismatch: {sa['checkpoint_sha256']}")
    if sa["motif_node_tensor_shape"] != [49, 192]:
        raise ValueError(f"Motif node tensor shape mismatch: {sa['motif_node_tensor_shape']}")

    # 2. Feature Manifest
    feat_manifest_path = DIAG_DIR / "FEATURE_MANIFEST.json"
    if not feat_manifest_path.is_file():
        raise FileNotFoundError("FEATURE_MANIFEST.json is missing")
    fm = json.load(open(feat_manifest_path, "r", encoding="utf-8"))
    if fm["source_checkpoint_sha256"] != EXPECTED_CKPT_SHA:
        raise ValueError(f"Feature source checkpoint SHA mismatch: {fm['source_checkpoint_sha256']}")
    if fm["train_samples"] != 28709:
        raise ValueError(f"Train sample count mismatch: {fm['train_samples']}")
    if fm["val_samples"] != 3589:
        raise ValueError(f"Val sample count mismatch: {fm['val_samples']}")
    if fm["h_m_shape"] != [49, 192]:
        raise ValueError(f"H_M shape mismatch: {fm['h_m_shape']}")

    return {"status": "PASS", "features_verified": True}


def validate_probes_and_metrics() -> dict:
    summary_path = DIAG_DIR / "READOUT_DIAGNOSTIC_SUMMARY.json"
    if not summary_path.is_file():
        raise FileNotFoundError("READOUT_DIAGNOSTIC_SUMMARY.json is missing")
    summary = json.load(open(summary_path, "r", encoding="utf-8"))

    # Check finite metrics
    r0_res = summary["r0_refit"]
    r1_res = summary["r1_multi_slot"]
    r2_res = summary["r2_class_conditioned"]

    for name, res in [("R0", r0_res), ("R1", r1_res), ("R2", r2_res)]:
        acc = res["val_accuracy"]
        f1 = res["val_macro_f1"]
        if not math.isfinite(acc) or not math.isfinite(f1):
            raise ValueError(f"Non-finite metric in {name}: acc={acc}, f1={f1}")

    # Check query count = 7 for R1 and R2
    r1_diag = json.load(open(DIAG_DIR / "R1_ATTENTION_DIAGNOSTICS.json", "r", encoding="utf-8"))
    r2_diag = json.load(open(DIAG_DIR / "R2_ATTENTION_DIAGNOSTICS.json", "r", encoding="utf-8"))

    if len(r1_diag["entropy_per_slot"]) != 7:
        raise ValueError(f"R1 slot count mismatch: expected 7, got {len(r1_diag['entropy_per_slot'])}")
    if len(r2_diag["entropy_per_class"]) != 7:
        raise ValueError(f"R2 class query count mismatch: expected 7, got {len(r2_diag['entropy_per_class'])}")

    # Check delta arithmetic
    deltas = summary["pairwise_deltas_pp"]
    r0_acc = r0_res["val_accuracy"] * 100.0
    r1_acc = r1_res["val_accuracy"] * 100.0
    r2_acc = r2_res["val_accuracy"] * 100.0

    exp_d_r1_r0 = r1_acc - r0_acc
    exp_d_r2_r0 = r2_acc - r0_acc
    exp_d_r2_r1 = r2_acc - r1_acc

    if abs(deltas["r1_minus_r0"] - exp_d_r1_r0) > 1e-4:
        raise ValueError(f"Delta R1 - R0 mismatch: {deltas['r1_minus_r0']} vs {exp_d_r1_r0}")
    if abs(deltas["r2_minus_r0"] - exp_d_r2_r0) > 1e-4:
        raise ValueError(f"Delta R2 - R0 mismatch: {deltas['r2_minus_r0']} vs {exp_d_r2_r0}")
    if abs(deltas["r2_minus_r1"] - exp_d_r2_r1) > 1e-4:
        raise ValueError(f"Delta R2 - R1 mismatch: {deltas['r2_minus_r1']} vs {exp_d_r2_r1}")

    # Decision logic check
    if exp_d_r1_r0 <= 0.20 and exp_d_r2_r0 <= 0.20:
        expected_status = "READOUT_DIAGNOSTIC_NO_GO"
    elif exp_d_r2_r1 >= 0.30 and exp_d_r2_r0 > 0.0:
        expected_status = "READOUT_DIAGNOSTIC_GO_CLASS_CONDITIONED"
    elif exp_d_r1_r0 >= 0.30:
        expected_status = "READOUT_DIAGNOSTIC_GO_MULTISLOT"
    else:
        expected_status = "READOUT_DIAGNOSTIC_AMBIGUOUS"

    if summary["decision_status_string"] != expected_status:
        raise ValueError(f"Decision mismatch: expected {expected_status}, got {summary['decision_status_string']}")

    return {"status": "PASS", "decision_verified": expected_status}


def main():
    print("Running independent Readout Diagnostic validation...")
    v1 = validate_source_and_features()
    print("1. Source audit & feature manifest:", v1["status"])

    v2 = validate_probes_and_metrics()
    print(f"2. Probe metrics & decision logic ({v2['decision_verified']}):", v2["status"])

    v3 = validate_checksums()
    print("3. Artifact checksums:", v3["status"])

    overall = "PASS" if v1["status"] == "PASS" and v2["status"] == "PASS" and v3["status"] == "PASS" else "FAIL"
    print(f"\nFINAL VALIDATOR STATUS: {overall}")
    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
