"""Independent validator for the Contextual Local Pixel Reinspection Diagnostic."""

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

PIX_DIR = ROOT / "analysis" / "mpg_fer_pixel_reinspection"
EXPECTED_CKPT_SHA = "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"


def validate_checksums() -> dict:
    chk_file = PIX_DIR / "checksums.sha256"
    if not chk_file.is_file():
        raise FileNotFoundError(f"Missing {chk_file}")
    lines = chk_file.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        expected_sha = parts[0]
        fname = parts[1]
        fpath = PIX_DIR / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing file: {fname}")
        actual_sha = sha256_file(fpath)
        if actual_sha != expected_sha:
            raise ValueError(f"Checksum mismatch for {fname}: expected {expected_sha}, got {actual_sha}")
    return {"status": "PASS", "verified_files": len(lines)}


def validate_source_and_features() -> dict:
    # 1. Source Audit
    source_audit_path = PIX_DIR / "SOURCE_AUDIT.json"
    if not source_audit_path.is_file():
        raise FileNotFoundError("SOURCE_AUDIT.json is missing")
    sa = json.load(open(source_audit_path, "r", encoding="utf-8"))
    if sa["checkpoint_sha256"] != EXPECTED_CKPT_SHA:
        raise ValueError(f"Checkpoint SHA mismatch: {sa['checkpoint_sha256']}")
    
    dims = sa["canonical_dimensions"]
    if dims["num_pixels"] != 2304:
        raise ValueError(f"Pixel count mismatch: {dims['num_pixels']}")
    if dims["num_occurrences_M"] != 49:
        raise ValueError(f"Occurrence count mismatch: {dims['num_occurrences_M']}")
    if dims["support_size_max"] != 256:
        raise ValueError(f"Support size mismatch: {dims['support_size_max']}")

    # 2. Feature Manifest
    feat_manifest_path = PIX_DIR / "FEATURE_MANIFEST.json"
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
    if shapes["p_m_i_prior"] != [49, 256]:
        raise ValueError(f"p_m_i shape mismatch: {shapes['p_m_i_prior']}")
    if shapes["h_m_L"] != [49, 192]:
        raise ValueError(f"h_m_L shape mismatch: {shapes['h_m_L']}")
    if shapes["h_bar_P_prior_weighted"] != [49, 96]:
        raise ValueError(f"h_bar_P shape mismatch: {shapes['h_bar_P_prior_weighted']}")

    return {"status": "PASS", "features_verified": True}


def validate_probes_and_decision() -> dict:
    summary_path = PIX_DIR / "PIXEL_REINSPECTION_SUMMARY.json"
    if not summary_path.is_file():
        raise FileNotFoundError("PIXEL_REINSPECTION_SUMMARY.json is missing")
    summary = json.load(open(summary_path, "r", encoding="utf-8"))

    p0_res = summary["p0_baseline"]
    p1_res = summary["p1_local_reinjection"]
    p2_res = summary["p2_contextual_reinspection"]

    for name, res in [("P0", p0_res), ("P1", p1_res), ("P2", p2_res)]:
        acc = res["val_accuracy"]
        f1 = res["val_macro_f1"]
        if not math.isfinite(acc) or not math.isfinite(f1):
            raise ValueError(f"Non-finite metric in {name}: acc={acc}, f1={f1}")

    # Check delta arithmetic
    deltas = summary["pairwise_deltas_pp"]
    p0_acc = p0_res["val_accuracy"] * 100.0
    p1_acc = p1_res["val_accuracy"] * 100.0
    p2_acc = p2_res["val_accuracy"] * 100.0

    exp_d_p1_p0 = p1_acc - p0_acc
    exp_d_p2_p0 = p2_acc - p0_acc
    exp_d_p2_p1 = p2_acc - p1_acc

    if abs(deltas["p1_minus_p0"] - exp_d_p1_p0) > 1e-4:
        raise ValueError(f"Delta P1 - P0 mismatch: {deltas['p1_minus_p0']} vs {exp_d_p1_p0}")
    if abs(deltas["p2_minus_p0"] - exp_d_p2_p0) > 1e-4:
        raise ValueError(f"Delta P2 - P0 mismatch: {deltas['p2_minus_p0']} vs {exp_d_p2_p0}")
    if abs(deltas["p2_minus_p1"] - exp_d_p2_p1) > 1e-4:
        raise ValueError(f"Delta P2 - P1 mismatch: {deltas['p2_minus_p1']} vs {exp_d_p2_p1}")

    # Decision rule check
    f1_p0 = p0_res["val_macro_f1"] * 100.0
    f1_p2 = p2_res["val_macro_f1"] * 100.0

    if exp_d_p2_p0 >= 0.30 and exp_d_p2_p1 >= 0.20 and (f1_p2 >= f1_p0 - 0.10):
        expected_status = "PIXEL_REINSPECTION_GO_CONTEXTUAL"
    elif exp_d_p1_p0 >= 0.30 and abs(p2_acc - p1_acc) < 0.20:
        expected_status = "PIXEL_REINSPECTION_GO_REINJECTION"
    elif exp_d_p1_p0 <= 0.20 and exp_d_p2_p0 <= 0.20:
        expected_status = "PIXEL_REINSPECTION_NO_GO"
    else:
        expected_status = "PIXEL_REINSPECTION_AMBIGUOUS"

    if summary["decision_status_string"] != expected_status:
        raise ValueError(f"Decision mismatch: expected {expected_status}, got {summary['decision_status_string']}")

    # Check P2 reinspection diagnostics
    p2_diag = summary["p2_diagnostics"]
    if not (0.0 <= p2_diag["argmax_pixel_change_rate"] <= 1.0):
        raise ValueError("Invalid argmax pixel change rate")
    if p2_diag["effective_attended_pixels"] <= 0.0:
        raise ValueError("Effective attended pixels must be positive")

    return {"status": "PASS", "decision_verified": expected_status}


def main():
    print("Running independent Contextual Local Pixel Reinspection Diagnostic validation...")
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
