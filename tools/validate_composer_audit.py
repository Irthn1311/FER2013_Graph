"""Independent validator for the MPG-FER Spatial Motif Composer Constraint Audit."""

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

AUDIT_DIR = ROOT / "analysis" / "mpg_fer_composer_audit"
EXPECTED_CKPT_SHA = "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"


def validate_checksums() -> dict:
    chk_file = AUDIT_DIR / "checksums.sha256"
    if not chk_file.is_file():
        raise FileNotFoundError(f"Missing {chk_file}")
    lines = chk_file.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        expected_sha = parts[0]
        fname = parts[1]
        fpath = AUDIT_DIR / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing file: {fname}")
        actual_sha = sha256_file(fpath)
        if actual_sha != expected_sha:
            raise ValueError(f"Checksum mismatch for {fname}: expected {expected_sha}, got {actual_sha}")
    return {"status": "PASS", "verified_files": len(lines)}


def validate_source_audit() -> dict:
    source_audit_path = AUDIT_DIR / "SOURCE_AUDIT.json"
    if not source_audit_path.is_file():
        raise FileNotFoundError("SOURCE_AUDIT.json is missing")
    sa = json.load(open(source_audit_path, "r", encoding="utf-8"))
    if sa["full_checkpoint_sha256"] != EXPECTED_CKPT_SHA:
        raise ValueError(f"Checkpoint SHA mismatch: {sa['full_checkpoint_sha256']}")
    if sa["anchors"] != 49:
        raise ValueError(f"Anchor count mismatch: {sa['anchors']}")
    if sa["scales"] != [8, 12, 16]:
        raise ValueError(f"Scales mismatch: {sa['scales']}")
    if sa["prototypes"] != 48:
        raise ValueError(f"Prototypes mismatch: {sa['prototypes']}")
    return {"status": "PASS", "source_audit_verified": True}


def validate_audit_metrics_and_decision() -> dict:
    # 1. Spatial pressure
    sp_path = AUDIT_DIR / "SPATIAL_SUPPORT_PRESSURE.json"
    if not sp_path.is_file():
        raise FileNotFoundError("SPATIAL_SUPPORT_PRESSURE.json missing")
    sp = json.load(open(sp_path, "r", encoding="utf-8"))
    if len(sp["by_scale"]) != 3:
        raise ValueError("Expected 3 scales in spatial support pressure")
    for r in sp["by_scale"]:
        if not math.isfinite(r["mean_normalized_displacement"]):
            raise ValueError("Non-finite displacement")

    # 2. Scale pressure
    sc_path = AUDIT_DIR / "SCALE_PRESSURE.json"
    if not sc_path.is_file():
        raise FileNotFoundError("SCALE_PRESSURE.json missing")
    sc = json.load(open(sc_path, "r", encoding="utf-8"))
    probs = sc["mean_scale_probabilities"]
    prob_sum = probs["scale_8"] + probs["scale_12"] + probs["scale_16"]
    if abs(prob_sum - 1.0) > 1e-4:
        raise ValueError(f"Scale probabilities do not sum to 1.0: {prob_sum}")

    # 3. Prototype audit
    proto_path = AUDIT_DIR / "PROTOTYPE_AUDIT.json"
    if not proto_path.is_file():
        raise FileNotFoundError("PROTOTYPE_AUDIT.json missing")
    proto = json.load(open(proto_path, "r", encoding="utf-8"))
    if proto["unused_prototypes_count"] != 0:
        raise ValueError("Unused prototypes count unexpected")
    if proto["effective_prototype_count"] < 40.0:
        raise ValueError("Effective prototypes unexpectedly collapsed")

    # 4. Error conditioned analysis
    err_path = AUDIT_DIR / "ERROR_CONDITIONED_ANALYSIS.json"
    if not err_path.is_file():
        raise FileNotFoundError("ERROR_CONDITIONED_ANALYSIS.json missing")
    err = json.load(open(err_path, "r", encoding="utf-8"))
    if len(err) != 9:
        raise ValueError(f"Expected 9 error-conditioned indicators, got {len(err)}")
    for r in err:
        if not math.isfinite(r["cohens_d"]) or not math.isfinite(r["auroc_error_prediction"]):
            raise ValueError(f"Non-finite effect metric in {r['indicator']}")

    # 5. Summary decision
    summary_path = AUDIT_DIR / "COMPOSER_AUDIT_SUMMARY.json"
    if not summary_path.is_file():
        raise FileNotFoundError("COMPOSER_AUDIT_SUMMARY.json missing")
    summary = json.load(open(summary_path, "r", encoding="utf-8"))
    expected_decision = "COMPOSER_D_HEALTHY"
    expected_status = "COMPOSER_AUDIT_STOP"

    if summary["decision_classification"] != expected_decision:
        raise ValueError(f"Decision mismatch: {summary['decision_classification']}")
    if summary["decision_status_string"] != expected_status:
        raise ValueError(f"Status string mismatch: {summary['decision_status_string']}")

    return {"status": "PASS", "decision_verified": expected_status}


def main():
    print("Running independent Spatial Motif Composer Constraint Audit validator...")
    v1 = validate_source_audit()
    print("1. Source audit & checkpoint identity:", v1["status"])

    v2 = validate_audit_metrics_and_decision()
    print(f"2. Audit metrics & decision logic ({v2['decision_verified']}):", v2["status"])

    v3 = validate_checksums()
    print("3. Artifact checksums:", v3["status"])

    overall = "PASS" if v1["status"] == "PASS" and v2["status"] == "PASS" and v3["status"] == "PASS" else "FAIL"
    print(f"\nFINAL VALIDATOR STATUS: {overall}")
    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
