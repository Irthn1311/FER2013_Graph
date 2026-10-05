"""Independent validator for the MPG-FER Upstream Representation and Optimization Audit."""

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

AUDIT_DIR = ROOT / "analysis" / "mpg_fer_upstream_audit"
EXPECTED_FULL_SHA = "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"
EXPECTED_NPF_SHA = "f301895cd174f8adf998d7f622279510e45bf210e2cc52ebc25376db49e5f972"


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


def validate_checkpoints_and_sources() -> dict:
    source_audit_path = AUDIT_DIR / "SOURCE_AUDIT.json"
    if not source_audit_path.is_file():
        raise FileNotFoundError("SOURCE_AUDIT.json is missing")
    sa = json.load(open(source_audit_path, "r", encoding="utf-8"))
    if sa["full_checkpoint_sha256"] != EXPECTED_FULL_SHA:
        raise ValueError(f"FULL Checkpoint SHA mismatch: {sa['full_checkpoint_sha256']}")
    if sa["npf_checkpoint_sha256"] != EXPECTED_NPF_SHA:
        raise ValueError(f"NPF Checkpoint SHA mismatch: {sa['npf_checkpoint_sha256']}")

    shapes = sa["shapes"]
    for lyr in ["M0", "M1", "M2", "M3", "M4", "M5"]:
        if shapes[lyr] != [-1, 49, 192]:
            raise ValueError(f"Layer {lyr} shape mismatch: {shapes[lyr]}")
    if shapes["P"] != [-1, 2304, 96]:
        raise ValueError(f"Layer P shape mismatch: {shapes['P']}")

    return {"status": "PASS", "checkpoints_verified": True}


def validate_probes_and_geometry() -> dict:
    probes_path = AUDIT_DIR / "LAYERWISE_PROBE_RESULTS.json"
    if not probes_path.is_file():
        raise FileNotFoundError("LAYERWISE_PROBE_RESULTS.json missing")
    probes = json.load(open(probes_path, "r", encoding="utf-8"))
    if len(probes) != 14: # 7 for FULL, 7 for NPF
        raise ValueError(f"Expected 14 probe evaluations, got {len(probes)}")

    for p in probes:
        acc = p["val_accuracy"]
        f1 = p["val_macro_f1"]
        if not math.isfinite(acc) or not math.isfinite(f1):
            raise ValueError(f"Non-finite probe metric in {p['model']} {p['layer']}")

    geom_path = AUDIT_DIR / "REPRESENTATION_GEOMETRY.json"
    if not geom_path.is_file():
        raise FileNotFoundError("REPRESENTATION_GEOMETRY.json missing")
    geom = json.load(open(geom_path, "r", encoding="utf-8"))
    if len(geom) != 12: # 6 layers (M0..M5) x 2 models
        raise ValueError(f"Expected 12 geometry evaluations, got {len(geom)}")

    cka_path = AUDIT_DIR / "FULL_NPF_LAYERWISE_CKA.json"
    if not cka_path.is_file():
        raise FileNotFoundError("FULL_NPF_LAYERWISE_CKA.json missing")
    cka = json.load(open(cka_path, "r", encoding="utf-8"))
    if len(cka) != 7: # P + M0..M5
        raise ValueError(f"Expected 7 CKA evaluations, got {len(cka)}")

    return {"status": "PASS", "probes_and_geometry_verified": True}


def validate_gradient_alignment_and_decision() -> dict:
    grad_path = AUDIT_DIR / "GRADIENT_ALIGNMENT.json"
    if not grad_path.is_file():
        raise FileNotFoundError("GRADIENT_ALIGNMENT.json missing")
    grads = json.load(open(grad_path, "r", encoding="utf-8"))
    if len(grads) != 5: # 5 parameter groups
        raise ValueError(f"Expected 5 parameter groups in gradient alignment, got {len(grads)}")

    summary_path = AUDIT_DIR / "UPSTREAM_AUDIT_SUMMARY.json"
    if not summary_path.is_file():
        raise FileNotFoundError("UPSTREAM_AUDIT_SUMMARY.json missing")
    summary = json.load(open(summary_path, "r", encoding="utf-8"))

    expected_decision = "UPSTREAM_D: No clean layerwise degradation, no material gradient conflict, and no localized divergence"
    expected_status = "UPSTREAM_AUDIT_STOP_INCREMENTAL"

    if summary["decision_classification"] != expected_decision:
        raise ValueError(f"Decision mismatch: {summary['decision_classification']}")
    if summary["decision_status_string"] != expected_status:
        raise ValueError(f"Status string mismatch: {summary['decision_status_string']}")

    return {"status": "PASS", "decision_verified": expected_status}


def main():
    print("Running independent Upstream Representation & Optimization Audit validator...")
    v1 = validate_checkpoints_and_sources()
    print("1. Checkpoints and layer extraction points:", v1["status"])

    v2 = validate_probes_and_geometry()
    print("2. Layerwise probes & representation geometry:", v2["status"])

    v3 = validate_gradient_alignment_and_decision()
    print(f"3. Gradient alignment & decision ({v3['decision_verified']}):", v3["status"])

    v4 = validate_checksums()
    print("4. Artifact checksums:", v4["status"])

    overall = "PASS" if v1["status"] == "PASS" and v2["status"] == "PASS" and v3["status"] == "PASS" and v4["status"] == "PASS" else "FAIL"
    print(f"\nFINAL VALIDATOR STATUS: {overall}")
    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
