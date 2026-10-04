"""Independent validator for the MPG-FER paper ablation audit and recommended ladder."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "mpg_fer_v2_3" / "src"))

from mpg_fer_cumulative_ablation7.protocol import (
    CANONICAL_DATASET_HASHES,
    CANONICAL_DATASET_ROWS,
    sha256_file,
    validate_split_identity,
)

AUDIT_DIR = ROOT / "analysis" / "mpg_fer_ablation_paper_audit"
CUMULATIVE_DIR = ROOT / "analysis" / "mpg_fer_cumulative_ablation7"

EXPECTED_STAGES = ["P0", "P1", "P2", "P3", "P4", "P5"]


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
            raise FileNotFoundError(f"Missing audited file: {fname}")
        actual_sha = sha256_file(fpath)
        if actual_sha != expected_sha:
            raise ValueError(f"Checksum mismatch for {fname}: expected {expected_sha}, got {actual_sha}")
    return {"status": "PASS", "verified_files": len(lines)}


def validate_ladder_provenance() -> dict:
    results_path = AUDIT_DIR / "PAPER_ABLATION_RESULTS.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing {results_path}")
    data = json.load(open(results_path, "r", encoding="utf-8"))
    rows = data["rows"]
    if len(rows) != len(EXPECTED_STAGES):
        raise ValueError(f"Expected {len(EXPECTED_STAGES)} rows, got {len(rows)}")

    # Check sources for each row
    sources_to_check = {
        "P0": CUMULATIVE_DIR / "A0_RESULT.json",
        "P1": CUMULATIVE_DIR / "A1_RESULT.json",
        "P2": CUMULATIVE_DIR / "A3_RESULT.json",
        "P3": CUMULATIVE_DIR / "A4_RESULT.json",
        "P4": AUDIT_DIR / "runs" / "P4" / "canonical_private_metrics.json",
        "P5": CUMULATIVE_DIR / "A6_RESULT.json",
    }

    prev_tta = None
    for i, r in enumerate(rows):
        stage = r["stage"]
        if stage != EXPECTED_STAGES[i]:
            raise ValueError(f"Stage sequence error: expected {EXPECTED_STAGES[i]}, got {stage}")

        # Check finite metrics
        for k in ("raw_accuracy", "raw_macro_f1", "tta_accuracy", "tta_macro_f1"):
            v = r[k]
            if not math.isfinite(v):
                raise ValueError(f"Non-finite metric {k} in {stage}: {v}")

        # Gain arithmetic
        tta = r["tta_accuracy"]
        if i == 0:
            if r["gain"] is not None:
                raise ValueError(f"Gain for {stage} must be None")
        else:
            expected_gain = tta - prev_tta
            if abs(r["gain"] - expected_gain) > 1e-4:
                raise ValueError(f"Gain mismatch in {stage}: expected {expected_gain}, got {r['gain']}")
        prev_tta = tta

        # Check raw artifact matching
        src_path = sources_to_check[stage]
        if not src_path.is_file():
            raise FileNotFoundError(f"Raw source artifact missing for {stage}: {src_path}")
        src_data = json.load(open(src_path, "r", encoding="utf-8"))

        if stage == "P4":
            priv = src_data["views"]
            src_raw_acc = priv["raw"]["accuracy"] * 100.0
            src_tta_acc = priv["horizontal_flip_tta"]["accuracy"] * 100.0
            src_sha = src_data["checkpoint_sha256"]
            src_epoch = src_data["selected_epoch"]
        else:
            priv = src_data["private_metrics"]
            src_raw_acc = priv["raw"]["accuracy"] * 100.0
            src_tta_acc = priv["tta"]["accuracy"] * 100.0
            src_sha = src_data["checkpoint_sha256"]
            src_epoch = src_data["selected_epoch"]

        if abs(r["raw_accuracy"] - src_raw_acc) > 1e-4:
            raise ValueError(f"Raw accuracy mismatch for {stage}: {r['raw_accuracy']} vs {src_raw_acc}")
        if abs(r["tta_accuracy"] - src_tta_acc) > 1e-4:
            raise ValueError(f"TTA accuracy mismatch for {stage}: {r['tta_accuracy']} vs {src_tta_acc}")
        if r["checkpoint_sha256"] != src_sha:
            raise ValueError(f"Checkpoint SHA mismatch for {stage}: {r['checkpoint_sha256']} vs {src_sha}")
        if r["selected_epoch"] != src_epoch:
            raise ValueError(f"Selected epoch mismatch for {stage}: {r['selected_epoch']} vs {src_epoch}")

    return {"status": "PASS", "stages_verified": len(rows)}


def validate_paper_tables() -> dict:
    table_md = AUDIT_DIR / "PAPER_ABLATION_TABLE.md"
    results_json = AUDIT_DIR / "PAPER_ABLATION_RESULTS.json"
    rows = json.load(open(results_json, "r", encoding="utf-8"))["rows"]

    txt = table_md.read_text(encoding="utf-8")
    for r in rows:
        disp_name = r["display_name"]
        tta_str = f"{r['tta_accuracy']:.2f}"
        if disp_name not in txt or tta_str not in txt:
            raise ValueError(f"Table markdown missing {disp_name} or value {tta_str}")

    return {"status": "PASS", "tables_verified": True}


def main() -> None:
    print("Running independent paper audit validation...")
    r1 = validate_ladder_provenance()
    print("1. Ladder provenance & gain arithmetic:", r1["status"])

    r2 = validate_paper_tables()
    print("2. Paper table consistency:", r2["status"])

    r3 = validate_checksums()
    print("3. Checksum integrity:", r3["status"])

    overall = "PASS" if r1["status"] == "PASS" and r2["status"] == "PASS" and r3["status"] == "PASS" else "FAIL"
    print(f"\nFINAL VALIDATOR STATUS: {overall}")
    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
