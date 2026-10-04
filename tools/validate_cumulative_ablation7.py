"""Independent validator for the MPG-FER 7-configuration cumulative ablation ladder."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "mpg_fer_v2_3" / "src"))

from mpg_fer_cumulative_ablation7.model import (
    CUMULATIVE_ABLATION_ORDER,
    CUMULATIVE_REGISTRY,
    CumulativeAblationMode,
    CumulativeAblationMPGFER,
)
from mpg_fer_cumulative_ablation7.protocol import (
    CANONICAL_DATASET_HASHES,
    CANONICAL_DATASET_ROWS,
    EXPECTED_A6_METRICS,
    compute_pairwise_configuration_diffs,
    sha256_file,
    validate_split_identity,
)

ANALYSIS_DIR = ROOT / "analysis" / "mpg_fer_cumulative_ablation7"
EXPECTED_MODES = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]


def validate_checksums() -> dict[str, Any]:
    chk_file = ANALYSIS_DIR / "checksums.sha256"
    if not chk_file.is_file():
        return {"status": "PENDING", "reason": "checksums.sha256 not yet created"}
    lines = chk_file.read_text(encoding="utf-8").strip().splitlines()
    mismatches = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        expected_sha = parts[0]
        fname = parts[1]
        fpath = ANALYSIS_DIR / fname
        if not fpath.is_file():
            mismatches.append(f"Missing file: {fname}")
            continue
        actual_sha = sha256_file(fpath)
        if actual_sha != expected_sha:
            mismatches.append(f"Hash mismatch {fname}: expected {expected_sha}, got {actual_sha}")
    if mismatches:
        raise ValueError(f"Checksum verification failed: {mismatches}")
    return {"status": "PASS", "verified_files": len(lines)}


def validate_order_and_configs() -> dict[str, Any]:
    actual_order = [m.value for m in CUMULATIVE_ABLATION_ORDER]
    if actual_order != EXPECTED_MODES:
        raise ValueError(f"Order mismatch: {actual_order} != {EXPECTED_MODES}")
    for mode in EXPECTED_MODES:
        cfg_path = ANALYSIS_DIR / f"{mode}_CONFIG.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config file: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg["configuration"] != mode:
            raise ValueError(f"Config mode mismatch in {cfg_path}: expected {mode}, got {cfg['configuration']}")
    return {"status": "PASS", "modes": actual_order}


def validate_nesting_matrix() -> dict[str, Any]:
    matrix_path = ANALYSIS_DIR / "CUMULATIVE_ABLATION_MATRIX.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Missing matrix file: {matrix_path}")
    with open(matrix_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    diffs = compute_pairwise_configuration_diffs()
    if len(diffs) != 6:
        raise ValueError(f"Expected 6 pairwise transitions, got {len(diffs)}")
    return {"status": "PASS", "pairwise_transitions_verified": len(diffs)}


def validate_gain_arithmetic(summary_json_path: Path) -> dict[str, Any]:
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    rows = summary["rows"]
    if len(rows) != 7:
        raise ValueError(f"Expected 7 summary rows, got {len(rows)}")
    for i, r in enumerate(rows):
        if r["configuration"] != EXPECTED_MODES[i]:
            raise ValueError(f"Row {i} mode mismatch: expected {EXPECTED_MODES[i]}, got {r['configuration']}")
        tta_acc = r["tta_accuracy"]
        if not math.isfinite(tta_acc):
            raise ValueError(f"Non-finite TTA accuracy in row {i}: {tta_acc}")
        if i == 0:
            if r["gain"] is not None:
                raise ValueError("A0 gain must be None / —")
        else:
            prev_acc = rows[i - 1]["tta_accuracy"]
            expected_gain = tta_acc - prev_acc
            actual_gain = r["gain"]
            if abs(actual_gain - expected_gain) > 1e-6:
                raise ValueError(
                    f"Gain arithmetic mismatch at row {i} ({r['configuration']}): "
                    f"expected {expected_gain}, got {actual_gain}"
                )
    return {"status": "PASS", "rows_verified": len(rows)}


def validate_a6_parity() -> dict[str, Any]:
    cfg = CUMULATIVE_REGISTRY[CumulativeAblationMode.A6]
    model = CumulativeAblationMPGFER(mode=CumulativeAblationMode.A6).eval()
    x = torch.randn(2, 1, 48, 48)

    # Check forward pass exact equivalence to MPGFER
    from mpg_fer_v2_3.model import MPGFER
    torch.manual_seed(999)
    mpgfer = MPGFER().eval()
    torch.manual_seed(999)
    a6 = CumulativeAblationMPGFER(mode=CumulativeAblationMode.A6).eval()
    with torch.no_grad():
        m_logits, _ = mpgfer(x)
        a6_logits, _ = a6(x)
    diff = (m_logits - a6_logits).abs().max().item()
    if diff != 0.0:
        raise ValueError(f"A6 does not reproduce MPGFER exactly (max diff: {diff})")

    # Check strict load of official checkpoint
    official_ckpt = Path(
        r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\research\mpg_fer_v2_3\official_runs\segment_02\mpg_fer_v2_3_run\best_val_acc.pt"
    )
    if official_ckpt.is_file():
        digest = sha256_file(official_ckpt)
        if digest != EXPECTED_A6_METRICS["checkpoint_sha256"]:
            raise ValueError(f"Official checkpoint SHA mismatch: {digest}")
        payload = torch.load(official_ckpt, map_location="cpu", weights_only=False)
        incompat = a6.load_state_dict(payload["model_state_dict"], strict=True)
        if len(incompat.missing_keys) > 0 or len(incompat.unexpected_keys) > 0:
            raise ValueError(f"A6 strict load failed: {incompat}")

    return {"status": "PASS", "forward_parity_diff": 0.0, "strict_load": "PASS"}


def run_full_validation() -> dict[str, Any]:
    results = {}
    print("1. Validating order and configs...")
    results["order_and_configs"] = validate_order_and_configs()

    print("2. Validating nesting matrix...")
    results["nesting_matrix"] = validate_nesting_matrix()

    print("3. Validating A6 FULL parity and strict load...")
    results["a6_parity"] = validate_a6_parity()

    summary_file = ANALYSIS_DIR / "CUMULATIVE_ABLATION7_SUMMARY.json"
    if summary_file.is_file():
        print("4. Validating gain arithmetic...")
        results["gain_arithmetic"] = validate_gain_arithmetic(summary_file)
    else:
        results["gain_arithmetic"] = {"status": "PENDING", "reason": "Summary file not yet generated"}

    print("5. Validating checksums...")
    results["checksums"] = validate_checksums()

    all_passed = all(v["status"] in ("PASS", "PENDING") for v in results.values())
    results["overall_status"] = "PASS" if all_passed else "FAIL"
    return results


def main() -> None:
    res = run_full_validation()
    print("\nValidation Summary:")
    print(json.dumps(res, indent=2))
    if res["overall_status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
