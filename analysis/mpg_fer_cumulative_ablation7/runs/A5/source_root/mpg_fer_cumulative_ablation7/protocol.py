"""Preregistered cumulative ablation protocol, dataset checks, and pairwise diff validation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .model import CUMULATIVE_ABLATION_ORDER, CUMULATIVE_REGISTRY, CumulativeAblationMode, CumulativeAblationSpec

CANONICAL_DATASET_HASHES: dict[str, str] = {
    "train.csv": "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8",
    "val.csv": "412036d077c6ec203047b2935ab14bc858d8136ee26e8db3e23023f1fc9dee08",
    "test.csv": "be385344b93606c75cb65cac3d39b13cdedb5b91621640ee08f73b9a551fbd9d",
}

CANONICAL_DATASET_ROWS: dict[str, int] = {
    "train.csv": 28709,
    "val.csv": 3589,
    "test.csv": 3589,
}

EXPECTED_A6_METRICS = {
    "raw_accuracy": 0.6876567288938423,
    "raw_macro_f1": 0.6734818933345407,
    "tta_accuracy": 0.7066035107272220,
    "tta_macro_f1": 0.6981583150632577,
    "checkpoint_sha256": "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e",
    "selected_epoch": 57,
}


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def count_csv_rows(path: Path | str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        return sum(1 for _ in reader)


def validate_split_identity(name: str, path: Path | str) -> dict[str, Any]:
    """Fail closed on dataset identity using both SHA256 and row count."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Required dataset file missing: {p}")
    expected_hash = CANONICAL_DATASET_HASHES[name]
    expected_rows = CANONICAL_DATASET_ROWS[name]
    actual_hash = sha256_file(p)
    actual_rows = count_csv_rows(p)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Dataset SHA256 mismatch for {name}: expected {expected_hash}, got {actual_hash}"
        )
    if actual_rows != expected_rows:
        raise ValueError(
            f"Dataset row count mismatch for {name}: expected {expected_rows}, got {actual_rows}"
        )
    return {
        "file": name,
        "path": str(p.resolve()),
        "rows": actual_rows,
        "sha256": actual_hash,
        "status": "PASS",
    }


def validate_all_splits(train_path: Path, val_path: Path, test_path: Path) -> dict[str, Any]:
    train_res = validate_split_identity("train.csv", train_path)
    val_res = validate_split_identity("val.csv", val_path)
    test_res = validate_split_identity("test.csv", test_path)
    return {
        "train": train_res,
        "val": val_res,
        "test": test_res,
        "all_passed": True,
    }


def compute_pairwise_configuration_diffs() -> list[dict[str, Any]]:
    """Generate and validate pairwise configuration diff: A0->A1, A1->A2, ..., A5->A6.
    
    Each transition must contain exactly the intended architectural addition.
    """
    diffs = []
    order = list(CUMULATIVE_ABLATION_ORDER)
    for i in range(len(order) - 1):
        prev_mode = order[i]
        next_mode = order[i + 1]
        prev_spec = CUMULATIVE_REGISTRY[prev_mode]
        next_spec = CUMULATIVE_REGISTRY[next_mode]

        prev_dict = asdict(prev_spec)
        next_dict = asdict(next_spec)

        changed_keys = {}
        for k in prev_dict:
            if k in ("internal_id", "paper_name", "vietnamese_name", "cumulative_description", "notes"):
                continue
            if prev_dict[k] != next_dict[k]:
                changed_keys[k] = {"from": prev_dict[k], "to": next_dict[k]}

        # Check intended architectural change per step
        intended_change = None
        if prev_mode == CumulativeAblationMode.A0 and next_mode == CumulativeAblationMode.A1:
            intended_change = "Add Spatial Motif Composer (12x12) + Fixed pooling + enable motif losses"
            expected_keys = {"composer", "composer_scales", "motif_pooling", "applicable_losses"}
        elif prev_mode == CumulativeAblationMode.A1 and next_mode == CumulativeAblationMode.A2:
            intended_change = "Enable Motif GNN (dense relations)"
            expected_keys = {"motif_gnn", "relation_mode", "motif_topk_schedule"}
        elif prev_mode == CumulativeAblationMode.A2 and next_mode == CumulativeAblationMode.A3:
            intended_change = "Enable Geometry Bias in Motif GNN"
            expected_keys = {"geometry_bias"}
        elif prev_mode == CumulativeAblationMode.A3 and next_mode == CumulativeAblationMode.A4:
            intended_change = "Enable Multi-scale Composition (8, 12, 16)"
            expected_keys = {"composer", "composer_scales"}
        elif prev_mode == CumulativeAblationMode.A4 and next_mode == CumulativeAblationMode.A5:
            intended_change = "Enable Dynamic Top-K Relations [8, 16, 16, 16, 24]"
            expected_keys = {"relation_mode", "motif_topk_schedule"}
        elif prev_mode == CumulativeAblationMode.A5 and next_mode == CumulativeAblationMode.A6:
            intended_change = "Replace Fixed Pooling with Learnable Attention Pooling (Full Model)"
            expected_keys = {"motif_pooling"}
        else:
            raise ValueError(f"Unexpected step: {prev_mode} -> {next_mode}")

        actual_keys = set(changed_keys.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"Pairwise diff violation {prev_mode.value}->{next_mode.value}: "
                f"expected change in {expected_keys}, but got {actual_keys} with diffs: {changed_keys}"
            )

        diffs.append({
            "transition": f"{prev_mode.value}->{next_mode.value}",
            "from_config": prev_spec.paper_name,
            "to_config": next_spec.paper_name,
            "intended_change": intended_change,
            "changed_fields": changed_keys,
            "status": "VALID",
        })
    return diffs


def build_cumulative_ablation_matrix() -> dict[str, Any]:
    rows = []
    for mode in CUMULATIVE_ABLATION_ORDER:
        spec = CUMULATIVE_REGISTRY[mode]
        rows.append({
            "configuration": spec.internal_id,
            "display_name": spec.paper_name,
            "vietnamese_name": spec.vietnamese_name,
            "pixel_gnn": spec.pixel_gnn,
            "composer": spec.composer or "None",
            "composer_scales": spec.composer_scales,
            "motif_gnn": spec.motif_gnn,
            "geometry_bias": spec.geometry_bias,
            "relation_mode": spec.relation_mode,
            "motif_topk_schedule": spec.motif_topk_schedule,
            "motif_pooling": spec.motif_pooling,
            "direct_pixel_fusion": spec.direct_pixel_fusion,
            "classifier": spec.classifier,
            "applicable_losses": spec.applicable_losses,
            "notes": spec.notes,
        })
    pairwise_diffs = compute_pairwise_configuration_diffs()
    return {
        "schema_version": 1,
        "method": "MPG-FER",
        "nested_ladder_property": "A0 ⊂ A1 ⊂ A2 ⊂ A3 ⊂ A4 ⊂ A5 ⊂ A6",
        "rows": rows,
        "pairwise_transitions": pairwise_diffs,
    }
