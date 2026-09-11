"""Tests verifying historical protocol contracts, normalization divergence, and role governance."""

import json
from pathlib import Path
import pytest

from pure_gnn_v31.scientific.config import load_scientific_config
from pure_gnn_v31.scientific.governance import (
    validate_split_row_counts,
    validate_dataset_path,
    assert_not_test_access,
    DataGovernanceError,
)


def test_historical_split_row_counts_and_no_internal_split():
    config = load_scientific_config()

    # 1. No internal train split exists in scientific protocol
    assert config.raw_config["data_protocol"]["internal_train_split"] is False
    assert config.raw_config["data_protocol"]["research_dev_split"] is False

    # 2. Expected row counts are 28,709 for Train and 3,589 for Val
    assert config.train_rows == 28709
    assert config.val_rows == 3589

    # 3. Validation assertions
    validate_split_row_counts("train", 28709, 28709)
    validate_split_row_counts("validation", 3589, 3589)

    with pytest.raises(DataGovernanceError):
        validate_split_row_counts("train", 24000, 28709)

    with pytest.raises(DataGovernanceError):
        validate_split_row_counts("validation", 4000, 3589)


def test_no_test_sha_field_exists():
    """Asserts that no test hash field exists in scientific config or data protocol."""
    config = load_scientific_config()
    proto = config.raw_config["data_protocol"]
    assert "test_sha256" not in proto
    assert "test_sha" not in proto
    assert "test_checksum" not in proto


def test_historical_normalization_divergence_recorded():
    """Verifies that audit correctly records divergent normalization across LAP, Gen2, and Gen3."""
    audit_json_path = (
        Path(__file__).resolve().parents[1]
        / "HISTORICAL_PROTOCOL_AUDIT.json"
    )
    assert audit_json_path.is_file(), f"Audit JSON missing: {audit_json_path}"

    with audit_json_path.open("r", encoding="utf-8") as f:
        audit = json.load(f)

    # LAP normalization citation
    lap_norm = audit["LAP"]["pixel_normalization"]["value"]
    assert "255.0" in lap_norm or "image / 255.0" in lap_norm
    assert "graph/builder.py:569" in audit["LAP"]["pixel_normalization"]["evidence"][0]

    # Gen2 CF-HPG normalization is [-1, 1]
    gen2_norm = audit["Gen2_CF_RA"]["pixel_normalization"]["value"]
    assert "[-1, 1]" in gen2_norm
    assert "patchify_and_scale" in audit["Gen2_CF_RA"]["pixel_normalization"]["evidence"][0]

    # Gen3 WS-HPG normalization is [0, 1]
    gen3_norm = audit["Gen3_WS_HPG"]["pixel_normalization"]["value"]
    assert "[0, 1]" in gen3_norm

    # LAP image augmentation is UNKNOWN, while prior corruption is SOURCE_CONFIRMED
    assert audit["LAP"]["image_augmentation"]["value"] == "UNKNOWN"
    assert "attenuate_prior" in audit["LAP"]["structural_prior_corruption"]["value"]


def test_role_and_path_mismatch_governance():
    """Enforces:
    - train.csv as train -> accepted
    - val.csv as validation -> accepted
    - val.csv as train -> rejected
    - train.csv as validation -> rejected
    - test.csv -> rejected
    """
    # 1. train.csv as train -> accepted
    p_train = validate_dataset_path("path/to/train.csv", expected_role="train")
    assert p_train.name == "train.csv"

    # 2. val.csv as validation -> accepted
    p_val = validate_dataset_path("path/to/val.csv", expected_role="validation")
    assert p_val.name == "val.csv"

    # 3. val.csv as train -> rejected
    with pytest.raises(DataGovernanceError) as exc_info:
        validate_dataset_path("path/to/val.csv", expected_role="train")
    assert "Role-path mismatch" in str(exc_info.value)

    # 4. train.csv as validation -> rejected
    with pytest.raises(DataGovernanceError) as exc_info:
        validate_dataset_path("path/to/train.csv", expected_role="validation")
    assert "Role-path mismatch" in str(exc_info.value)

    # 5. test.csv -> rejected for any role
    with pytest.raises(DataGovernanceError) as exc_info:
        validate_dataset_path("path/to/test.csv", expected_role="train")
    assert "STRICT DATA GOVERNANCE VIOLATION" in str(exc_info.value)

    with pytest.raises(DataGovernanceError) as exc_info:
        validate_dataset_path("path/to/test.csv", expected_role="validation")
    assert "STRICT DATA GOVERNANCE VIOLATION" in str(exc_info.value)

    with pytest.raises(DataGovernanceError):
        assert_not_test_access("data/test.csv")


def test_true_binary_sha256_with_crlf_fixture():
    """P0-1: Verifies that hashing uses exact raw binary chunk streaming and preserves CRLF bytes."""
    import hashlib
    import tempfile
    from pure_gnn_v31.scientific.dataset import validate_and_hash_fer_csv, load_fer_csv_split

    tmp_dir = tempfile.mkdtemp()
    crlf_csv_path = Path(tmp_dir) / "train.csv"

    # Construct explicit CRLF bytes
    row1 = "0," + " ".join(["100.0"] * 2304) + "\r\n"
    header = "emotion,pixels\r\n"
    raw_content = (header + row1).encode("utf-8")
    crlf_csv_path.write_bytes(raw_content)

    expected_sha = hashlib.sha256(raw_content).hexdigest()
    assert crlf_csv_path.read_bytes() == raw_content

    # 1. validate_and_hash_fer_csv must return exact binary hash
    val_info = validate_and_hash_fer_csv(crlf_csv_path, expected_role="train", expected_rows=1)
    assert val_info["sha256"] == expected_sha
    assert val_info["sha256"] == hashlib.sha256(crlf_csv_path.read_bytes()).hexdigest()

    # 2. load_fer_csv_split must return exact whole-file binary hash
    _, _, _, load_sha = load_fer_csv_split(crlf_csv_path, role="train", validate_row_count=False)
    assert load_sha == expected_sha
    assert load_sha == hashlib.sha256(crlf_csv_path.read_bytes()).hexdigest()
