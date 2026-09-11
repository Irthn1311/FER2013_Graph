"""Tests verifying historical protocol contracts (train=28709, val=3589, no internal split)."""

import pytest
from pure_gnn_v31.scientific.config import load_scientific_config
from pure_gnn_v31.scientific.governance import validate_split_row_counts, DataGovernanceError


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
