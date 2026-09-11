"""Data governance and split isolation contracts for Pure-GNN scientific line."""

from pathlib import Path
from typing import Sequence, Union


class DataGovernanceError(PermissionError):
    """Raised when data governance rules or test set isolation boundaries are violated."""


FORBIDDEN_TEST_LEXICAL_SUBSTRINGS = [
    "test.csv",
    "/test/",
    "\\test\\",
    "official_test",
    "privatetest",
    "privatetest.csv",
]


def assert_not_test_access(path: Union[str, Path]) -> None:
    """Rejects any path that references or targets the official test holdout set."""
    p_str = str(path).lower().replace("\\", "/")
    for sub in FORBIDDEN_TEST_LEXICAL_SUBSTRINGS:
        if sub in p_str:
            raise DataGovernanceError(
                f"STRICT DATA GOVERNANCE VIOLATION: Test holdout access is forbidden! Path: {path}"
            )


def validate_dataset_path(path: Union[str, Path], expected_role: str) -> Path:
    """Validates allowed dataset paths based on expected split role ('train' or 'validation')."""
    if expected_role.lower() not in ("train", "validation"):
        raise DataGovernanceError(
            f"Invalid dataset role '{expected_role}'. Only 'train' and 'validation' are authorized."
        )
    assert_not_test_access(path)
    p = Path(path)
    return p


def validate_split_row_counts(role: str, actual_count: int, expected_count: int) -> None:
    """Asserts exact match with historical row counts (28,709 for train, 3,589 for val)."""
    if actual_count != expected_count:
        raise DataGovernanceError(
            f"Row count mismatch for {role} set: observed {actual_count} rows, expected {expected_count}."
        )
