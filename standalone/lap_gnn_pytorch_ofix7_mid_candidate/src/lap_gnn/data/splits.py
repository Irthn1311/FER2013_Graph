"""Locked FER2013 split contract."""

from lap_gnn.constants import SPLIT_COUNTS


def expected_split_counts() -> dict[str, int]:
    return dict(SPLIT_COUNTS)
