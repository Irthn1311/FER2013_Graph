"""Unit tests enforcing data governance and strict isolation of official test.csv."""

import pytest
from pathlib import Path
from pure_gnn_v31.data import assert_not_test_access, load_fer2013_train_csv


def test_assert_not_test_access_triggers_on_test_paths():
    forbidden_paths = [
        "data/test.csv",
        "data/test/image.png",
        "/kaggle/input/datasets/doduyquynii/fer13-split/test.csv",
        "test.csv",
        "official_test.csv",
    ]
    for path in forbidden_paths:
        with pytest.raises(PermissionError) as exc_info:
            assert_not_test_access(path)
        assert "DATA GOVERNANCE VIOLATION" in str(exc_info.value)


def test_load_fer_train_rejects_test_csv():
    with pytest.raises(PermissionError):
        load_fer2013_train_csv("data/test.csv")
