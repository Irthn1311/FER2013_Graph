"""Unit tests enforcing data governance and strict isolation of official test.csv."""

import pytest
import os
import subprocess
import sys
from pathlib import Path
from pure_gnn_v31.data import (
    assert_not_test_access,
    assert_preflight_train_only_path,
    create_research_split_manifest,
    load_fer2013_train_csv,
)


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


@pytest.mark.parametrize("path", [
    "data/val.csv",
    "/data/val/images",
    "official_val/cache",
    "official_test/cache",
    "publictest/records",
    "privatetest/records",
    "/data/test/train.csv",
])
def test_preflight_train_only_guard_rejects_val_and_test_lexically(path):
    with pytest.raises(PermissionError):
        assert_preflight_train_only_path(path)


def test_preflight_train_only_guard_accepts_approved_train_shape_without_io():
    assert_preflight_train_only_path("/kaggle/input/fer13-split/train.csv")


def test_split_api_has_no_seed_or_ratio_defaults():
    import inspect

    signature = inspect.signature(create_research_split_manifest)
    assert signature.parameters["seed"].default is inspect.Parameter.empty
    assert signature.parameters["dev_ratio"].default is inspect.Parameter.empty


def test_split_cli_requires_explicit_seed_and_dev_ratio_before_data_access():
    package_root = Path(__file__).resolve().parents[1]
    script = package_root / "tools" / "create_research_split.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root / "src")
    result = subprocess.run(
        [sys.executable, str(script), "--train-csv", "train.csv", "--output", "unused.json"],
        cwd=package_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 2
    assert "--seed" in result.stdout
    assert "--dev-ratio" in result.stdout
