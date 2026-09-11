"""Tests verifying data governance, test isolation, and rejection of test role/path."""

import pytest
from pathlib import Path
from pure_gnn_v31.scientific.governance import (
    assert_not_test_access,
    validate_dataset_path,
    DataGovernanceError,
)
from pure_gnn_v31.scientific.evaluator import ScientificEvaluator
from pure_gnn_v31.model import PureGNNv31


def test_test_role_rejected():
    model = PureGNNv31(condition="G1")
    evaluator = ScientificEvaluator(model)

    import tensorflow as tf
    dummy_ds = tf.data.Dataset.from_tensor_slices((tf.zeros([2, 48, 48, 1]), tf.zeros([2], dtype=tf.int32))).batch(2)

    with pytest.raises(DataGovernanceError) as exc:
        evaluator.evaluate_split(dummy_ds, split_role="test")
    assert "STRICT DATA GOVERNANCE VIOLATION" in str(exc.value)

    with pytest.raises(DataGovernanceError) as exc:
        evaluator.evaluate_split(dummy_ds, split_role="official_test")
    assert "STRICT DATA GOVERNANCE VIOLATION" in str(exc.value)


def test_test_path_rejected():
    forbidden_paths = [
        "data/test.csv",
        "data/test/image.png",
        "/kaggle/input/datasets/doduyquynii/fer13-split/test.csv",
        "test.csv",
        "official_test.csv",
        "path/PrivateTest/img.png",
    ]
    for path in forbidden_paths:
        with pytest.raises(DataGovernanceError):
            assert_not_test_access(path)
        with pytest.raises(DataGovernanceError):
            validate_dataset_path(path, expected_role="train")


def test_validation_role_accepted_but_not_test():
    p = validate_dataset_path("data/val.csv", expected_role="validation")
    assert p == Path("data/val.csv")

    with pytest.raises(DataGovernanceError):
        validate_dataset_path("data/val.csv", expected_role="test")
