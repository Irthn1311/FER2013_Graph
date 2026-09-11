"""Tests verifying streaming FER CSV validation, malformed input rejection, and data governance."""

import tempfile
from pathlib import Path
import pytest

from pure_gnn_v31.scientific.dataset import validate_and_hash_fer_csv, load_fer_csv_split
from pure_gnn_v31.scientific.governance import (
    assert_not_test_access,
    validate_dataset_path,
    DataGovernanceError,
)
from pure_gnn_v31.scientific.evaluator import ScientificEvaluator
from pure_gnn_v31.model import PureGNNv31


def make_synthetic_fer_csv(
    rows: list,
    filename: str = "train.csv",
    header: str = "emotion,pixels\n",
) -> Path:
    # Use delete=False or keep directory alive
    tmp_dir = tempfile.mkdtemp()
    p = Path(tmp_dir) / filename
    with p.open("w", encoding="utf-8") as f:
        f.write(header)
        for r in rows:
            f.write(r + "\n")
    return p


def test_streaming_csv_validation_valid():
    valid_pixels = " ".join(["128.0"] * 2304)
    rows = [f"0,{valid_pixels}", f"3,{valid_pixels}"]
    csv_path = make_synthetic_fer_csv(rows, filename="train.csv")

    info = validate_and_hash_fer_csv(csv_path, expected_role="train", expected_rows=2)
    assert info["row_count"] == 2
    assert len(info["sha256"]) == 64


def test_streaming_csv_validation_bad_label():
    valid_pixels = " ".join(["100"] * 2304)
    # Emotion 7 is invalid (only 0..6 allowed)
    rows = [f"7,{valid_pixels}"]
    csv_path = make_synthetic_fer_csv(rows, filename="train.csv")

    with pytest.raises(ValueError) as exc:
        validate_and_hash_fer_csv(csv_path, expected_role="train")
    assert "out of range [0, 6]" in str(exc.value)


def test_streaming_csv_validation_wrong_pixel_count():
    # Only 100 pixels instead of 2304
    bad_pixels = " ".join(["50"] * 100)
    rows = [f"2,{bad_pixels}"]
    csv_path = make_synthetic_fer_csv(rows, filename="train.csv")

    with pytest.raises(ValueError) as exc:
        validate_and_hash_fer_csv(csv_path, expected_role="train")
    assert "Pixel count mismatch" in str(exc.value)


def test_streaming_csv_validation_pixel_below_zero():
    pixels = ["10"] * 2303 + ["-1.5"]
    rows = [f"1,{' '.join(pixels)}"]
    csv_path = make_synthetic_fer_csv(rows, filename="train.csv")

    with pytest.raises(ValueError) as exc:
        validate_and_hash_fer_csv(csv_path, expected_role="train")
    assert "out of range [0, 255]" in str(exc.value)


def test_streaming_csv_validation_pixel_above_255():
    pixels = ["10"] * 2303 + ["256.0"]
    rows = [f"1,{' '.join(pixels)}"]
    csv_path = make_synthetic_fer_csv(rows, filename="train.csv")

    with pytest.raises(ValueError) as exc:
        validate_and_hash_fer_csv(csv_path, expected_role="train")
    assert "out of range [0, 255]" in str(exc.value)


def test_streaming_csv_validation_nan():
    pixels = ["10"] * 2303 + ["NaN"]
    rows = [f"4,{' '.join(pixels)}"]
    csv_path = make_synthetic_fer_csv(rows, filename="train.csv")

    with pytest.raises(ValueError) as exc:
        validate_and_hash_fer_csv(csv_path, expected_role="train")
    assert "Non-finite pixel value" in str(exc.value)


def test_streaming_csv_validation_wrong_row_count():
    valid_pixels = " ".join(["128.0"] * 2304)
    rows = [f"0,{valid_pixels}"]
    csv_path = make_synthetic_fer_csv(rows, filename="train.csv")

    with pytest.raises(DataGovernanceError) as exc:
        validate_and_hash_fer_csv(csv_path, expected_role="train", expected_rows=28709)
    assert "Row count mismatch" in str(exc.value)


def test_test_role_and_path_strictly_rejected():
    model = PureGNNv31(condition="G1")
    evaluator = ScientificEvaluator(model)

    import tensorflow as tf
    dummy_ds = tf.data.Dataset.from_tensor_slices((tf.zeros([2, 48, 48, 1]), tf.zeros([2], dtype=tf.int32))).batch(2)

    with pytest.raises(DataGovernanceError):
        evaluator.evaluate_split(dummy_ds, split_role="test")

    with pytest.raises(DataGovernanceError):
        validate_dataset_path("data/test.csv", expected_role="train")

    with pytest.raises(DataGovernanceError):
        assert_not_test_access("path/to/PrivateTest.csv")
