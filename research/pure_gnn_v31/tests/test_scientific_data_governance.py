"""Tests verifying streaming FER CSV validation, malformed input rejection, and data governance."""

import tempfile
from pathlib import Path
import numpy as np
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


def test_validation_two_tuple_dataset_rejected():
    """P0-2: Asserts that a two-tuple dataset without source_row_index is rejected with DataGovernanceError."""
    import tensorflow as tf
    # Fast dummy model to test index governance without slow full-GNN CPU forward passes
    dummy_model = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(7)])
    evaluator = ScientificEvaluator(dummy_model)

    # Dataset yielding only 2 elements: (image, label)
    two_tuple_ds = tf.data.Dataset.from_tensor_slices((
        tf.zeros([3589, 48, 48, 1], dtype=tf.float32),
        tf.zeros([3589], dtype=tf.int32),
    )).batch(256)

    with pytest.raises(DataGovernanceError) as exc:
        evaluator.evaluate_split(two_tuple_ds, split_role="validation", assert_full_validation_count=True)
    assert "Missing source index is strictly prohibited" in str(exc.value)


def test_validation_duplicate_indices_rejected():
    """P0-2: Asserts that duplicate source_row_index entries are rejected with DataGovernanceError."""
    import tensorflow as tf
    dummy_model = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(7)])
    evaluator = ScientificEvaluator(dummy_model)

    # Construct 3589 items but with a duplicate index (index 0 repeated, index 3588 omitted)
    dup_indices = np.arange(3589, dtype=np.int32)
    dup_indices[3588] = 0

    dup_ds = tf.data.Dataset.from_tensor_slices((
        tf.zeros([3589, 48, 48, 1], dtype=tf.float32),
        tf.zeros([3589], dtype=tf.int32),
        dup_indices,
    )).batch(256)

    with pytest.raises(DataGovernanceError) as exc:
        evaluator.evaluate_split(dup_ds, split_role="validation", assert_full_validation_count=True)
    assert "Duplicate validation indices detected" in str(exc.value)


def test_validation_index_gap_rejected():
    """P0-2: Asserts that an incomplete index set with gaps is rejected with DataGovernanceError."""
    import tensorflow as tf
    dummy_model = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(7)])
    evaluator = ScientificEvaluator(dummy_model)

    # Construct 3589 items but with gap: index 10 missing, index 3589 included
    gap_indices = np.arange(3589, dtype=np.int32)
    gap_indices[10] = 3589

    gap_ds = tf.data.Dataset.from_tensor_slices((
        tf.zeros([3589, 48, 48, 1], dtype=tf.float32),
        tf.zeros([3589], dtype=tf.int32),
        gap_indices,
    )).batch(256)

    with pytest.raises(DataGovernanceError) as exc:
        evaluator.evaluate_split(gap_ds, split_role="validation", assert_full_validation_count=True)
    assert "Validation index set is incomplete" in str(exc.value)


def test_validation_reordered_indices_accepted():
    """P0-2: Permuted/reordered indices that still form the complete 0..3588 set are accepted."""
    import tensorflow as tf
    dummy_model = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(7)])
    evaluator = ScientificEvaluator(dummy_model)

    # Shuffled 0..3588 permutation
    rng = np.random.RandomState(42)
    reordered_indices = np.arange(3589, dtype=np.int32)
    rng.shuffle(reordered_indices)

    reordered_ds = tf.data.Dataset.from_tensor_slices((
        tf.zeros([3589, 48, 48, 1], dtype=tf.float32),
        tf.zeros([3589], dtype=tf.int32),
        reordered_indices,
    )).batch(256)

    metrics, records = evaluator.evaluate_split(
        reordered_ds, split_role="validation", assert_full_validation_count=True
    )
    assert len(records["source_row_index"]) == 3589
    np.testing.assert_array_equal(records["source_row_index"], reordered_indices)
