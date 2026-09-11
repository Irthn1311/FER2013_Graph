"""Tests verifying weights checkpoint round-trip (max error <= 1e-7) and 0-based / 1-based epoch tracking."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.scientific.checkpoints import CheckpointSelector


@pytest.mark.parametrize("condition", ["G0", "G0.5", "G1"])
def test_checkpoint_weights_roundtrip(condition):
    """Verifies that save_weights / load_weights round-trip produces exact logits (max_abs_error <= 1e-7)."""
    model_orig = PureGNNv31(condition=condition)
    fixed_input = tf.random.normal([2, 48, 48, 1], seed=1234)

    # 1. Obtain logits before saving
    logits_before = model_orig(fixed_input, training=False).numpy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        weights_path = Path(tmp_dir) / f"model_{condition}.weights.h5"
        model_orig.save_weights(str(weights_path))

        # 2. Instantiate fresh model of the same condition
        model_fresh = PureGNNv31(condition=condition)
        # Build fresh model
        _ = model_fresh(fixed_input, training=False)

        # 3. Load weights
        model_fresh.load_weights(str(weights_path))

        # 4. Obtain logits after restoring
        logits_after = model_fresh(fixed_input, training=False).numpy()

        max_err = float(np.max(np.abs(logits_before - logits_after)))
        assert max_err <= 1e-7, f"Condition {condition} roundtrip error {max_err} exceeds 1e-7"


def test_epoch_number_semantics():
    """Verifies that both 0-based index and 1-based epoch number are tracked accurately."""
    selector = CheckpointSelector(monitor="val_accuracy", mode="max")

    # Epoch 0 (0-based) is Epoch 1 (1-based)
    assert selector.update(epoch_zero_based=0, metrics={"val_accuracy": 0.52}) is True
    assert selector.selected_epoch_index_zero_based == 0
    assert selector.selected_epoch_number_one_based == 1

    # Epoch 1 (0-based) is Epoch 2 (1-based) - worse
    assert selector.update(epoch_zero_based=1, metrics={"val_accuracy": 0.50}) is False
    assert selector.selected_epoch_index_zero_based == 0
    assert selector.selected_epoch_number_one_based == 1

    # Epoch 2 (0-based) is Epoch 3 (1-based) - strictly better
    assert selector.update(epoch_zero_based=2, metrics={"val_accuracy": 0.58}) is True
    assert selector.selected_epoch_index_zero_based == 2
    assert selector.selected_epoch_number_one_based == 3
