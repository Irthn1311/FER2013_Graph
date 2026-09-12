"""Tests for exact checkpoint weights and 0-based / 1-based epoch tracking."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.scientific.checkpoints import CheckpointSelector


FUNCTIONAL_LOGIT_TOLERANCE = 1e-5
ORIGINAL_MODEL_SEED = 1234
FRESH_MODEL_SEED = 1235


def _weight_identity(variable, index):
    """Return a stable identity for a serialized Keras weight variable."""
    path = getattr(variable, "path", None)
    if path:
        return str(path)
    shape = tuple(int(dim) for dim in variable.shape)
    return f"{variable.name}|shape={shape}|index={index}"


def _snapshot_checkpoint_weights(model):
    """Snapshot every tensor represented by the model.weights checkpoint contract."""
    snapshot = {}
    for index, variable in enumerate(model.weights):
        identity = _weight_identity(variable, index)
        assert identity not in snapshot, f"Duplicate checkpoint weight identity: {identity}"
        value = variable.numpy().copy()
        snapshot[identity] = {
            "dtype": str(value.dtype),
            "shape": tuple(int(dim) for dim in value.shape),
            "value": value,
        }
    return snapshot


@pytest.mark.parametrize("condition", ["G0", "G0.5", "G1"])
def test_checkpoint_weights_roundtrip(condition):
    """Require exact restored weights before allowing numerical logit tolerance."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(ORIGINAL_MODEL_SEED)
    fixed_input = tf.convert_to_tensor(
        np.random.default_rng(ORIGINAL_MODEL_SEED)
        .normal(size=(2, 48, 48, 1))
        .astype(np.float32)
    )
    model_orig = PureGNNv31(condition=condition)
    logits_before = model_orig(fixed_input, training=False).numpy()
    original_weights = _snapshot_checkpoint_weights(model_orig)

    with tempfile.TemporaryDirectory() as tmp_dir:
        weights_path = Path(tmp_dir) / f"model_{condition}.weights.h5"
        model_orig.save_weights(str(weights_path))

        # Use a different deterministic initialization so equality can only come
        # from loading the checkpoint, not from identical fresh initialization.
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(FRESH_MODEL_SEED)
        model_fresh = PureGNNv31(condition=condition)
        _ = model_fresh(fixed_input, training=False)
        model_fresh.load_weights(str(weights_path))
        restored_weights = _snapshot_checkpoint_weights(model_fresh)

    assert len(restored_weights) == len(original_weights)
    assert list(restored_weights) == list(original_weights), (
        "Checkpoint weight identity/order mapping changed"
    )

    max_parameter_difference = 0.0
    for identity, original in original_weights.items():
        restored = restored_weights[identity]
        assert restored["shape"] == original["shape"], f"Shape changed for {identity}"
        assert restored["dtype"] == original["dtype"], f"Dtype changed for {identity}"
        np.testing.assert_array_equal(
            restored["value"],
            original["value"],
            err_msg=f"Checkpoint tensor changed for {identity}",
        )
        tensor_difference = float(
            np.max(np.abs(restored["value"] - original["value"]))
        )
        max_parameter_difference = max(max_parameter_difference, tensor_difference)

    assert max_parameter_difference == 0.0

    logits_after = model_fresh(fixed_input, training=False).numpy()
    max_logit_error = float(np.max(np.abs(logits_before - logits_after)))
    assert max_logit_error <= FUNCTIONAL_LOGIT_TOLERANCE, (
        f"Condition {condition} roundtrip logit error {max_logit_error} "
        f"exceeds {FUNCTIONAL_LOGIT_TOLERANCE}"
    )


def test_epoch_number_semantics():
    """Verifies that both 0-based index and 1-based epoch number are tracked accurately.

    Requires explicit monitor and mode (no defaults).
    """
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


def test_checkpoint_selector_requires_explicit_monitor_and_mode():
    """Asserts that CheckpointSelector raises error when monitor or mode is missing/empty."""
    with pytest.raises(ValueError):
        CheckpointSelector(monitor="", mode="max")
    with pytest.raises(ValueError):
        CheckpointSelector(monitor="val_accuracy", mode="")
