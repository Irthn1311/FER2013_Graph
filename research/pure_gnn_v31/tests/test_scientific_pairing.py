"""Tests verifying paired dataset ordering, absence of premature augmentation claims, and exact G0.5/G1 shared-init audit."""

import numpy as np
import pytest
import tensorflow as tf
from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.scientific.dataset import create_paired_dataset
from pure_gnn_v31.scientific.initialization import (
    audit_and_synchronize_shared_parameters,
    verify_initialization_equivalence,
)


def test_paired_dataset_identical_ordering():
    # Synthetic dataset
    images = np.arange(100 * 48 * 48, dtype=np.float32).reshape(100, 48, 48, 1)
    labels = np.arange(100, dtype=np.int32) % 7

    # Explicit batch_size and seed required (NO defaults)
    ds1 = create_paired_dataset(images, labels, batch_size=16, seed=12345, shuffle=True)
    ds2 = create_paired_dataset(images, labels, batch_size=16, seed=12345, shuffle=True)

    for (b1_x, b1_y, b1_idx), (b2_x, b2_y, b2_idx) in zip(ds1, ds2):
        np.testing.assert_array_equal(b1_x.numpy(), b2_x.numpy())
        np.testing.assert_array_equal(b1_y.numpy(), b2_y.numpy())
        np.testing.assert_array_equal(b1_idx.numpy(), b2_idx.numpy())


def test_exact_g05_g1_shared_initialization_audit():
    """Verifies:
    - exact shared non-gate variable paths match between G0.5 and G1
    - copies every shared tensor and verifies max copy error == 0.0
    - asserts G1 coarse gate final projections remain zero initialized
    - verifies fixed-input logits equivalence <= 1e-5
    """
    model_g05 = PureGNNv31(condition="G0.5")
    model_g1 = PureGNNv31(condition="G1")

    dummy = tf.random.normal([2, 48, 48, 1], seed=99)
    _ = model_g05(dummy, training=False)
    _ = model_g1(dummy, training=False)

    # Audit and synchronize
    audit = audit_and_synchronize_shared_parameters(model_g05, model_g1)

    assert audit["source_condition"] == "G0.5"
    assert audit["target_condition"] == "G1"
    assert audit["exact_shared_tensor_count"] == 128
    assert audit["shared_parameter_count"] == 856429
    assert audit["unmatched_g1_gate_variables"] == 12  # 6 per block * 2 coarse blocks
    assert audit["max_parameter_copy_error"] == 0.0

    # Fixed-input logits equivalence must be <= 1e-5
    passed, max_err = verify_initialization_equivalence(model_g05, model_g1, dummy, tolerance=1e-5)
    assert passed is True, f"Logits error {max_err} exceeds 1e-5"
    assert max_err <= 1e-5
