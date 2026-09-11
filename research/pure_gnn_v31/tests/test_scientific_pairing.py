"""Tests verifying paired dataset ordering, augmentation reproducibility, and G0.5/G1 synchronization."""

import numpy as np
import pytest
import tensorflow as tf
from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.scientific.dataset import create_paired_dataset
from pure_gnn_v31.scientific.initialization import (
    synchronize_shared_parameters,
    verify_initialization_equivalence,
)


def test_paired_dataset_identical_ordering():
    # Synthetic dataset
    images = np.arange(100 * 48 * 48, dtype=np.float32).reshape(100, 48, 48, 1)
    labels = np.arange(100, dtype=np.int32) % 7

    ds1 = create_paired_dataset(images, labels, batch_size=16, seed=12345, shuffle=True)
    ds2 = create_paired_dataset(images, labels, batch_size=16, seed=12345, shuffle=True)

    for (b1_x, b1_y), (b2_x, b2_y) in zip(ds1, ds2):
        np.testing.assert_array_equal(b1_x.numpy(), b2_x.numpy())
        np.testing.assert_array_equal(b1_y.numpy(), b2_y.numpy())


def test_g05_g1_shared_initialization_synchronization():
    # Model G0.5 and Model G1
    model_g05 = PureGNNv31(condition="G0.5")
    model_g1 = PureGNNv31(condition="G1")

    # Build models with dummy input
    dummy = tf.random.normal([2, 48, 48, 1], seed=99)
    _ = model_g05(dummy, training=False)
    _ = model_g1(dummy, training=False)

    # Before synchronization, weights are randomly initialized differently
    diff_before = tf.reduce_max(tf.abs(model_g05(dummy, training=False) - model_g1(dummy, training=False)))
    assert float(diff_before.numpy()) > 1e-4

    # Synchronize shared parameters from G0.5 to G1
    synced = synchronize_shared_parameters(model_g05, model_g1)
    assert synced > 50

    # After synchronization, G0.5 and G1 must produce identical output within 1e-5
    equiv, max_diff = verify_initialization_equivalence(model_g05, model_g1, dummy, tolerance=1e-5)
    assert equiv is True, f"Max difference {max_diff} exceeds 1e-5"
