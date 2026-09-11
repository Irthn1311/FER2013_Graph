"""Tests verifying checkpoint selection policy and obedience to injected monitor/mode."""

import pytest
from pure_gnn_v31.scientific.checkpoints import CheckpointSelector


def test_checkpoint_selector_max_accuracy():
    selector = CheckpointSelector(monitor="val_accuracy", mode="max")

    # Epoch 0: 0.50 -> new best
    assert selector.update(0, {"val_accuracy": 0.50}) is True
    assert selector.selected_epoch == 0
    assert selector.best_value == 0.50

    # Epoch 1: 0.49 -> worse
    assert selector.update(1, {"val_accuracy": 0.49}) is False
    assert selector.selected_epoch == 0

    # Epoch 2: 0.50 -> equal, not strictly better, earliest tie-break
    assert selector.update(2, {"val_accuracy": 0.50}) is False
    assert selector.selected_epoch == 0

    # Epoch 3: 0.55 -> strict improvement
    assert selector.update(3, {"val_accuracy": 0.55}) is True
    assert selector.selected_epoch == 3
    assert selector.best_value == 0.55


def test_checkpoint_selector_min_loss():
    selector = CheckpointSelector(monitor="val_loss", mode="min")

    assert selector.update(0, {"val_loss": 1.50}) is True
    assert selector.selected_epoch == 0

    assert selector.update(1, {"val_loss": 1.55}) is False
    assert selector.selected_epoch == 0

    # Strict improvement
    assert selector.update(2, {"val_loss": 1.40}) is True
    assert selector.selected_epoch == 2
    assert selector.best_value == 1.40
