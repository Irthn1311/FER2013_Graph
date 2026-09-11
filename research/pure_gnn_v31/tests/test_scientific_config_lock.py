"""Tests enforcing that scientific execution is locked and fails closed on unresolved fields."""

import pytest
from pure_gnn_v31.scientific.config import load_scientific_config
from pure_gnn_v31.scientific.trainer import ScientificTrainer


def test_scientific_execution_defaults_false():
    config = load_scientific_config()
    assert config.scientific_execution_authorized is False


def test_unresolved_config_prevents_execution():
    config = load_scientific_config()
    assert config.has_unresolved_hyperparameters is True
    unresolved = config.get_unresolved_fields()
    # At minimum batch_size, lr_scheduler, weight_decay must be in unresolved
    assert "batch_size" in unresolved
    assert "lr_scheduler" in unresolved

    # Assert assert_ready_for_execution raises PermissionError because authorized=False
    with pytest.raises(PermissionError) as exc:
        config.assert_ready_for_execution()
    assert "SCIENTIFIC EXECUTION BLOCKED" in str(exc.value)


def test_trainer_fails_closed_before_execution():
    trainer = ScientificTrainer()
    with pytest.raises(PermissionError) as exc:
        trainer.train_condition("G1")
    assert "SCIENTIFIC EXECUTION BLOCKED" in str(exc.value)
