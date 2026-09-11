"""Tests enforcing that scientific configuration has NO silent defaults and fails closed."""

import pytest
from pure_gnn_v31.scientific.config import (
    ScientificConfig,
    load_scientific_config,
    ConfigurationError,
)
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


def test_strict_no_default_config_rejections():
    """Verifies that load_scientific_config raises ConfigurationError when required fields are missing."""
    import yaml
    import tempfile

    # 1. Missing scientific_execution_authorized
    bad_yaml1 = """
data_protocol:
  train_rows: 28709
  val_rows: 3589
  test_access_authorized: false
conditions:
  active: ["G0"]
  primary_comparison: "G1 - G0.5"
hyperparameters:
  seed: {value: 42}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(bad_yaml1)
        tf_path = tf_file.name

    with pytest.raises(ConfigurationError) as exc:
        load_scientific_config(tf_path)
    assert "Missing required field: 'scientific_execution_authorized'" in str(exc.value)

    # 2. Missing train_rows
    bad_yaml2 = """
scientific_execution_authorized: false
data_protocol:
  val_rows: 3589
  test_access_authorized: false
conditions:
  active: ["G0"]
  primary_comparison: "G1 - G0.5"
hyperparameters:
  seed: {value: 42}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(bad_yaml2)
        tf_path2 = tf_file.name

    with pytest.raises(ConfigurationError) as exc:
        load_scientific_config(tf_path2)
    assert "Missing required field in data_protocol: 'train_rows'" in str(exc.value)


def test_trainer_fails_closed_before_execution():
    trainer = ScientificTrainer()
    with pytest.raises(PermissionError) as exc:
        trainer.train_condition("G1")
    assert "SCIENTIFIC EXECUTION BLOCKED" in str(exc.value)
