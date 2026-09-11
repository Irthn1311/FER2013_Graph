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


def test_strict_hyperparameter_schema_enforcement():
    """P0-3: Tests that missing hyperparameter keys, scalar specs, or malformed specs raise ConfigurationError."""
    import tempfile
    from pure_gnn_v31.scientific.config import load_scientific_config

    base_yaml = """
scientific_execution_authorized: false
data_protocol:
  train_rows: 28709
  val_rows: 3589
  test_access_authorized: false
conditions:
  active: ["G0", "G0.5", "G1"]
  primary_comparison: "G1 - G0.5"
hyperparameters:
  optimizer_type: {value: "AdamW", status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  learning_rate: {value: 0.0003, status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  seed: {value: 42, status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  checkpoint_monitor: {value: "val_accuracy", status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  checkpoint_mode: {value: "max", status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  checkpoint_tie_break: {value: "earliest", status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  early_stopping_monitor: {value: "val_loss", status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  early_stopping_patience: {value: 15, status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  validation_frequency_epochs: {value: 1, status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  batch_size: {value: null, status: "REQUIRES_REVIEW", provenance: ["ref"]}
  lr_scheduler: {value: null, status: "REQUIRES_REVIEW", provenance: ["ref"]}
  weight_decay: {value: null, status: "REQUIRES_REVIEW", provenance: ["ref"]}
  max_epochs: {value: null, status: "REQUIRES_REVIEW", provenance: ["ref"]}
  label_smoothing: {value: null, status: "REQUIRES_REVIEW", provenance: ["ref"]}
  global_clipnorm: {value: null, status: "REQUIRES_REVIEW", provenance: ["ref"]}
  augmentation_policy: {value: null, status: "REQUIRES_REVIEW", provenance: ["ref"]}
"""
    # 1. Valid full schema loads successfully
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(base_yaml)
        valid_path = tf_file.name
    cfg = load_scientific_config(valid_path)
    assert cfg.train_rows == 28709

    # 2. Missing a required hyperparameter key (e.g. augmentation_policy)
    bad_yaml_missing_key = base_yaml.replace("augmentation_policy:", "# augmentation_policy:")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(bad_yaml_missing_key)
        bad_path1 = tf_file.name
    with pytest.raises(ConfigurationError) as exc1:
        load_scientific_config(bad_path1)
    assert "Missing required hyperparameter key" in str(exc1.value)

    # 3. Scalar spec instead of mapping (e.g. batch_size: 64 instead of dict)
    bad_yaml_scalar = base_yaml.replace(
        'batch_size: {value: null, status: "REQUIRES_REVIEW", provenance: ["ref"]}',
        "batch_size: 64",
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(bad_yaml_scalar)
        bad_path2 = tf_file.name
    with pytest.raises(ConfigurationError) as exc2:
        load_scientific_config(bad_path2)
    assert "must be a mapping" in str(exc2.value)

    # 4. Malformed spec missing 'status'
    bad_yaml_malformed = base_yaml.replace(
        'status: "REQUIRES_REVIEW"',
        'invalid_key: "test"',
        1,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(bad_yaml_malformed)
        bad_path3 = tf_file.name
    with pytest.raises(ConfigurationError) as exc3:
        load_scientific_config(bad_path3)
    assert "missing required spec field" in str(exc3.value)


def test_trainer_fails_closed_before_execution():
    trainer = ScientificTrainer()
    with pytest.raises(PermissionError) as exc:
        trainer.train_condition("G1")
    assert "SCIENTIFIC EXECUTION BLOCKED" in str(exc.value)
