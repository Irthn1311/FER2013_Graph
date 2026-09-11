"""Tests enforcing that scientific configuration has NO silent defaults, validates semantic schema, and fails closed."""

import tempfile
import pytest
from pure_gnn_v31.scientific.config import (
    ScientificConfig,
    load_scientific_config,
    ConfigurationError,
)
from pure_gnn_v31.scientific.trainer import ScientificTrainer


def test_scientific_execution_defaults_false_and_zero_unresolved():
    """In the preregistered configuration, all required hyperparameters are registered, but execution remains false."""
    config = load_scientific_config()
    assert config.scientific_execution_authorized is False
    assert config.has_unresolved_hyperparameters is False
    assert config.get_unresolved_fields() == []

    # Execution remains blocked strictly by the authorization gate
    with pytest.raises(PermissionError) as exc:
        config.assert_ready_for_execution()
    assert "SCIENTIFIC EXECUTION BLOCKED: 'scientific_execution_authorized' is false" in str(exc.value)


def test_unresolved_config_prevents_execution_when_field_is_requires_review():
    """Tests that any config with REQUIRES_REVIEW or null value raises ConfigurationError."""
    base_yaml = """
scientific_execution_authorized: true
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
  checkpoint_tie_break: {value: "earliest_strict_max_val_accuracy", status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  early_stopping_monitor: {value: "val_loss", status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  early_stopping_patience: {value: 15, status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  validation_frequency_epochs: {value: 1, status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  batch_size: {value: null, status: "REQUIRES_REVIEW", provenance: ["ref"]}
  lr_scheduler: {value: {type: "WarmupCosine", warmup_epochs: 5, initial_learning_rate: 3e-4, final_learning_rate: 1e-6, max_epochs: 100}, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  weight_decay: {value: 0.0005, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  max_epochs: {value: 100, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  label_smoothing: {value: 0.05, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  global_clipnorm: {value: 1.0, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  augmentation_policy: {value: "gen2_gen3_stateless_image_v1", status: "PREREGISTERED_DECISION", provenance: ["ref"]}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(base_yaml)
        p = tf_file.name

    cfg = load_scientific_config(p)
    assert cfg.has_unresolved_hyperparameters is True
    assert "batch_size" in cfg.get_unresolved_fields()

    with pytest.raises(ConfigurationError) as exc:
        cfg.assert_ready_for_execution()
    assert "Unresolved hyperparameters requiring review: ['batch_size']" in str(exc.value)


def test_strict_no_default_config_rejections():
    """Verifies that load_scientific_config raises ConfigurationError when required fields are missing."""
    bad_yaml1 = """
data_protocol:
  train_rows: 28709
  val_rows: 3589
  test_access_authorized: false
conditions:
  active: ["G0"]
  primary_comparison: "G1 - G0.5"
hyperparameters:
  seed: {value: 42, status: "SOURCE_CONFIRMED", provenance: ["ref"]}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(bad_yaml1)
        tf_path1 = tf_file.name

    with pytest.raises(ConfigurationError) as exc:
        load_scientific_config(tf_path1)
    assert "Missing required field: 'scientific_execution_authorized'" in str(exc.value)


def test_strict_semantic_hyperparameter_value_rejections():
    """Verifies that invalid types or out-of-domain values raise ConfigurationError."""
    base_valid = """
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
  checkpoint_tie_break: {value: "earliest_strict_max_val_accuracy", status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  early_stopping_monitor: {value: "val_loss", status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  early_stopping_patience: {value: 15, status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  validation_frequency_epochs: {value: 1, status: "SOURCE_CONFIRMED", provenance: ["ref"]}
  batch_size: {value: 64, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  lr_scheduler: {value: {type: "WarmupCosine", warmup_epochs: 5, initial_learning_rate: 3e-4, final_learning_rate: 1e-6, max_epochs: 100}, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  weight_decay: {value: 0.0005, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  max_epochs: {value: 100, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  label_smoothing: {value: 0.05, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  global_clipnorm: {value: 1.0, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  augmentation_policy: {value: "gen2_gen3_stateless_image_v1", status: "PREREGISTERED_DECISION", provenance: ["ref"]}
"""
    # 1. Invalid optimizer_type
    bad_opt = base_valid.replace('optimizer_type: {value: "AdamW"', 'optimizer_type: {value: "SGD"')
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(bad_opt)
        p = tf_file.name
    with pytest.raises(ConfigurationError) as exc:
        load_scientific_config(p)
    assert "optimizer_type must be 'AdamW'" in str(exc.value)

    # 2. Invalid learning_rate <= 0
    bad_lr = base_valid.replace('learning_rate: {value: 0.0003', 'learning_rate: {value: -0.01')
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(bad_lr)
        p = tf_file.name
    with pytest.raises(ConfigurationError) as exc:
        load_scientific_config(p)
    assert "learning_rate must be a finite positive float" in str(exc.value)

    # 3. Invalid status
    bad_status = base_valid.replace('status: "PREREGISTERED_DECISION"', 'status: "MY_INVENTED_STATUS"')
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(bad_status)
        p = tf_file.name
    with pytest.raises(ConfigurationError) as exc:
        load_scientific_config(p)
    assert "invalid status 'MY_INVENTED_STATUS'" in str(exc.value)


def test_trainer_fails_closed_before_execution():
    trainer = ScientificTrainer()
    with pytest.raises(PermissionError) as exc:
        trainer.train_condition("G1")
    assert "SCIENTIFIC EXECUTION BLOCKED" in str(exc.value)
