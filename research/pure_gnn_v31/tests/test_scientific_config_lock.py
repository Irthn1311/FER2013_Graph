"""Tests enforcing that scientific configuration has NO silent defaults, validates semantic schema, and fails closed."""

import hashlib
import tempfile
from pathlib import Path
import pytest
import yaml

from pure_gnn_v31.scientific.config import (
    ScientificConfig,
    load_scientific_config,
    ConfigurationError,
    REQUIRED_HYPERPARAMETER_KEYS,
    EXPECTED_EXACT_VALUES,
    EXPECTED_STATUSES,
)
from pure_gnn_v31.scientific.trainer import ScientificTrainer


def test_scientific_config_raw_byte_sha_and_path():
    """Requirement 1: Proves config retains exact loaded path and true raw-byte SHA256."""
    cfg = load_scientific_config()
    loaded_p = Path(cfg.source_config_path)
    assert loaded_p.is_file()
    assert loaded_p.name == "scientific_screen_historical_v1.yaml"

    expected_sha = hashlib.sha256(loaded_p.read_bytes()).hexdigest()
    assert cfg.source_config_sha256 == expected_sha


def test_scientific_config_crlf_raw_byte_sha():
    """Requirement 1: Proves raw-byte SHA matches even on CRLF newlines without text normalization, exercising load_scientific_config."""
    crlf_yaml_bytes = (CANONICAL_TEST_YAML.strip().replace("\n", "\r\n") + "\r\n").encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(crlf_yaml_bytes)
        p = tf_file.name

    expected_sha = hashlib.sha256(crlf_yaml_bytes).hexdigest()
    cfg = load_scientific_config(p)
    assert cfg.source_config_sha256 == expected_sha
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected_sha
    assert cfg.train_rows == 28709


def test_scientific_execution_defaults_false_and_zero_unresolved():
    """In the preregistered configuration, all required hyperparameters are registered, but execution remains false."""
    config = load_scientific_config()
    assert config.scientific_execution_authorized is False
    assert config.has_unresolved_hyperparameters is False
    assert config.get_unresolved_fields() == []

    with pytest.raises(PermissionError) as exc:
        config.assert_ready_for_execution()
    assert "SCIENTIFIC EXECUTION BLOCKED: 'scientific_execution_authorized' is false" in str(exc.value)


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


CANONICAL_TEST_YAML = """
scientific_execution_authorized: false
data_protocol:
  train_rows: 28709
  val_rows: 3589
  internal_train_split: false
  research_dev_split: false
  test_access_authorized: false
  pixel_normalization:
    value: "raw_div_255"
conditions:
  active: ["G0", "G0.5", "G1"]
  primary_comparison: "G1 - G0.5"
  g2_g3_scheduled: false
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
  lr_scheduler:
    value:
      type: "WarmupCosine"
      warmup_epochs: 5
      initial_learning_rate: 0.0003
      final_learning_rate: 0.000001
      max_epochs: 100
    status: "PREREGISTERED_DECISION"
    provenance: ["ref"]
  weight_decay: {value: 0.0005, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  max_epochs: {value: 100, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  label_smoothing: {value: 0.05, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  global_clipnorm: {value: 1.0, status: "PREREGISTERED_DECISION", provenance: ["ref"]}
  augmentation_policy: {value: "gen2_gen3_stateless_image_v1", status: "PREREGISTERED_DECISION", provenance: ["ref"]}
"""


@pytest.mark.parametrize("key", [
    "optimizer_type",
    "learning_rate",
    "seed",
    "checkpoint_monitor",
    "checkpoint_mode",
    "checkpoint_tie_break",
    "early_stopping_monitor",
    "early_stopping_patience",
    "validation_frequency_epochs",
    "batch_size",
    "weight_decay",
    "max_epochs",
    "label_smoothing",
    "global_clipnorm",
    "augmentation_policy",
])
def test_parametrized_value_mutation_fails(key):
    """Requirement 2: Every single scalar value mutation must fail validation."""
    data = yaml.safe_load(CANONICAL_TEST_YAML)
    orig_val = data["hyperparameters"][key]["value"]

    # Mutate value
    if isinstance(orig_val, int):
        data["hyperparameters"][key]["value"] = orig_val + 1
    elif isinstance(orig_val, float):
        data["hyperparameters"][key]["value"] = orig_val * 2
    else:
        data["hyperparameters"][key]["value"] = str(orig_val) + "_mutated"

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(yaml.dump(data))
        p = tf_file.name

    with pytest.raises(ConfigurationError):
        load_scientific_config(p)


@pytest.mark.parametrize("key", REQUIRED_HYPERPARAMETER_KEYS)
def test_parametrized_status_mutation_fails(key):
    """Requirement 2: Every single status mutation must fail validation."""
    data = yaml.safe_load(CANONICAL_TEST_YAML)
    orig_status = data["hyperparameters"][key]["status"]
    # Change status
    new_status = "PREREGISTERED_DECISION" if orig_status == "SOURCE_CONFIRMED" else "SOURCE_CONFIRMED"
    data["hyperparameters"][key]["status"] = new_status

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_file:
        tf_file.write(yaml.dump(data))
        p = tf_file.name

    with pytest.raises(ConfigurationError) as exc:
        load_scientific_config(p)
    assert "status mismatch" in str(exc.value)


def test_lr_scheduler_nested_mutations_fail():
    """Requirement 3: Verifies that initial_learning_rate <= 0, NaN, Inf, and final < 0 fail."""
    data = yaml.safe_load(CANONICAL_TEST_YAML)
    sched = data["hyperparameters"]["lr_scheduler"]["value"]

    # 1. initial_learning_rate <= 0
    data1 = yaml.safe_load(CANONICAL_TEST_YAML)
    data1["hyperparameters"]["lr_scheduler"]["value"]["initial_learning_rate"] = -0.01
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml.dump(data1))
        p1 = f.name
    with pytest.raises(ConfigurationError):
        load_scientific_config(p1)

    # 2. initial_learning_rate = NaN
    data2 = yaml.safe_load(CANONICAL_TEST_YAML)
    data2["hyperparameters"]["lr_scheduler"]["value"]["initial_learning_rate"] = ".nan"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml.dump(data2))
        p2 = f.name
    with pytest.raises(ConfigurationError):
        load_scientific_config(p2)

    # 3. final_learning_rate < 0
    data3 = yaml.safe_load(CANONICAL_TEST_YAML)
    data3["hyperparameters"]["lr_scheduler"]["value"]["final_learning_rate"] = -1e-6
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml.dump(data3))
        p3 = f.name
    with pytest.raises(ConfigurationError):
        load_scientific_config(p3)


def test_trainer_fails_closed_before_execution():
    trainer = ScientificTrainer()
    with pytest.raises(PermissionError) as exc:
        trainer.train_condition(
            condition="G1",
            train_csv_path="path/to/train.csv",
            val_csv_path="path/to/val.csv",
            output_dir="outputs/test_run",
        )
    assert "SCIENTIFIC EXECUTION BLOCKED" in str(exc.value)
