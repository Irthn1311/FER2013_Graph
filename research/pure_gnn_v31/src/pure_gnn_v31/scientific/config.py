"""Configuration loader and strict semantic schema validation for Pure-GNN scientific screen."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class ConfigurationError(ValueError):
    """Raised when scientific configuration schema or requirements are violated."""


ALLOWED_STATUSES = {
    "SOURCE_CONFIRMED",
    "PREREGISTERED_DECISION",
    "REQUIRES_REVIEW",
    "PURE_GNN_SPECIFIC_INPUT_CONTRACT",
}

REQUIRED_HYPERPARAMETER_KEYS = [
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
    "lr_scheduler",
    "weight_decay",
    "max_epochs",
    "label_smoothing",
    "global_clipnorm",
    "augmentation_policy",
]

EXPECTED_STATUSES = {
    "optimizer_type": "SOURCE_CONFIRMED",
    "learning_rate": "SOURCE_CONFIRMED",
    "seed": "SOURCE_CONFIRMED",
    "checkpoint_monitor": "SOURCE_CONFIRMED",
    "checkpoint_mode": "SOURCE_CONFIRMED",
    "checkpoint_tie_break": "SOURCE_CONFIRMED",
    "early_stopping_monitor": "SOURCE_CONFIRMED",
    "early_stopping_patience": "SOURCE_CONFIRMED",
    "validation_frequency_epochs": "SOURCE_CONFIRMED",
    "batch_size": "PREREGISTERED_DECISION",
    "lr_scheduler": "PREREGISTERED_DECISION",
    "weight_decay": "PREREGISTERED_DECISION",
    "max_epochs": "PREREGISTERED_DECISION",
    "label_smoothing": "PREREGISTERED_DECISION",
    "global_clipnorm": "PREREGISTERED_DECISION",
    "augmentation_policy": "PREREGISTERED_DECISION",
}

EXPECTED_EXACT_VALUES = {
    "optimizer_type": "AdamW",
    "learning_rate": 0.0003,
    "seed": 42,
    "checkpoint_monitor": "val_accuracy",
    "checkpoint_mode": "max",
    "checkpoint_tie_break": "earliest_strict_max_val_accuracy",
    "early_stopping_monitor": "val_loss",
    "early_stopping_patience": 15,
    "validation_frequency_epochs": 1,
    "batch_size": 64,
    "weight_decay": 0.0005,
    "max_epochs": 100,
    "label_smoothing": 0.05,
    "global_clipnorm": 1.0,
    "augmentation_policy": "gen2_gen3_stateless_image_v1",
}


@dataclass
class WarmupCosineConfig:
    type: str
    warmup_epochs: int
    initial_learning_rate: float
    final_learning_rate: float
    max_epochs: int


@dataclass
class ScientificConfig:
    raw_config: Dict[str, Any]
    scientific_execution_authorized: bool
    train_rows: int
    val_rows: int
    test_access_authorized: bool
    seed: Optional[int]
    active_conditions: List[str]
    primary_comparison: str
    hyperparameters: Dict[str, Any]
    source_config_path: str
    source_config_sha256: str
    warmup_cosine_config: Optional[WarmupCosineConfig] = None

    @property
    def has_unresolved_hyperparameters(self) -> bool:
        """Returns True if any hyperparameter is marked REQUIRES_REVIEW or has null value."""
        for key in REQUIRED_HYPERPARAMETER_KEYS:
            spec = self.hyperparameters.get(key)
            if not isinstance(spec, dict):
                return True
            if spec.get("status") == "REQUIRES_REVIEW" or spec.get("value") is None:
                return True
        return False

    def get_unresolved_fields(self) -> List[str]:
        unresolved = []
        for key in REQUIRED_HYPERPARAMETER_KEYS:
            spec = self.hyperparameters.get(key)
            if not isinstance(spec, dict) or spec.get("status") == "REQUIRES_REVIEW" or spec.get("value") is None:
                unresolved.append(key)
        return unresolved

    def assert_ready_for_execution(self) -> None:
        """Fails closed if execution is not authorized or hyperparameters are unresolved."""
        if not self.scientific_execution_authorized:
            raise PermissionError(
                "SCIENTIFIC EXECUTION BLOCKED: 'scientific_execution_authorized' is false. "
                "Independent source and configuration review is required."
            )
        if self.seed is None:
            raise ConfigurationError(
                "SCIENTIFIC EXECUTION BLOCKED: 'seed' is null or unresolved."
            )
        unresolved = self.get_unresolved_fields()
        if unresolved:
            raise ConfigurationError(
                f"SCIENTIFIC EXECUTION BLOCKED: Unresolved hyperparameters requiring review: {unresolved}"
            )
        if self.test_access_authorized:
            raise PermissionError("DATA GOVERNANCE VIOLATION: Test access cannot be authorized.")


def _validate_frozen_hyperparameter(key: str, spec: Dict[str, Any]) -> None:
    """Strictly validates exact value and status for the frozen scientific recipe."""
    val = spec["value"]
    status = spec["status"]

    expected_status = EXPECTED_STATUSES.get(key)
    if expected_status and status != expected_status:
        raise ConfigurationError(
            f"Hyperparameter '{key}' status mismatch: expected '{expected_status}', got '{status}'"
        )

    if key == "lr_scheduler":
        if not isinstance(val, dict):
            raise ConfigurationError(f"lr_scheduler must be a structured mapping, got: {type(val)}")
        if val.get("type") != "WarmupCosine":
            raise ConfigurationError(f"lr_scheduler type must be 'WarmupCosine', got: {val.get('type')}")

        warmup = val.get("warmup_epochs")
        init_lr = val.get("initial_learning_rate")
        fin_lr = val.get("final_learning_rate")
        max_ep = val.get("max_epochs")

        if not isinstance(warmup, int) or warmup != 5:
            raise ConfigurationError(f"lr_scheduler.warmup_epochs must be 5, got: {warmup}")

        try:
            init_lr_f = float(init_lr)
        except (ValueError, TypeError):
            raise ConfigurationError(f"lr_scheduler.initial_learning_rate must be float, got: {init_lr}")
        if not math.isfinite(init_lr_f) or abs(init_lr_f - 0.0003) > 1e-12:
            raise ConfigurationError(f"lr_scheduler.initial_learning_rate must be 0.0003, got: {init_lr}")

        try:
            fin_lr_f = float(fin_lr)
        except (ValueError, TypeError):
            raise ConfigurationError(f"lr_scheduler.final_learning_rate must be float, got: {fin_lr}")
        if not math.isfinite(fin_lr_f) or abs(fin_lr_f - 0.000001) > 1e-12:
            raise ConfigurationError(f"lr_scheduler.final_learning_rate must be 0.000001, got: {fin_lr}")

        if not isinstance(max_ep, int) or max_ep != 100:
            raise ConfigurationError(f"lr_scheduler.max_epochs must be 100, got: {max_ep}")
        return

    # Check scalar exact values
    if key in EXPECTED_EXACT_VALUES:
        exp_val = EXPECTED_EXACT_VALUES[key]
        if isinstance(exp_val, float):
            try:
                val_f = float(val)
            except (ValueError, TypeError):
                raise ConfigurationError(f"Hyperparameter '{key}' must be numeric, got: {val}")
            if not math.isfinite(val_f) or abs(val_f - exp_val) > 1e-12:
                raise ConfigurationError(
                    f"Hyperparameter '{key}' exact value mismatch: expected {exp_val}, got {val}"
                )
        else:
            if val != exp_val:
                raise ConfigurationError(
                    f"Hyperparameter '{key}' exact value mismatch: expected '{exp_val}', got '{val}'"
                )


def load_scientific_config(config_path: Optional[str] = None) -> ScientificConfig:
    """Loads and strictly parses scientific_screen_historical_v1.yaml with NO silent defaults.

    Computes exact SHA256 of the raw file bytes loaded.
    """
    if config_path is None:
        config_path = str(
            Path(__file__).resolve().parents[3]
            / "configs"
            / "scientific_screen_historical_v1.yaml"
        )

    resolved_path = Path(config_path).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Scientific config not found: {resolved_path}")

    raw_bytes = resolved_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    data = yaml.safe_load(raw_bytes)
    if not isinstance(data, dict):
        raise ConfigurationError("Root YAML must be a dictionary.")

    # 1. Enforcement: 'scientific_execution_authorized' MUST be explicitly provided
    if "scientific_execution_authorized" not in data:
        raise ConfigurationError("Missing required field: 'scientific_execution_authorized'.")
    exec_auth = data["scientific_execution_authorized"]
    if not isinstance(exec_auth, bool):
        raise ConfigurationError("'scientific_execution_authorized' must be a boolean.")

    # 2. Enforcement: 'data_protocol' MUST be explicitly provided and freeze protocol invariants
    if "data_protocol" not in data or not isinstance(data["data_protocol"], dict):
        raise ConfigurationError("Missing required dictionary: 'data_protocol'.")
    data_proto = data["data_protocol"]

    if "train_rows" not in data_proto:
        raise ConfigurationError("Missing required field in data_protocol: 'train_rows'.")
    train_rows = data_proto["train_rows"]
    if not isinstance(train_rows, int) or train_rows != 28709:
        raise ConfigurationError(f"Protocol invariant violation: train_rows must be 28709, got: {train_rows}")

    if "val_rows" not in data_proto:
        raise ConfigurationError("Missing required field in data_protocol: 'val_rows'.")
    val_rows = data_proto["val_rows"]
    if not isinstance(val_rows, int) or val_rows != 3589:
        raise ConfigurationError(f"Protocol invariant violation: val_rows must be 3589, got: {val_rows}")

    if data_proto.get("internal_train_split") is not False:
        raise ConfigurationError("Protocol invariant violation: internal_train_split must be False")

    if data_proto.get("research_dev_split") is not False:
        raise ConfigurationError("Protocol invariant violation: research_dev_split must be False")

    if "test_access_authorized" not in data_proto:
        raise ConfigurationError("Missing required field in data_protocol: 'test_access_authorized'.")
    test_auth = data_proto["test_access_authorized"]
    if not isinstance(test_auth, bool) or test_auth is not False:
        raise ConfigurationError("Protocol invariant violation: test_access_authorized must be False")

    pix_norm = data_proto.get("pixel_normalization")
    if not isinstance(pix_norm, dict) or pix_norm.get("value") != "raw_div_255":
        raise ConfigurationError(
            f"Protocol invariant violation: pixel_normalization.value must be 'raw_div_255', got: {pix_norm}"
        )

    # 3. Enforcement: 'conditions' MUST be explicitly provided and freeze protocol invariants
    if "conditions" not in data or not isinstance(data["conditions"], dict):
        raise ConfigurationError("Missing required dictionary: 'conditions'.")
    cond_dict = data["conditions"]

    active_conditions = cond_dict.get("active")
    if active_conditions != ["G0", "G0.5", "G1"]:
        raise ConfigurationError(
            f"Protocol invariant violation: conditions.active must be ['G0', 'G0.5', 'G1'], got: {active_conditions}"
        )

    primary_comparison = cond_dict.get("primary_comparison")
    if primary_comparison != "G1 - G0.5":
        raise ConfigurationError(
            f"Protocol invariant violation: primary_comparison must be 'G1 - G0.5', got: '{primary_comparison}'"
        )

    if cond_dict.get("g2_g3_scheduled") is not False:
        raise ConfigurationError("Protocol invariant violation: conditions.g2_g3_scheduled must be False")

    # 4. Enforcement: 'hyperparameters' MUST be explicitly provided
    if "hyperparameters" not in data or not isinstance(data["hyperparameters"], dict):
        raise ConfigurationError("Missing required dictionary: 'hyperparameters'.")
    hyperparams = data["hyperparameters"]

    # Validate every required hyperparameter key
    for req_key in REQUIRED_HYPERPARAMETER_KEYS:
        if req_key not in hyperparams:
            raise ConfigurationError(f"Missing required hyperparameter key: '{req_key}'.")

        spec = hyperparams[req_key]
        if not isinstance(spec, dict):
            raise ConfigurationError(
                f"Hyperparameter '{req_key}' must be a mapping with value, status, and provenance. "
                f"Got scalar or non-dict: {spec}"
            )

        for req_field in ("value", "status", "provenance"):
            if req_field not in spec:
                raise ConfigurationError(
                    f"Hyperparameter '{req_key}' missing required spec field '{req_field}'."
                )

        status_val = spec["status"]
        if status_val not in ALLOWED_STATUSES:
            raise ConfigurationError(
                f"Hyperparameter '{req_key}' has invalid status '{status_val}'. "
                f"Allowed: {sorted(ALLOWED_STATUSES)}"
            )

        if not isinstance(spec["provenance"], list) or not spec["provenance"]:
            raise ConfigurationError(
                f"Hyperparameter '{req_key}' provenance must be a non-empty list. Got: {spec.get('provenance')}"
            )

        # Freeze the exact registered recipe value and status
        _validate_frozen_hyperparameter(req_key, spec)

    seed_spec = hyperparams["seed"]
    seed_val = seed_spec["value"]

    # Cross-field consistency
    init_lr_val = hyperparams["learning_rate"]["value"]
    sched_val = hyperparams["lr_scheduler"]["value"]
    if isinstance(sched_val, dict):
        sched_init_lr = float(sched_val.get("initial_learning_rate", 0.0))
        if abs(float(init_lr_val) - sched_init_lr) > 1e-12:
            raise ConfigurationError(
                f"Cross-field consistency violation: hyperparameters.learning_rate ({init_lr_val}) "
                f"!= lr_scheduler.initial_learning_rate ({sched_init_lr})"
            )

        max_ep_val = hyperparams["max_epochs"]["value"]
        sched_max_ep = int(sched_val.get("max_epochs", 0))
        if int(max_ep_val) != sched_max_ep:
            raise ConfigurationError(
                f"Cross-field consistency violation: hyperparameters.max_epochs ({max_ep_val}) "
                f"!= lr_scheduler.max_epochs ({sched_max_ep})"
            )

    # Extract WarmupCosineConfig
    warmup_cfg = None
    if isinstance(sched_val, dict) and sched_val.get("type") == "WarmupCosine":
        warmup_cfg = WarmupCosineConfig(
            type="WarmupCosine",
            warmup_epochs=int(sched_val["warmup_epochs"]),
            initial_learning_rate=float(sched_val["initial_learning_rate"]),
            final_learning_rate=float(sched_val["final_learning_rate"]),
            max_epochs=int(sched_val["max_epochs"]),
        )

    return ScientificConfig(
        raw_config=data,
        scientific_execution_authorized=exec_auth,
        train_rows=train_rows,
        val_rows=val_rows,
        test_access_authorized=test_auth,
        seed=seed_val,
        active_conditions=active_conditions,
        primary_comparison=primary_comparison,
        hyperparameters=hyperparams,
        source_config_path=str(resolved_path),
        source_config_sha256=raw_sha256,
        warmup_cosine_config=warmup_cfg,
    )
