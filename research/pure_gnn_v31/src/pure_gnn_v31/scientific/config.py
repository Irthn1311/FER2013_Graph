"""Configuration loader and strict semantic schema validation for Pure-GNN scientific screen."""

from dataclasses import dataclass
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


def _validate_semantic_hyperparameter_value(key: str, val: Any) -> None:
    """Strictly validates exact value types and domains for known scientific hyperparameters."""
    if key == "optimizer_type":
        if val != "AdamW":
            raise ConfigurationError(f"optimizer_type must be 'AdamW', got: {val}")

    elif key == "learning_rate":
        if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0:
            raise ConfigurationError(f"learning_rate must be a finite positive float, got: {val}")

    elif key == "seed":
        if val is not None and (not isinstance(val, int) or val < 0):
            raise ConfigurationError(f"seed must be a non-negative integer or null, got: {val}")

    elif key == "checkpoint_monitor":
        if val != "val_accuracy":
            raise ConfigurationError(f"checkpoint_monitor must be 'val_accuracy', got: {val}")

    elif key == "checkpoint_mode":
        if val != "max":
            raise ConfigurationError(f"checkpoint_mode must be 'max', got: {val}")

    elif key == "checkpoint_tie_break":
        if val != "earliest_strict_max_val_accuracy":
            raise ConfigurationError(f"checkpoint_tie_break must be 'earliest_strict_max_val_accuracy', got: {val}")

    elif key == "early_stopping_monitor":
        if val != "val_loss":
            raise ConfigurationError(f"early_stopping_monitor must be 'val_loss', got: {val}")

    elif key == "early_stopping_patience":
        if not isinstance(val, int) or val <= 0:
            raise ConfigurationError(f"early_stopping_patience must be a positive integer, got: {val}")

    elif key == "validation_frequency_epochs":
        if not isinstance(val, int) or val <= 0:
            raise ConfigurationError(f"validation_frequency_epochs must be a positive integer, got: {val}")

    elif key == "batch_size":
        if not isinstance(val, int) or val <= 0:
            raise ConfigurationError(f"batch_size must be a positive integer, got: {val}")

    elif key == "weight_decay":
        if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
            raise ConfigurationError(f"weight_decay must be a finite non-negative float, got: {val}")

    elif key == "max_epochs":
        if not isinstance(val, int) or val <= 0:
            raise ConfigurationError(f"max_epochs must be a positive integer, got: {val}")

    elif key == "label_smoothing":
        if not isinstance(val, (int, float)) or not math.isfinite(val) or not (0.0 <= val < 1.0):
            raise ConfigurationError(f"label_smoothing must be a float in [0.0, 1.0), got: {val}")

    elif key == "global_clipnorm":
        if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0:
            raise ConfigurationError(f"global_clipnorm must be a finite positive float, got: {val}")

    elif key == "augmentation_policy":
        if val != "gen2_gen3_stateless_image_v1":
            raise ConfigurationError(f"Unsupported augmentation_policy: '{val}'. Expected 'gen2_gen3_stateless_image_v1'.")

    elif key == "lr_scheduler":
        if not isinstance(val, dict):
            raise ConfigurationError(f"lr_scheduler must be a structured mapping, got: {type(val)}")
        if val.get("type") != "WarmupCosine":
            raise ConfigurationError(f"lr_scheduler type must be 'WarmupCosine', got: {val.get('type')}")
        warmup = val.get("warmup_epochs")
        init_lr = val.get("initial_learning_rate")
        fin_lr = val.get("final_learning_rate")
        max_ep = val.get("max_epochs")
        if not isinstance(warmup, int) or warmup <= 0:
            raise ConfigurationError(f"lr_scheduler.warmup_epochs must be positive int, got: {warmup}")
            try:
                init_lr_f = float(init_lr)
            except (ValueError, TypeError):
                raise ConfigurationError(f"lr_scheduler.initial_learning_rate must be float, got: {init_lr}")
            if not math.isfinite(init_lr_f) or init_lr_f <= 0:
                raise ConfigurationError(f"lr_scheduler.initial_learning_rate must be positive float, got: {init_lr}")

            try:
                fin_lr_f = float(fin_lr)
            except (ValueError, TypeError):
                raise ConfigurationError(f"lr_scheduler.final_learning_rate must be float, got: {fin_lr}")
            if not math.isfinite(fin_lr_f) or fin_lr_f < 0:
                raise ConfigurationError(f"lr_scheduler.final_learning_rate must be non-negative float, got: {fin_lr}")
        if not isinstance(max_ep, int) or max_ep <= 0:
            raise ConfigurationError(f"lr_scheduler.max_epochs must be positive int, got: {max_ep}")


def load_scientific_config(config_path: Optional[str] = None) -> ScientificConfig:
    """Loads and strictly parses scientific_screen_historical_v1.yaml with NO silent defaults."""
    if config_path is None:
        config_path = str(
            Path(__file__).resolve().parents[3]
            / "configs"
            / "scientific_screen_historical_v1.yaml"
        )

    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Scientific config not found: {p}")

    with p.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise ConfigurationError("Root YAML must be a dictionary.")

    # 1. Enforcement: 'scientific_execution_authorized' MUST be explicitly provided
    if "scientific_execution_authorized" not in data:
        raise ConfigurationError("Missing required field: 'scientific_execution_authorized'.")
    exec_auth = data["scientific_execution_authorized"]
    if not isinstance(exec_auth, bool):
        raise ConfigurationError("'scientific_execution_authorized' must be a boolean.")

    # 2. Enforcement: 'data_protocol' MUST be explicitly provided
    if "data_protocol" not in data or not isinstance(data["data_protocol"], dict):
        raise ConfigurationError("Missing required dictionary: 'data_protocol'.")
    data_proto = data["data_protocol"]

    if "train_rows" not in data_proto:
        raise ConfigurationError("Missing required field in data_protocol: 'train_rows'.")
    train_rows = data_proto["train_rows"]
    if not isinstance(train_rows, int) or train_rows <= 0:
        raise ConfigurationError("'train_rows' must be a positive integer.")

    if "val_rows" not in data_proto:
        raise ConfigurationError("Missing required field in data_protocol: 'val_rows'.")
    val_rows = data_proto["val_rows"]
    if not isinstance(val_rows, int) or val_rows <= 0:
        raise ConfigurationError("'val_rows' must be a positive integer.")

    if "test_access_authorized" not in data_proto:
        raise ConfigurationError("Missing required field in data_protocol: 'test_access_authorized'.")
    test_auth = data_proto["test_access_authorized"]
    if not isinstance(test_auth, bool):
        raise ConfigurationError("'test_access_authorized' must be a boolean.")

    # 3. Enforcement: 'conditions' MUST be explicitly provided
    if "conditions" not in data or not isinstance(data["conditions"], dict):
        raise ConfigurationError("Missing required dictionary: 'conditions'.")
    cond_dict = data["conditions"]

    if "active" not in cond_dict or not isinstance(cond_dict["active"], list) or not cond_dict["active"]:
        raise ConfigurationError("Missing or empty required list in conditions: 'active'.")
    active_conditions = cond_dict["active"]

    if "primary_comparison" not in cond_dict or not isinstance(cond_dict["primary_comparison"], str):
        raise ConfigurationError("Missing required string in conditions: 'primary_comparison'.")
    primary_comparison = cond_dict["primary_comparison"]

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

        # Semantic validation of value if not None
        val = spec["value"]
        if val is not None:
            _validate_semantic_hyperparameter_value(req_key, val)

    seed_spec = hyperparams["seed"]
    seed_val = seed_spec["value"]

    # Extract WarmupCosineConfig
    warmup_cfg = None
    sched_val = hyperparams["lr_scheduler"]["value"]
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
        warmup_cosine_config=warmup_cfg,
    )
