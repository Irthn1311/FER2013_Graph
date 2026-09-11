"""Configuration loader and strict schema validation for Pure-GNN scientific screen."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class ConfigurationError(ValueError):
    """Raised when scientific configuration schema or requirements are violated."""


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

    @property
    def has_unresolved_hyperparameters(self) -> bool:
        """Returns True if any hyperparameter is marked REQUIRES_REVIEW or is None."""
        for name, spec in self.hyperparameters.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("status") == "REQUIRES_REVIEW" or spec.get("value") is None:
                return True
        return False

    def get_unresolved_fields(self) -> List[str]:
        unresolved = []
        for name, spec in self.hyperparameters.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("status") == "REQUIRES_REVIEW" or spec.get("value") is None:
                unresolved.append(name)
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
    if not isinstance(train_rows, int):
        raise ConfigurationError("'train_rows' must be an integer.")

    if "val_rows" not in data_proto:
        raise ConfigurationError("Missing required field in data_protocol: 'val_rows'.")
    val_rows = data_proto["val_rows"]
    if not isinstance(val_rows, int):
        raise ConfigurationError("'val_rows' must be an integer.")

    if "test_access_authorized" not in data_proto:
        raise ConfigurationError("Missing required field in data_protocol: 'test_access_authorized'.")
    test_auth = data_proto["test_access_authorized"]
    if not isinstance(test_auth, bool):
        raise ConfigurationError("'test_access_authorized' must be a boolean.")

    # 3. Enforcement: 'conditions' MUST be explicitly provided
    if "conditions" not in data or not isinstance(data["conditions"], dict):
        raise ConfigurationError("Missing required dictionary: 'conditions'.")
    cond_dict = data["conditions"]

    if "active" not in cond_dict or not isinstance(cond_dict["active"], list):
        raise ConfigurationError("Missing required list in conditions: 'active'.")
    active_conditions = cond_dict["active"]

    if "primary_comparison" not in cond_dict or not isinstance(cond_dict["primary_comparison"], str):
        raise ConfigurationError("Missing required string in conditions: 'primary_comparison'.")
    primary_comparison = cond_dict["primary_comparison"]

    # 4. Enforcement: 'hyperparameters' MUST be explicitly provided
    if "hyperparameters" not in data or not isinstance(data["hyperparameters"], dict):
        raise ConfigurationError("Missing required dictionary: 'hyperparameters'.")
    hyperparams = data["hyperparameters"]

    # Seed may be None if unresolved, but must be represented
    seed_val: Optional[int] = None
    if "seed" in hyperparams and isinstance(hyperparams["seed"], dict):
        val = hyperparams["seed"].get("value")
        if val is not None:
            if not isinstance(val, int):
                raise ConfigurationError("'seed' value must be an integer or null.")
            seed_val = val

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
    )
