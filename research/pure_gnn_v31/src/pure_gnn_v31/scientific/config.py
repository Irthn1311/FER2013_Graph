"""Configuration loader and schema validation for Pure-GNN scientific screen."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ScientificConfig:
    raw_config: Dict[str, Any]
    scientific_execution_authorized: bool
    train_rows: int
    val_rows: int
    test_access_authorized: bool
    seed: int
    active_conditions: List[str]
    primary_comparison: str
    hyperparameters: Dict[str, Any]

    @property
    def has_unresolved_hyperparameters(self) -> bool:
        """Returns True if any hyperparameter is marked REQUIRES_REVIEW or is None."""
        for name, spec in self.hyperparameters.items():
            if spec.get("status") == "REQUIRES_REVIEW" or spec.get("value") is None:
                return True
        return False

    def get_unresolved_fields(self) -> List[str]:
        unresolved = []
        for name, spec in self.hyperparameters.items():
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
        unresolved = self.get_unresolved_fields()
        if unresolved:
            raise ValueError(
                f"SCIENTIFIC EXECUTION BLOCKED: Unresolved hyperparameters requiring review: {unresolved}"
            )
        if self.test_access_authorized:
            raise PermissionError("DATA GOVERNANCE VIOLATION: Test access cannot be authorized.")


def load_scientific_config(config_path: Optional[str] = None) -> ScientificConfig:
    """Loads and parses scientific_screen_historical_v1.yaml."""
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

    data_proto = data.get("data_protocol", {})
    conditions = data.get("conditions", {})
    hyperparams = data.get("hyperparameters", {})

    return ScientificConfig(
        raw_config=data,
        scientific_execution_authorized=bool(data.get("scientific_execution_authorized", False)),
        train_rows=int(data_proto.get("train_rows", 28709)),
        val_rows=int(data_proto.get("val_rows", 3589)),
        test_access_authorized=bool(data_proto.get("test_access_authorized", False)),
        seed=int(hyperparams.get("seed", {}).get("value", 42)),
        active_conditions=list(conditions.get("active", ["G0", "G0.5", "G1"])),
        primary_comparison=str(conditions.get("primary_comparison", "G1 - G0.5")),
        hyperparameters=hyperparams,
    )
