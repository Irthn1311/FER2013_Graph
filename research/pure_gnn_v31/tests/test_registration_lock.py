"""Scientific execution must remain locked pending explicit registration."""

from pathlib import Path

import pytest
import yaml

from pure_gnn_v31.cli.train_research import validate_scientific_registration


def test_committed_research_scaffold_has_only_null_registration_fields():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "research_screen.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["meta"]["authorized"] is False
    assert config["registration"]
    assert all(value is None for value in config["registration"].values())
    with pytest.raises(PermissionError):
        validate_scientific_registration(str(config_path))
