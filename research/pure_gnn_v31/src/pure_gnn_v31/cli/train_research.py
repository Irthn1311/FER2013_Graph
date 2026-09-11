"""Research training harness scaffold for Pure-GNN v3.1.

SCIENTIFIC TRAINING IS NOT AUTHORIZED IN THIS TASK.
This CLI exists strictly for pipeline scaffolding and preflight integration.
"""

import argparse
from pathlib import Path

import yaml


REQUIRED_REGISTRATION_FIELDS = (
    "research_split_seed",
    "research_dev_ratio",
    "scientific_seeds",
    "epochs",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "early_stopping",
    "augmentation",
    "loss",
)


def validate_scientific_registration(config_path: str) -> dict:
    """Fail closed unless a future reviewed config is explicit and authorized."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    registration = config.get("registration") or {}
    missing = [field for field in REQUIRED_REGISTRATION_FIELDS if registration.get(field) is None]
    if config.get("meta", {}).get("authorized") is not True or missing:
        raise PermissionError(
            "Scientific training is NOT authorized; registration is locked or incomplete. "
            f"Missing explicit fields: {missing}"
        )
    return config


def main():
    parser = argparse.ArgumentParser(description="Pure-GNN v3.1 Research Training Harness")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--condition", type=str, default="G1", choices=["G0", "G0.5", "G1", "G2", "G3"])
    parser.add_argument("--execute-scientific-training", action="store_true", default=False)
    args = parser.parse_args()

    validate_scientific_registration(args.config)
    raise PermissionError(
        "Scientific training execution is intentionally unavailable in this technical snapshot."
    )


if __name__ == "__main__":
    main()
