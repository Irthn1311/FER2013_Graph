"""Build machine-readable A3 config, source-diff, and parameter audits."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_ROOT = PROJECT_ROOT / "research" / "mpg_fer_audit"
V21_ROOT = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "src" / "mpg_fer_v2_1"
V22_ROOT = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "src" / "mpg_fer_v2_2"
sys.path.insert(0, str(V21_ROOT.parent))
sys.path.insert(0, str(V22_ROOT.parent))

from mpg_fer_v2_1.config import MPGConfig as MPGConfigV21  # noqa: E402
from mpg_fer_v2_1.model import MPGFER as MPGFERV21  # noqa: E402
from mpg_fer_v2_1.train import source_tree_hash as source_tree_hash_v21  # noqa: E402
from mpg_fer_v2_2.config import MPGConfig  # noqa: E402
from mpg_fer_v2_2.model import MPGFER  # noqa: E402
from mpg_fer_v2_2.train import source_tree_hash as source_tree_hash_v22  # noqa: E402


FILE_CLASSIFICATION = {
    "__init__.py": (
        "identity_metadata",
        False,
        "package and Issue identity only",
    ),
    "checkpoint.py": (
        "identity_metadata",
        False,
        "module identity only; bundle layout and schema remain unchanged",
    ),
    "config.py": (
        "scientific_configuration",
        False,
        "declares and fail-closed validates the locked Top-K schedule; operator behavior is implemented in model.py",
    ),
    "data.py": ("byte_identical", False, "no difference"),
    "ema.py": ("byte_identical", False, "no difference"),
    "evaluate.py": ("byte_identical", False, "no difference"),
    "features.py": ("byte_identical", False, "no difference"),
    "graph.py": ("byte_identical", False, "no difference"),
    "kaggle.py": (
        "engineering_identity",
        False,
        "v2.2-only resume dataset identity; resume verification semantics unchanged",
    ),
    "losses.py": ("identity_metadata", False, "module identity only"),
    "model.py": (
        "scientific_operator",
        True,
        "dynamic hard Top-K non-self support before softmax plus read-only routing diagnostics",
    ),
    "motif.py": ("identity_metadata", False, "version text only"),
    "train.py": ("identity_metadata", False, "version text only"),
    "utils.py": ("byte_identical", False, "no difference"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _top_level_parameters(model) -> dict[str, int]:
    totals: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        totals[group] = totals.get(group, 0) + parameter.numel()
    return totals


def main() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    v21_config = asdict(MPGConfigV21())
    v22_config = asdict(MPGConfig())
    config_fields = sorted(set(v21_config) | set(v22_config))
    config_differences = [
        {
            "field": field,
            "v2_1": v21_config.get(field, "__ABSENT__"),
            "v2_2": v22_config.get(field, "__ABSENT__"),
            "classification": "registered_sparse_routing_field",
        }
        for field in config_fields
        if v21_config.get(field, "__ABSENT__")
        != v22_config.get(field, "__ABSENT__")
    ]
    _write(
        AUDIT_ROOT / "a3_config_diff.json",
        {
            "base_sha": "4967cc5dac3495be2300210215f72422f6f97aa4",
            "equal_except_registered_field": [
                item["field"] for item in config_differences
            ] == ["motif_topk_schedule"],
            "differences": config_differences,
            "resume_schema_version_v2_1": v21_config["resume_schema_version"],
            "resume_schema_version_v2_2": v22_config["resume_schema_version"],
        },
    )

    source_rows = []
    for name, (category, scientific_changed, reason) in FILE_CLASSIFICATION.items():
        v21_path = V21_ROOT / name
        v22_path = V22_ROOT / name
        source_rows.append(
            {
                "file": name,
                "v2_1_sha256": _sha256(v21_path),
                "v2_2_sha256": _sha256(v22_path),
                "byte_identical": v21_path.read_bytes() == v22_path.read_bytes(),
                "difference_category": category,
                "scientific_behavior_changed": scientific_changed,
                "reason": reason,
            }
        )
    _write(
        AUDIT_ROOT / "a3_source_diff.json",
        {
            "v2_1_source_hash": source_tree_hash_v21(V21_ROOT),
            "v2_2_source_hash": source_tree_hash_v22(V22_ROOT),
            "only_scientific_behavior_change_file": [
                row["file"]
                for row in source_rows
                if row["scientific_behavior_changed"]
            ],
            "files": source_rows,
        },
    )

    v21_parameters = _top_level_parameters(MPGFERV21(MPGConfigV21()))
    v22_parameters = _top_level_parameters(MPGFER(MPGConfig()))
    groups = sorted(set(v21_parameters) | set(v22_parameters))
    reconciliation = [
        {
            "module": group,
            "v2_1": v21_parameters.get(group, 0),
            "v2_2": v22_parameters.get(group, 0),
            "delta": v22_parameters.get(group, 0) - v21_parameters.get(group, 0),
        }
        for group in groups
    ]
    _write(
        AUDIT_ROOT / "a3_parameter_reconciliation.json",
        {
            "v2_1_total": sum(v21_parameters.values()),
            "v2_2_total": sum(v22_parameters.values()),
            "total_delta": sum(v22_parameters.values())
            - sum(v21_parameters.values()),
            "all_module_deltas_zero": all(
                row["delta"] == 0 for row in reconciliation
            ),
            "modules": reconciliation,
        },
    )


if __name__ == "__main__":
    main()
