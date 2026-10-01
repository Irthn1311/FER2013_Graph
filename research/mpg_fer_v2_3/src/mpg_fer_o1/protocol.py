"""Fail-closed scientific and artifact contracts for MPG-FER O1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from mpg_fer_v2_3.checkpoint import canonical_config, config_hash, sha256_file
from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.data import validate_split_path
from mpg_fer_v2_3.train import is_better_checkpoint


ISSUE_URL = "https://github.com/Irthn1311/FER2013_Graph/issues/103"
BASE_COMMIT = "232e7a9f09251e7c3353684d34351356bd2b023b"
FROZEN_SCIENTIFIC_SOURCE_SHA256 = (
    "1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87"
)
BASELINE_CONFIG_SHA256 = (
    "8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2"
)
HISTORICAL_PUBLIC_RESULTS_NAME = "v23_public_results.json"
HISTORICAL_PUBLIC_RESULTS_SHA256 = (
    "da99889dd993da18c57f2e0f74afd6921abb052e9705a452e543d840df6a9363"
)
HISTORICAL_METRIC_DATASET_ROLE = "PublicTest"
BASELINE_ID = "O1_C0_BASELINE"
SCREEN_STOP_EPOCH = 65
RIGHT_CENSORED_STATUS = "RIGHT_CENSORED_AT_SCREEN_LIMIT"
RAW_GUARDRAIL_STATUS = "RAW_GUARDRAIL_WARNING"
PRIVATE_PATH_MARKERS = frozenset(
    {"test.csv", "privatetest", "private_test", "private-test"}
)
CHECKPOINT_SELECTION = (
    "higher EMA Public horizontal-flip TTA accuracy, then higher EMA TTA "
    "Macro-F1, then lower EMA TTA CE loss"
)
ALLOWED_SCIENTIFIC_DELTA_FIELDS = frozenset(
    {"learning_rate", "lr_decay_end_epoch"}
)
REQUIRED_RUN_ARTIFACTS = (
    "resolved_config.json",
    "hpo_manifest.json",
    "history.json",
    "history.csv",
    "best_val_acc.pt",
    "execution_manifest.json",
    "selected_public_metrics.json",
    "segment_manifest.json",
    "resume_latest.pt",
    "checksums.sha256",
)


@dataclass(frozen=True)
class O1Spec:
    config_id: str
    learning_rate: float
    lr_decay_end_epoch: int
    seed: int = 42
    source_type: str = "new_screening_job"


_GRID = (
    ("O1_01", 1.5e-4, 65),
    ("O1_02", 2.0e-4, 65),
    ("O1_03", 2.5e-4, 65),
    ("O1_04", 3.0e-4, 65),
    ("O1_05", 4.0e-4, 65),
    ("O1_06", 1.5e-4, 75),
    ("O1_07", 2.0e-4, 75),
    ("O1_08", 2.5e-4, 75),
    ("O1_09", 3.0e-4, 75),
    ("O1_10", 4.0e-4, 75),
    ("O1_11", 1.5e-4, 85),
    ("O1_12", 2.0e-4, 85),
    ("O1_13", 2.5e-4, 85),
    ("O1_14", 4.0e-4, 85),
)
O1_REGISTRY = {
    config_id: O1Spec(config_id, learning_rate, horizon)
    for config_id, learning_rate, horizon in _GRID
}
O1_CONFIG_ORDER = tuple(O1_REGISTRY)
HISTORICAL_CONTROL = O1Spec(
    BASELINE_ID, 3.0e-4, 85, source_type="external_historical_control"
)


def frozen_baseline_config(**runtime_overrides: Any) -> MPGConfig:
    """Return the exact v2.3 seed-42 recipe with runtime-only overrides."""
    allowed = set(MPGConfig().runtime_safe_resume_fields) | {"device"}
    extra = set(runtime_overrides) - allowed
    if extra:
        raise ValueError(f"Non-runtime baseline overrides forbidden: {sorted(extra)}")
    config = MPGConfig(**runtime_overrides)
    if config_hash(config) != BASELINE_CONFIG_SHA256:
        raise RuntimeError("Frozen MPG-FER baseline config identity mismatch")
    return config


def resolve_o1_config(config_id: str, **runtime_overrides: Any) -> MPGConfig:
    """Resolve the only two scientific deltas through the immutable registry."""
    try:
        spec = O1_REGISTRY[config_id]
    except KeyError as exc:
        raise ValueError(f"Unknown or non-runnable O1 config ID: {config_id!r}") from exc
    baseline = frozen_baseline_config(**runtime_overrides)
    resolved = replace(
        baseline,
        learning_rate=spec.learning_rate,
        lr_decay_end_epoch=spec.lr_decay_end_epoch,
    )
    verify_single_delta(config_id, resolved)
    return resolved


def scientific_diff(
    left: MPGConfig, right: MPGConfig
) -> dict[str, tuple[Any, Any]]:
    a, b = canonical_config(left), canonical_config(right)
    return {key: (a[key], b[key]) for key in a if a[key] != b[key]}


def verify_single_delta(config_id: str, resolved: MPGConfig) -> dict[str, Any]:
    spec = O1_REGISTRY[config_id]
    baseline = frozen_baseline_config(device=resolved.device)
    differences = scientific_diff(baseline, resolved)
    if not set(differences).issubset(ALLOWED_SCIENTIFIC_DELTA_FIELDS):
        raise RuntimeError(f"O1 scientific config drift: {sorted(differences)}")
    if (
        resolved.seed != 42
        or resolved.learning_rate != spec.learning_rate
        or resolved.lr_decay_end_epoch != spec.lr_decay_end_epoch
        or resolved.max_epochs != baseline.max_epochs
    ):
        raise RuntimeError("O1 registry/config resolution mismatch")
    return {
        "declared_delta_fields": sorted(ALLOWED_SCIENTIFIC_DELTA_FIELDS),
        "value_changed_fields": sorted(differences),
        "baseline_config_sha256": config_hash(baseline),
        "resolved_config_sha256": config_hash(resolved),
    }


def registry_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "issue": ISSUE_URL,
        "architecture_base_commit": BASE_COMMIT,
        "frozen_scientific_source_sha256": FROZEN_SCIENTIFIC_SOURCE_SHA256,
        "screening_seed": 42,
        "screen_stop_epoch": SCREEN_STOP_EPOCH,
        "allowed_scientific_delta_fields": sorted(ALLOWED_SCIENTIFIC_DELTA_FIELDS),
        "new_job_count": 14,
        "new_jobs": [asdict(O1_REGISTRY[key]) for key in O1_CONFIG_ORDER],
        "historical_control": asdict(HISTORICAL_CONTROL),
        "historical_control_is_new_job": False,
    }


def o1_source_tree_hash(source_root: str | Path | None = None) -> str:
    root = Path(source_root) if source_root else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for package_name in ("mpg_fer_v2_3", "mpg_fer_o1"):
        for path in sorted((root / package_name).glob("*.py")):
            relative = f"{package_name}/{path.name}"
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _finite_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and Infinity are forbidden in O1 JSON")
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_finite_json(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _contains_private_marker(path: Path) -> bool:
    return any(part.lower() in PRIVATE_PATH_MARKERS for part in path.parts)


def validate_historical_public_results_artifact(
    public_results_path: str | Path,
) -> dict[str, str]:
    """Bind baseline metrics to the exact already-reviewed PublicTest artifact."""
    path = Path(public_results_path)
    if _contains_private_marker(path):
        raise RuntimeError("BASELINE_PUBLIC_REFUSED: forbidden split/path marker")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = sha256_file(path)
    if (
        path.name != HISTORICAL_PUBLIC_RESULTS_NAME
        or actual_sha256 != HISTORICAL_PUBLIC_RESULTS_SHA256
    ):
        raise RuntimeError(
            "BASELINE_PUBLIC_REFUSED: metrics are not the exact reviewed "
            "PublicTest artifact"
        )
    return {
        "dataset_role": HISTORICAL_METRIC_DATASET_ROLE,
        "artifact_name": HISTORICAL_PUBLIC_RESULTS_NAME,
        "artifact_sha256": HISTORICAL_PUBLIC_RESULTS_SHA256,
        "identity_basis": "exact_reviewed_public_results_name_and_sha256",
    }


def validate_mounted_input_firewall(
    input_root: str | Path = "/kaggle/input",
) -> dict[str, Any]:
    """Inspect names only and reject any mounted PrivateTest marker."""
    root = Path(input_root)
    if not root.is_dir():
        raise RuntimeError(f"PRIVATE_FIREWALL: input root unavailable: {root}")

    def scan_error(error: OSError) -> None:
        raise RuntimeError(f"PRIVATE_FIREWALL: path enumeration failed: {error}")

    inspected = 0
    for current, directories, files in os.walk(
        root, topdown=True, onerror=scan_error, followlinks=False
    ):
        parent = Path(current).relative_to(root)
        for name in (*directories, *files):
            relative = parent / name
            inspected += 1
            if _contains_private_marker(relative):
                raise RuntimeError(
                    "PRIVATE_FIREWALL: forbidden mounted path marker: "
                    f"{relative.as_posix()}"
                )
    return {
        "input_root": str(root),
        "entries_inspected_by_name_only": inspected,
        "forbidden_private_markers_found": False,
    }


def validate_train_public_paths(
    train_csv: str | Path, public_csv: str | Path
) -> tuple[Path, Path]:
    paths = (Path(train_csv), Path(public_csv))
    if any(_contains_private_marker(path) for path in paths):
        raise RuntimeError("PRIVATE_FIREWALL: forbidden Train/Public path")
    train = validate_split_path(paths[0], "train")
    public = validate_split_path(paths[1], "val")
    if train.resolve() == public.resolve():
        raise RuntimeError("PRIVATE_FIREWALL: ambiguous split identity")
    return train, public


def validate_design_lock(
    design_path: str | Path,
    *,
    expected_source_sha256: str,
    expected_notebook_sha256: str,
) -> dict[str, Any]:
    path = Path(design_path)
    if not path.is_file():
        raise RuntimeError("O1_EXECUTION_REFUSED: O1_HPO_DESIGN_LOCK.json absent")
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("o1_source_tree_sha256") != expected_source_sha256
        or document.get("notebook_sha256") != expected_notebook_sha256
        or document.get("frozen_scientific_source_sha256")
        != FROZEN_SCIENTIFIC_SOURCE_SHA256
        or document.get("wave1_execution_authorized") is not True
    ):
        raise RuntimeError("O1_EXECUTION_REFUSED: design/source authorization mismatch")
    return document


def screen_stop_identity(config: MPGConfig) -> dict[str, Any]:
    """Operational stop metadata deliberately outside scientific config identity."""
    return {
        "screen_stop_epoch": SCREEN_STOP_EPOCH,
        "scientific_max_epochs": config.max_epochs,
        "scientific_config_sha256": config_hash(config),
        "scheduler_identity": {
            "base_lr": config.learning_rate,
            "warmup_epochs": config.warmup_epochs,
            "lr_decay_end_epoch": config.lr_decay_end_epoch,
            "max_epochs": config.max_epochs,
            "eta_min": config.min_learning_rate,
        },
    }


def right_censor_decision(
    *, selected_epoch: int, comparator_trajectory: Sequence[Mapping[str, float]]
) -> dict[str, Any]:
    if not comparator_trajectory:
        raise ValueError("Comparator trajectory is required")
    ordered = sorted(comparator_trajectory, key=lambda item: int(item["epoch"]))
    if int(ordered[-1]["epoch"]) != SCREEN_STOP_EPOCH:
        raise ValueError("Trajectory must end at completed epoch 65")
    improving_at_stop = len(ordered) >= 2 and is_better_checkpoint(
        dict(ordered[-1]), dict(ordered[-2])
    )
    selected_near_limit = 61 <= int(selected_epoch) <= SCREEN_STOP_EPOCH
    censored = selected_near_limit or improving_at_stop
    return {
        "status": RIGHT_CENSORED_STATUS if censored else "NOT_RIGHT_CENSORED",
        "right_censored": censored,
        "selected_epoch_in_61_65": selected_near_limit,
        "comparator_still_improving_at_epoch_65": improving_at_stop,
        "requires_same_config_extension_before_final_promotion": censored,
    }


def raw_guardrail(
    candidate_raw: Mapping[str, float], baseline_raw: Mapping[str, float]
) -> dict[str, Any]:
    # Compare decimal representations so the preregistered strict threshold
    # does not accidentally turn an exact -0.50 pp delta into a warning due
    # to binary floating-point rounding.
    accuracy_delta = Decimal(str(candidate_raw["accuracy"])) - Decimal(
        str(baseline_raw["accuracy"])
    )
    macro_f1_delta = Decimal(str(candidate_raw["macro_f1"])) - Decimal(
        str(baseline_raw["macro_f1"])
    )
    threshold = Decimal("-0.005")
    warning = accuracy_delta < threshold or macro_f1_delta < threshold
    accuracy_delta_pp = float(Decimal(100) * accuracy_delta)
    macro_f1_delta_pp = float(Decimal(100) * macro_f1_delta)
    return {
        "status": RAW_GUARDRAIL_STATUS if warning else "RAW_GUARDRAIL_PASS",
        "warning": warning,
        "accuracy_delta_percentage_points": accuracy_delta_pp,
        "macro_f1_delta_percentage_points": macro_f1_delta_pp,
        "explicit_review_required_before_promotion": warning,
    }


def validate_baseline_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "status",
        "usable_for_o1_aggregation",
        "config_id",
        "frozen_scientific_source_sha256",
        "seed",
        "learning_rate",
        "lr_decay_end_epoch",
        "metric_dataset_role",
        "public_results_provenance",
        "checkpoint_provenance",
        "checkpoint_sha256",
        "history_provenance",
        "history_sha256",
        "selected_epoch",
        "public_metrics",
        "private_test_artifacts_read",
    }
    if set(reference) != required:
        raise RuntimeError("BASELINE_REFERENCE_REFUSED: incomplete or extra fields")
    if (
        reference["status"] != "VERIFIED"
        or reference["usable_for_o1_aggregation"] is not True
        or reference["config_id"] != BASELINE_ID
        or reference["frozen_scientific_source_sha256"]
        != FROZEN_SCIENTIFIC_SOURCE_SHA256
        or reference["seed"] != 42
        or reference["learning_rate"] != 3.0e-4
        or reference["lr_decay_end_epoch"] != 85
        or reference["metric_dataset_role"] != HISTORICAL_METRIC_DATASET_ROLE
        or reference["private_test_artifacts_read"] is not False
    ):
        raise RuntimeError("BASELINE_REFERENCE_REFUSED: provenance/identity mismatch")
    public_provenance = reference["public_results_provenance"]
    if public_provenance != {
        "dataset_role": HISTORICAL_METRIC_DATASET_ROLE,
        "artifact_name": HISTORICAL_PUBLIC_RESULTS_NAME,
        "artifact_sha256": HISTORICAL_PUBLIC_RESULTS_SHA256,
        "identity_basis": "exact_reviewed_public_results_name_and_sha256",
    }:
        raise RuntimeError("BASELINE_REFERENCE_REFUSED: PublicTest identity mismatch")
    for field in ("checkpoint_sha256", "history_sha256"):
        value = reference[field]
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"BASELINE_REFERENCE_REFUSED: invalid {field}")
    metrics = reference["public_metrics"]
    if set(metrics) != {"raw", "tta"} or any(
        set(metrics[view]) != {"loss", "accuracy", "macro_f1"}
        for view in ("raw", "tta")
    ):
        raise RuntimeError("BASELINE_REFERENCE_REFUSED: incomplete Public metrics")
    _finite_json(reference)
    return dict(reference)


def write_resolved_config(path: str | Path, config: MPGConfig) -> Path:
    return write_json(
        path,
        {
            "schema_version": 1,
            "scientific_config_sha256": config_hash(config),
            "scientific_config": canonical_config(config),
            "runtime_config": {
                key: asdict(config)[key]
                for key in config.runtime_safe_resume_fields
            },
        },
    )


def write_hpo_manifest(
    output_dir: str | Path,
    *,
    config_id: str,
    config: MPGConfig,
    source_sha256: str,
    notebook_sha256: str,
) -> Path:
    spec = O1_REGISTRY[config_id]
    return write_json(
        Path(output_dir) / "hpo_manifest.json",
        {
            "schema_version": 1,
            "issue": ISSUE_URL,
            "config_id": config_id,
            "seed": 42,
            "learning_rate": spec.learning_rate,
            "lr_decay_end_epoch": spec.lr_decay_end_epoch,
            "source_sha256": source_sha256,
            "scientific_config_sha256": config_hash(config),
            "notebook_sha256": notebook_sha256,
            "frozen_scientific_source_sha256": FROZEN_SCIENTIFIC_SOURCE_SHA256,
            "screen_stop_epoch": SCREEN_STOP_EPOCH,
            "scientific_max_epochs": config.max_epochs,
            "checkpoint_selection": CHECKPOINT_SELECTION,
            "PRIVATE_EVALUATED": False,
        },
    )


def write_run_checksums(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    missing = [name for name in REQUIRED_RUN_ARTIFACTS[:-1] if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Cannot finalize O1 checksums; missing: {missing}")
    target = root / "checksums.sha256"
    target.write_text(
        "\n".join(
            f"{sha256_file(root / name)}  {name}" for name in REQUIRED_RUN_ARTIFACTS[:-1]
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def _comparator_key(metrics: Mapping[str, float]) -> tuple[float, float, float]:
    return (
        float(metrics["accuracy"]),
        float(metrics["macro_f1"]),
        -float(metrics["loss"]),
    )


def _strictly_monotone(keys: Sequence[tuple[float, float, float]]) -> bool:
    return len(keys) >= 2 and all(a > b for a, b in zip(keys, keys[1:]))


def _canonical_mapping_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _locked_relative_artifact(design_path: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str):
        raise RuntimeError(f"O1 aggregation design lacks {label} path")
    root = design_path.parent.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"O1 aggregation design {label} path escapes lock root") from exc
    if not candidate.is_file():
        raise RuntimeError(f"O1 aggregation design {label} artifact is absent")
    return candidate


def validate_aggregation_design_lock(
    design_lock_path: str | Path,
    baseline_reference_path: str | Path,
) -> dict[str, Any]:
    """Verify the authorized design and every immutable artifact it binds."""
    design_path = Path(design_lock_path)
    if not design_path.is_file():
        raise RuntimeError("O1 aggregation requires a reviewed design lock")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if (
        design.get("wave1_execution_authorized") is not True
        or design.get("private_test_permitted") is not False
        or design.get("frozen_scientific_source_sha256")
        != FROZEN_SCIENTIFIC_SOURCE_SHA256
        or design.get("o1_source_tree_sha256") != o1_source_tree_hash()
        or design.get("new_config_ids") != list(O1_CONFIG_ORDER)
        or design.get("historical_control_id") != BASELINE_ID
        or design.get("screen_stop_epoch") != SCREEN_STOP_EPOCH
        or design.get("scientific_max_epochs") != 120
    ):
        raise RuntimeError("O1 aggregation reviewed design identity mismatch")

    registry_path = _locked_relative_artifact(
        design_path, design.get("registry_path"), "registry"
    )
    notebook_path = _locked_relative_artifact(
        design_path, design.get("notebook_path"), "notebook"
    )
    baseline_path = Path(baseline_reference_path)
    if (
        sha256_file(registry_path) != design.get("registry_sha256")
        or sha256_file(notebook_path) != design.get("notebook_sha256")
        or not baseline_path.is_file()
        or sha256_file(baseline_path) != design.get("baseline_reference_sha256")
    ):
        raise RuntimeError("O1 aggregation reviewed artifact hash mismatch")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry != _finite_json(registry_document()):
        raise RuntimeError("O1 aggregation registry content mismatch")
    return design


def _validate_completed_run_artifacts(
    root: Path, design: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    missing = [name for name in REQUIRED_RUN_ARTIFACTS if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"O1 aggregation refuses partial run {root}: missing {missing}")

    expected_names = set(REQUIRED_RUN_ARTIFACTS[:-1])
    recorded: dict[str, str] = {}
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"Malformed O1 checksum line in {root}")
        digest, name = fields
        if name in recorded or name not in expected_names or Path(name).name != name:
            raise RuntimeError(f"Unexpected O1 checksum entry in {root}: {name}")
        recorded[name] = digest
    if set(recorded) != expected_names:
        raise RuntimeError(f"Incomplete O1 checksums in {root}")
    for name, expected in recorded.items():
        if len(expected) != 64 or sha256_file(root / name) != expected:
            raise RuntimeError(f"O1 artifact checksum mismatch: {root / name}")

    manifest = json.loads((root / "hpo_manifest.json").read_text(encoding="utf-8"))
    resolved = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    selected = json.loads(
        (root / "selected_public_metrics.json").read_text(encoding="utf-8")
    )
    execution = json.loads(
        (root / "execution_manifest.json").read_text(encoding="utf-8")
    )
    segment = json.loads((root / "segment_manifest.json").read_text(encoding="utf-8"))
    config_id = manifest.get("config_id")
    if config_id not in O1_REGISTRY:
        raise RuntimeError(f"Unexpected O1 config: {config_id}")
    expected_config = resolve_o1_config(config_id)
    expected_scientific = canonical_config(expected_config)
    expected_runtime_fields = set(expected_config.runtime_safe_resume_fields)
    if (
        set(resolved) != {
            "schema_version",
            "scientific_config_sha256",
            "scientific_config",
            "runtime_config",
        }
        or resolved.get("schema_version") != 1
        or not isinstance(resolved.get("scientific_config"), dict)
        or not isinstance(resolved.get("runtime_config"), dict)
        or set(resolved["runtime_config"]) != expected_runtime_fields
    ):
        raise RuntimeError(f"O1 resolved config schema mismatch: {root}")
    resolved_hash = _canonical_mapping_hash(resolved["scientific_config"])
    if (
        resolved.get("scientific_config_sha256") != resolved_hash
        or resolved["scientific_config"] != _finite_json(expected_scientific)
        or manifest.get("scientific_config_sha256") != resolved_hash
    ):
        raise RuntimeError(f"O1 resolved config identity mismatch: {root}")
    if (
        execution.get("status") != "SCREENING_COMPLETED"
        or segment.get("status") != "SCREENING_COMPLETED"
        or execution.get("screen_stop_epoch") != SCREEN_STOP_EPOCH
        or execution.get("scientific_max_epochs") != 120
        or execution.get("PRIVATE_EVALUATED") is not False
        or selected.get("PRIVATE_EVALUATED") is not False
    ):
        raise RuntimeError(f"O1 aggregation refuses incomplete/non-public run: {root}")
    if (
        manifest.get("source_sha256") != design.get("o1_source_tree_sha256")
        or manifest.get("notebook_sha256") != design.get("notebook_sha256")
        or execution.get("source_sha256") != design.get("o1_source_tree_sha256")
        or execution.get("scientific_config_sha256") != resolved_hash
        or execution.get("config_id") != config_id
        or selected.get("config_id") != config_id
    ):
        raise RuntimeError(f"O1 run does not match reviewed design identity: {root}")
    checkpoint_sha = sha256_file(root / "best_val_acc.pt")
    if (
        execution.get("best_checkpoint_sha256") != checkpoint_sha
        or selected.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise RuntimeError(f"O1 selected checkpoint provenance mismatch: {root}")
    resumed_from = execution.get("resumed_from")
    if resumed_from is not None and (
        not isinstance(resumed_from, str) or not resumed_from.strip()
    ):
        raise RuntimeError(f"O1 exact-resume provenance mismatch: {root}")
    return manifest, selected, execution


def aggregate_o1(
    run_directories: Iterable[str | Path],
    baseline_reference_path: str | Path,
    design_lock_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    baseline_path = Path(baseline_reference_path)
    design = validate_aggregation_design_lock(design_lock_path, baseline_path)
    baseline = validate_baseline_reference(
        json.loads(baseline_path.read_text(encoding="utf-8"))
    )
    records: dict[str, dict[str, Any]] = {}
    for raw in run_directories:
        root = Path(raw)
        manifest, metrics, execution = _validate_completed_run_artifacts(root, design)
        config_id = manifest["config_id"]
        if config_id not in O1_REGISTRY or config_id in records:
            raise RuntimeError(f"Unexpected or duplicate O1 config: {config_id}")
        spec = O1_REGISTRY[config_id]
        if (
            manifest["seed"] != 42
            or manifest["learning_rate"] != spec.learning_rate
            or manifest["lr_decay_end_epoch"] != spec.lr_decay_end_epoch
            or manifest.get("screen_stop_epoch") != SCREEN_STOP_EPOCH
            or manifest.get("scientific_max_epochs") != 120
            or manifest.get("checkpoint_selection") != CHECKPOINT_SELECTION
            or manifest.get("PRIVATE_EVALUATED") is not False
            or metrics.get("config_id") != config_id
        ):
            raise RuntimeError(f"Scientific manifest mismatch: {config_id}")
        records[config_id] = {
            "config_id": config_id,
            "source_type": "new_screening_job",
            "learning_rate": spec.learning_rate,
            "lr_decay_end_epoch": spec.lr_decay_end_epoch,
            "completion_status": "SCREENING_COMPLETED",
            "exact_resume_used": execution.get("resumed_from") is not None,
            "resumed_from": execution.get("resumed_from"),
            **metrics,
        }
    missing = [key for key in O1_CONFIG_ORDER if key not in records]
    if missing:
        raise RuntimeError(f"O1 aggregation incomplete: {missing}")
    baseline_record = {
        "config_id": BASELINE_ID,
        "source_type": "external_historical_control",
        "learning_rate": 3.0e-4,
        "lr_decay_end_epoch": 85,
        "metric_dataset_role": HISTORICAL_METRIC_DATASET_ROLE,
        "selected_epoch": baseline["selected_epoch"],
        "public": baseline["public_metrics"],
        "right_censor": {
            "status": "HISTORICAL_CONTROL_NOT_SCREEN_CENSORED",
            "right_censored": False,
        },
        "checkpoint_sha256": baseline["checkpoint_sha256"],
    }
    canonical_rows = [records[key] for key in O1_CONFIG_ORDER] + [baseline_record]
    baseline_raw = baseline["public_metrics"]["raw"]
    for record in records.values():
        record["raw_guardrail"] = raw_guardrail(record["public"]["raw"], baseline_raw)

    ranking = sorted(
        records.values(), key=lambda item: _comparator_key(item["public"]["tta"]), reverse=True
    )
    best = ranking[0]
    same_horizon = sorted(
        (item for item in records.values() if item["lr_decay_end_epoch"] == best["lr_decay_end_epoch"]),
        key=lambda item: item["learning_rate"],
    )
    same_lr = sorted(
        (item for item in canonical_rows if item["learning_rate"] == best["learning_rate"]),
        key=lambda item: item["lr_decay_end_epoch"],
    )
    boundary: list[str] = []
    if best["learning_rate"] == 1.5e-4 and _strictly_monotone(
        [_comparator_key(item["public"]["tta"]) for item in same_horizon]
    ):
        boundary.append("LOW_LR_BOUNDARY_EXTENSION_REQUIRED")
    if best["learning_rate"] == 4.0e-4 and _strictly_monotone(
        [_comparator_key(item["public"]["tta"]) for item in reversed(same_horizon)]
    ):
        boundary.append("HIGH_LR_BOUNDARY_EXTENSION_REQUIRED")
    if best["lr_decay_end_epoch"] == 65 and _strictly_monotone(
        [_comparator_key(item["public"]["tta"]) for item in same_lr]
    ):
        boundary.append("SHORTER_DECAY_EXTENSION_REQUIRED")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "reviewed_design_lock_sha256": sha256_file(design_lock_path),
        "reviewed_source_sha256": design["o1_source_tree_sha256"],
        "reviewed_notebook_sha256": design["notebook_sha256"],
        "reviewed_registry_sha256": design["registry_sha256"],
        "reviewed_baseline_reference_sha256": design[
            "baseline_reference_sha256"
        ],
        "canonical_order": [*O1_CONFIG_ORDER, BASELINE_ID],
        "results": canonical_rows,
        "performance_ranking_is_separate": True,
    }
    write_json(output / "o1_results.json", result)
    with (output / "o1_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["config_id", "source_type", "learning_rate", "lr_decay_end_epoch"],
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in writer.fieldnames} for row in canonical_rows
        )
    surface_rows = []
    by_pair = {
        (row["learning_rate"], row["lr_decay_end_epoch"]): row
        for row in canonical_rows
    }
    for horizon in (65, 75, 85):
        for learning_rate in (1.5e-4, 2.0e-4, 2.5e-4, 3.0e-4, 4.0e-4):
            row = by_pair[(learning_rate, horizon)]
            surface_rows.append(
                {
                    "learning_rate": learning_rate,
                    "lr_decay_end_epoch": horizon,
                    "config_id": row["config_id"],
                    "source_type": row["source_type"],
                    "tta_accuracy": row["public"]["tta"]["accuracy"],
                    "tta_macro_f1": row["public"]["tta"]["macro_f1"],
                    "tta_loss": row["public"]["tta"]["loss"],
                }
            )
    with (output / "o1_response_surface.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(surface_rows[0]))
        writer.writeheader()
        writer.writerows(surface_rows)
    censored = [
        item["config_id"] for item in ranking if item["right_censor"]["right_censored"]
    ]
    warnings = [
        item["config_id"] for item in ranking if item["raw_guardrail"]["warning"]
    ]
    exact_resumed = [
        item["config_id"] for item in canonical_rows if item.get("exact_resume_used")
    ]
    fresh_completed = len(O1_CONFIG_ORDER) - len(exact_resumed)
    report = [
        "# MPG-FER O1 provisional promotion report",
        "",
        "Historical control: O1_C0_BASELINE (external; not a new job).",
        "New screening jobs: 14.",
        (
            "Completed new jobs accepted: 14 "
            f"(fresh: {fresh_completed}; exact-resumed: {len(exact_resumed)})."
        ),
        (
            "Completed exact-resume jobs accepted: "
            f"{', '.join(exact_resumed) if exact_resumed else 'none'}."
        ),
        "Incomplete/partial run artifacts: refused before aggregation.",
        f"Provisional top 3 novel configs: {', '.join(item['config_id'] for item in ranking[:3])}.",
        f"Right-censored (not safely eliminable): {', '.join(censored) if censored else 'none'}.",
        f"Raw guardrail warnings (explicit review required): {', '.join(warnings) if warnings else 'none'}.",
        f"Boundary-extension status: {', '.join(boundary) if boundary else 'NO_BOUNDARY_EXTENSION_SIGNAL'}.",
        "",
        "Exact-resumed runs are accepted only after reaching SCREENING_COMPLETED.",
        "No extension values are invented by this report.",
    ]
    (output / "o1_promotion_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    checksum_names = (
        "o1_registry.csv",
        "o1_results.json",
        "o1_response_surface.csv",
        "o1_promotion_report.md",
    )
    (output / "o1_checksums.sha256").write_text(
        "\n".join(f"{sha256_file(output / name)}  {name}" for name in checksum_names)
        + "\n",
        encoding="utf-8",
    )
    return {
        **result,
        "provisional_top3": [item["config_id"] for item in ranking[:3]],
        "right_censored_not_safely_eliminable": censored,
        "raw_guardrail_warnings": warnings,
        "boundary_extension_status": boundary,
        "completed_exact_resume_config_ids": exact_resumed,
    }
