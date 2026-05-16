"""Resolve and sanity-check the current D12A runtime parity config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "d12a_quality_b32_amp_no_compile_screen12.yaml"

FIELDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("run.config_name", ("run", "config_name")),
    ("training.epochs", ("training", "epochs")),
    ("training.batch_size", ("training", "batch_size")),
    ("data.batch_size", ("data", "batch_size")),
    ("training.amp", ("training", "amp")),
    ("training.multi_gpu", ("training", "multi_gpu")),
    ("training.use_compile", ("training", "use_compile")),
    ("training.torch_compile", ("training", "torch_compile")),
    ("ddp.enabled", ("ddp", "enabled")),
    ("data.fixed_batch_size", ("data", "fixed_batch_size")),
    ("data.ddp_chunk_aware", ("data", "ddp_chunk_aware")),
    ("data.chunk_cache_size", ("data", "chunk_cache_size")),
    ("data.graph_cache_chunks", ("data", "graph_cache_chunks")),
    ("model.use_global_branch", ("model", "use_global_branch")),
    ("model.encoder.use_scale2", ("model", "encoder", "use_scale2")),
    ("model.slot_iterations", ("model", "slot_iterations")),
    ("model.residual_slot_connection", ("model", "residual_slot_connection")),
    ("loss.lambda_supcon", ("loss", "lambda_supcon")),
    ("loss.lambda_div", ("loss", "lambda_div")),
    ("loss.lambda_spatial", ("loss", "lambda_spatial")),
    ("loss.ce_warmup_epochs", ("loss", "ce_warmup_epochs")),
    ("loss.class_weight_power", ("loss", "class_weight_power")),
    ("loss.label_smoothing", ("loss", "label_smoothing")),
)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config_tree(config_path: Path, seen: set[Path] | None = None) -> Dict[str, Any]:
    path = config_path.resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"Circular config inheritance detected at: {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    inherited = cfg.pop("inherits", [])
    if isinstance(inherited, (str, Path)):
        inherited = [inherited]

    merged: Dict[str, Any] = {}
    for parent in inherited:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = deep_update(merged, load_config_tree(parent_path, seen))
    seen.remove(path)
    merged = deep_update(merged, cfg)
    merged.setdefault("run", {}).setdefault("config_name", path.stem)
    return merged


def get_nested(config: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def print_table(config: Dict[str, Any]) -> None:
    rows: List[Tuple[str, str]] = [
        (label, fmt(get_nested(config, keys))) for label, keys in FIELDS
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label.ljust(width)} : {value}")


def assert_runtime(config: Dict[str, Any]) -> None:
    name = get_nested(config, ("run", "config_name"))
    assert name == "d12a_quality_b32_amp_no_compile_screen12", name
    assert get_nested(config, ("training", "epochs")) == 12
    assert get_nested(config, ("training", "batch_size")) == 32
    assert get_nested(config, ("data", "batch_size")) == 32
    assert get_nested(config, ("training", "amp")) is True
    assert get_nested(config, ("training", "multi_gpu")) is True
    assert get_nested(config, ("training", "use_compile")) is False
    assert get_nested(config, ("training", "torch_compile"), False) is False
    assert get_nested(config, ("ddp", "enabled")) is False
    assert get_nested(config, ("loss", "lambda_supcon")) == 0.0
    assert get_nested(config, ("loss", "lambda_div")) == 0.0
    assert get_nested(config, ("loss", "lambda_spatial")) == 0.0
    assert get_nested(config, ("loss", "ce_warmup_epochs")) == 0
    assert get_nested(config, ("model", "use_global_branch")) is True
    assert get_nested(config, ("model", "encoder", "use_scale2")) is True
    assert get_nested(config, ("model", "slot_iterations")) == 3
    assert get_nested(config, ("model", "residual_slot_connection")) is False


def main() -> None:
    config = load_config_tree(CONFIG_PATH)
    print_table(config)
    assert_runtime(config)
    print("\nD12 runtime config check OK")


if __name__ == "__main__":
    main()
