"""Resolve and sanity-check the D12A FastIO-safe parity configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"

CONFIG_PATHS = (
    CONFIG_DIR / "d12a_fastio_safe_screen8.yaml",
    CONFIG_DIR / "d12a_fastio_safe_val15.yaml",
    CONFIG_DIR / "d12a_quality_safe_b32_noamp_screen8.yaml",
)

FIELDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("config_name", ("run", "config_name")),
    ("training.epochs", ("training", "epochs")),
    ("data.batch_size", ("data", "batch_size")),
    ("training.batch_size", ("training", "batch_size")),
    ("training.amp", ("training", "amp")),
    ("training.multi_gpu", ("training", "multi_gpu")),
    ("training.use_compile", ("training", "use_compile")),
    ("training.torch_compile", ("training", "torch_compile")),
    ("data.chunk_cache_size", ("data", "chunk_cache_size")),
    ("data.graph_cache_chunks", ("data", "graph_cache_chunks")),
    ("data.chunk_aware_shuffle", ("data", "chunk_aware_shuffle")),
    ("data.shuffle_chunks", ("data", "shuffle_chunks")),
    ("data.shuffle_within_chunk", ("data", "shuffle_within_chunk")),
    ("model.use_global_branch", ("model", "use_global_branch")),
    ("model.encoder.use_scale2", ("model", "encoder", "use_scale2")),
    ("model.slot_iterations", ("model", "slot_iterations")),
    ("loss.lambda_supcon", ("loss", "lambda_supcon")),
    ("loss.lambda_div", ("loss", "lambda_div")),
    ("loss.ce_warmup_epochs", ("loss", "ce_warmup_epochs")),
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


def assert_no_ddp(config: Dict[str, Any], name: str) -> None:
    for section in ("training", "data", "runtime"):
        value = get_nested(config, (section, "ddp"), None)
        assert value in (None, False), f"{name}: {section}.ddp must be false or absent, got {value!r}"
    distributed = get_nested(config, ("training", "distributed"), None)
    assert distributed in (None, False), f"{name}: training.distributed must be false or absent"
    ddp_chunk = get_nested(config, ("data", "ddp_chunk_aware"), False)
    assert ddp_chunk is False, f"{name}: data.ddp_chunk_aware must be false"


def assert_common_no_compile(config: Dict[str, Any], name: str) -> None:
    assert get_nested(config, ("training", "use_compile")) is False, f"{name}: use_compile must be false"
    assert get_nested(config, ("training", "torch_compile"), False) is False, f"{name}: torch_compile must be false"
    assert_no_ddp(config, name)


def assert_fastio(config: Dict[str, Any], name: str) -> None:
    assert get_nested(config, ("data", "batch_size")) == 64, f"{name}: data.batch_size must be 64"
    assert get_nested(config, ("training", "batch_size")) == 64, f"{name}: training.batch_size must be 64"
    assert get_nested(config, ("training", "amp")) is True, f"{name}: amp must be true"
    assert_common_no_compile(config, name)


def assert_quality_safe(config: Dict[str, Any], name: str) -> None:
    assert get_nested(config, ("data", "batch_size")) == 32, f"{name}: data.batch_size must be 32"
    assert get_nested(config, ("training", "batch_size")) == 32, f"{name}: training.batch_size must be 32"
    assert get_nested(config, ("training", "amp")) is False, f"{name}: amp must be false"
    assert_common_no_compile(config, name)


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [label for label, _ in FIELDS]
    table = [[fmt(row.get(header)) for header in headers] for row in rows]
    widths = [
        max(len(header), *(len(row[i]) for row in table))
        for i, header in enumerate(headers)
    ]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in table:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    rows: List[Dict[str, Any]] = []
    resolved: Dict[str, Dict[str, Any]] = {}
    for path in CONFIG_PATHS:
        config = load_config_tree(path)
        name = str(get_nested(config, ("run", "config_name"), path.stem))
        resolved[name] = config
        rows.append({label: get_nested(config, keys) for label, keys in FIELDS})

    print_table(rows)

    assert_fastio(resolved["d12a_fastio_safe_screen8"], "d12a_fastio_safe_screen8")
    assert_fastio(resolved["d12a_fastio_safe_val15"], "d12a_fastio_safe_val15")
    assert_quality_safe(
        resolved["d12a_quality_safe_b32_noamp_screen8"],
        "d12a_quality_safe_b32_noamp_screen8",
    )
    print("\nD12 FastIO-safe config check OK")


if __name__ == "__main__":
    main()
