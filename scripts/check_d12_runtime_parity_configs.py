"""Validate D12A runtime parity audit configs without training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


CONFIGS = {
    "d12a_parity_legacy_dp_ce_first": {
        "path": "configs/experiments/d12a_parity_legacy_dp_ce_first.yaml",
        "training.amp": False,
        "training.multi_gpu": True,
        "training.use_compile": False,
        "training.torch_compile": False,
        "ddp.enabled": False,
        "data.batch_size": 32,
        "data.fixed_batch_size": False,
        "data.ddp_chunk_aware": False,
        "data.drop_incomplete_batches": False,
        "data.carry_over_leftovers": False,
        "data.ddp_drop_last_batches": False,
    },
    "d12a_parity_legacy_dp_b64_amp_ce_first": {
        "path": "configs/experiments/d12a_parity_legacy_dp_b64_amp_ce_first.yaml",
        "training.amp": True,
        "training.multi_gpu": True,
        "training.use_compile": False,
        "training.torch_compile": False,
        "ddp.enabled": False,
        "data.batch_size": 64,
        "data.fixed_batch_size": False,
        "data.ddp_chunk_aware": False,
        "data.drop_incomplete_batches": False,
        "data.carry_over_leftovers": False,
        "data.ddp_drop_last_batches": False,
    },
    "d12a_parity_ddp_eager_noamp_ce_first": {
        "path": "configs/experiments/d12a_parity_ddp_eager_noamp_ce_first.yaml",
        "training.amp": False,
        "training.multi_gpu": False,
        "training.use_compile": False,
        "training.torch_compile": False,
        "ddp.enabled": True,
        "ddp.find_unused_parameters": True,
        "ddp.compile": False,
        "data.batch_size": 64,
        "data.fixed_batch_size": True,
        "data.ddp_chunk_aware": True,
        "data.drop_incomplete_batches": True,
        "data.carry_over_leftovers": True,
        "data.ddp_drop_last_batches": True,
    },
    "d12a_parity_ddp_amp_no_compile_ce_first": {
        "path": "configs/experiments/d12a_parity_ddp_amp_no_compile_ce_first.yaml",
        "training.amp": True,
        "training.multi_gpu": False,
        "training.use_compile": False,
        "training.torch_compile": False,
        "ddp.enabled": True,
        "ddp.find_unused_parameters": True,
        "ddp.compile": False,
        "data.batch_size": 64,
        "data.fixed_batch_size": True,
        "data.ddp_chunk_aware": True,
        "data.drop_incomplete_batches": True,
        "data.carry_over_leftovers": True,
        "data.ddp_drop_last_batches": True,
    },
    "d12a_parity_ddp_amp_compile_fixedshape_ce_first": {
        "path": "configs/experiments/d12a_parity_ddp_amp_compile_fixedshape_ce_first.yaml",
        "training.amp": True,
        "training.multi_gpu": False,
        "training.use_compile": True,
        "training.torch_compile": True,
        "training.compile_order": "before_ddp",
        "ddp.enabled": True,
        "ddp.find_unused_parameters": True,
        "ddp.compile": True,
        "ddp.compile_order": "before_ddp",
        "data.batch_size": 64,
        "data.fixed_batch_size": True,
        "data.ddp_chunk_aware": True,
        "data.drop_incomplete_batches": True,
        "data.carry_over_leftovers": True,
        "data.ddp_drop_last_batches": True,
    },
}

COMMON_EXPECTED = {
    "model.use_global_branch": True,
    "model.encoder.use_scale2": True,
    "model.slot_iterations": 3,
    "model.residual_slot_connection": False,
    "loss.class_weight_power": 0.25,
    "loss.label_smoothing": 0.05,
    "loss.lambda_supcon": 0.0,
    "loss.lambda_div": 0.0,
    "loss.lambda_spatial": 0.0,
    "loss.ce_warmup_epochs": 0,
    "training.epochs": 30,
    "training.early_stopping_patience": 15,
}

FORBIDDEN_LOSS_KEYS = (
    "lambda_rare_aux",
    "lambda_rare_margin",
    "logit_adjust_tau",
    "focal_gamma",
)

FORBIDDEN_DATA_KEYS = ("target_class_repeat_factors",)


def get_nested(config: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config_tree(config_path: str | Path, seen: set[Path] | None = None) -> Dict[str, Any]:
    path = (PROJECT_ROOT / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    seen = seen or set()
    if path in seen:
        raise ValueError(f"Circular config inheritance detected at {path}")
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
    run_cfg = dict(cfg.get("run", {}) or {})
    run_cfg.setdefault("config_name", path.stem)
    cfg["run"] = run_cfg
    return deep_update(merged, cfg)


def assert_equal(config: Dict[str, Any], key: str, expected: Any) -> None:
    actual = get_nested(config, key)
    if actual != expected:
        raise AssertionError(f"{key}: expected {expected!r}, got {actual!r}")


def row(config_name: str, config: Dict[str, Any]) -> str:
    fields = [
        config_name,
        str(get_nested(config, "training.amp")),
        str(get_nested(config, "training.multi_gpu")),
        str(get_nested(config, "training.use_compile")),
        str(get_nested(config, "training.torch_compile")),
        str(get_nested(config, "ddp.enabled")),
        str(get_nested(config, "ddp.find_unused_parameters")),
        str(get_nested(config, "ddp.compile")),
        str(get_nested(config, "training.compile_order", get_nested(config, "ddp.compile_order"))),
        str(get_nested(config, "data.batch_size")),
        str(get_nested(config, "data.fixed_batch_size")),
        str(get_nested(config, "data.ddp_chunk_aware")),
        str(get_nested(config, "data.drop_incomplete_batches")),
        str(get_nested(config, "data.carry_over_leftovers")),
        str(get_nested(config, "data.ddp_drop_last_batches")),
        str(get_nested(config, "model.use_global_branch")),
        str(get_nested(config, "model.encoder.use_scale2")),
        str(get_nested(config, "model.slot_iterations")),
        str(get_nested(config, "model.residual_slot_connection")),
        str(get_nested(config, "loss.class_weight_power")),
        str(get_nested(config, "loss.label_smoothing")),
        str(get_nested(config, "loss.lambda_supcon")),
        str(get_nested(config, "loss.lambda_div")),
        str(get_nested(config, "loss.ce_warmup_epochs")),
    ]
    return " | ".join(fields)


def main() -> None:
    header = [
        "config_name",
        "amp",
        "multi_gpu",
        "use_compile",
        "torch_compile",
        "ddp",
        "find_unused",
        "ddp_compile",
        "compile_order",
        "data_bs",
        "fixed_bs",
        "ddp_chunk",
        "drop_incomplete",
        "carry_leftovers",
        "ddp_drop_last",
        "global",
        "scale2",
        "slots",
        "residual",
        "cw_power",
        "smooth",
        "supcon",
        "div",
        "ce_warmup",
    ]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for name, expected in CONFIGS.items():
        config = load_config_tree(expected["path"])
        config_name = get_nested(config, "run.config_name")
        if config_name != name:
            raise AssertionError(f"run.config_name expected {name!r}, got {config_name!r}")
        for key, value in {**COMMON_EXPECTED, **{k: v for k, v in expected.items() if k != "path"}}.items():
            assert_equal(config, key, value)
        loss_cfg = config.get("loss", {}) or {}
        data_cfg = config.get("data", {}) or {}
        for key in FORBIDDEN_LOSS_KEYS:
            if key in loss_cfg:
                raise AssertionError(f"{name}: forbidden loss key present: {key}")
        for key in FORBIDDEN_DATA_KEYS:
            if key in data_cfg:
                raise AssertionError(f"{name}: forbidden data key present: {key}")
        print(row(name, config))
    print(f"runtime_parity_config_check=OK count={len(CONFIGS)}")


if __name__ == "__main__":
    main()
