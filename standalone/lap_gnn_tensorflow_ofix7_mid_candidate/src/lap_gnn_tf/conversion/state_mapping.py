"""Complete explicit PyTorch NPZ to TensorFlow variable mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lap_gnn_tf.model.lap_gnn import LapGNN


def mapping_records(model: LapGNN) -> list[dict]:
    records = []
    for binding in model.state_bindings():
        records.append({
            "pytorch_key": binding.source_key,
            "tensorflow_variable": f"lap_gnn_tf/{binding.source_key.replace('.', '/')}",
            "keras_variable_path": binding.variable.path,
            "tensorflow_shape": list(binding.variable.shape),
            "transformation": binding.transform,
            "dtype": str(binding.variable.dtype),
        })
    return records


def export_mapping(model: LapGNN, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping_records(model), indent=2), encoding="utf-8")
    return path


def load_pytorch_npz(model: LapGNN, state_path: str | Path, strict: bool = True) -> dict:
    bindings = model.mapped_trainable_variables()
    with np.load(state_path, allow_pickle=False) as state:
        source_keys = set(state.files)
        destination_keys = set(bindings)
        missing = sorted(destination_keys - source_keys)
        extra = sorted(source_keys - destination_keys)
        shape_mismatches = []
        dtype_mismatches = []
        assigned = []
        for key in sorted(source_keys & destination_keys):
            array = np.asarray(state[key])
            binding = bindings[key]
            transformed = array.T if binding.transform == "transpose" else array
            if tuple(transformed.shape) != tuple(binding.variable.shape):
                shape_mismatches.append({
                    "key": key,
                    "source": list(array.shape),
                    "transformed": list(transformed.shape),
                    "destination": list(binding.variable.shape),
                })
                continue
            if array.dtype != np.float32:
                dtype_mismatches.append({"key": key, "source": str(array.dtype), "expected": "float32"})
                continue
            binding.variable.assign(transformed)
            assigned.append(key)
    duplicate_destinations = []
    variable_ids = [id(binding.variable) for binding in bindings.values()]
    for variable_id in sorted(set(variable_ids)):
        if variable_ids.count(variable_id) > 1:
            duplicate_destinations.append(str(variable_id))
    result = {
        "source_tensors": len(source_keys),
        "destination_variables": len(destination_keys),
        "assigned": len(assigned),
        "missing_tensors": missing,
        "extra_tensors": extra,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "duplicate_destinations": duplicate_destinations,
        "complete": not any([missing, extra, shape_mismatches, dtype_mismatches, duplicate_destinations]),
    }
    if strict and not result["complete"]:
        raise ValueError(f"Incomplete PyTorch/TensorFlow mapping: {result}")
    return result
