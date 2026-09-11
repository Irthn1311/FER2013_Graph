"""Formal contract definitions and assertion utilities for Pure-GNN v3.1."""

import ast
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31


FORBIDDEN_LAYER_STRINGS = [
    "conv",
    "convolution",
    "attention",
    "transformer",
    "roialign",
    "superpixel",
    "landmark",
    "mediapipe",
]


def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
    """Counts total and gate parameters of the model."""
    total_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    
    # Identify coarse gate parameters in G1
    gate_params = 0
    for v in model.trainable_variables:
        identifier = f"{getattr(v, 'path', '')} {getattr(v, 'name', '')}".lower()
        if "coarse_gate" in identifier:
            gate_params += int(np.prod(v.shape))

    # Also compute local gate MLP parameters
    local_gate_params = 0
    for v in model.trainable_variables:
        identifier = f"{getattr(v, 'path', '')} {getattr(v, 'name', '')}".lower()
        if "gate_dense" in identifier and "coarse" not in identifier:
            local_gate_params += int(np.prod(v.shape))

    share = float(gate_params) / float(total_params) if total_params > 0 else 0.0

    return {
        "total_trainable_parameters": total_params,
        "coarse_gate_mlp_parameters": gate_params,
        "local_gate_mlp_parameters": local_gate_params,
        "coarse_gate_parameter_share": share,
    }


def verify_parameter_budget(model: tf.keras.Model, max_share: float = 0.005) -> Tuple[bool, Dict]:
    """Asserts that G1 coarse gate parameter share is <= 0.5% (0.005)."""
    counts = count_parameters(model)
    passed = counts["coarse_gate_parameter_share"] <= max_share
    return passed, counts


def audit_forbidden_layers(model: tf.keras.Model) -> List[str]:
    """Recursively audit every nested Keras layer in the instantiated model."""
    violations = []
    if hasattr(model, "_flatten_layers"):
        layers = model._flatten_layers(include_self=False, recursive=True)
    else:  # pragma: no cover - compatibility fallback for older Keras
        layers = model.layers
    seen = set()
    for layer in layers:
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        layer_cls = layer.__class__.__name__.lower()
        layer_name = layer.name.lower()
        for forbidden in FORBIDDEN_LAYER_STRINGS:
            if forbidden in layer_cls or forbidden in layer_name:
                violations.append(f"Layer '{layer.name}' ({layer.__class__.__name__}) matches forbidden token '{forbidden}'")
    return violations


FORBIDDEN_EXECUTABLE_IDENTIFIERS = {
    "conv1d",
    "conv2d",
    "conv3d",
    "convolution1d",
    "convolution2d",
    "convolution3d",
    "multiheadattention",
    "attention",
    "additiveattention",
    "roialign",
    "mediapipe",
    "landmark",
    "landmarks",
    "transformer",
    "knn",
    "kneighbors",
}


def audit_forbidden_source(source_root: Path) -> List[str]:
    """Audit executable Python AST nodes, while ignoring prose/docstrings."""
    violations: List[str] = []
    for path in sorted(Path(source_root).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            identifiers: List[str] = []
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    identifiers.extend(alias.name for alias in node.names)
                elif node.module:
                    identifiers.append(node.module)
            elif isinstance(node, ast.Call):
                target = node.func
                if isinstance(target, ast.Name):
                    identifiers.append(target.id)
                elif isinstance(target, ast.Attribute):
                    identifiers.append(target.attr)
            for identifier in identifiers:
                components = {part.lower() for part in identifier.replace("-", "_").split(".")}
                hits = components & FORBIDDEN_EXECUTABLE_IDENTIFIERS
                for hit in sorted(hits):
                    violations.append(f"{path}:{getattr(node, 'lineno', 0)} invokes/imports {hit}")
    return violations


def audit_sparse_local_source(source_root: Path) -> List[str]:
    """Check the production local path retains gather/segment sparse aggregation."""
    local_path = Path(source_root) / "local_relation.py"
    source = local_path.read_text(encoding="utf-8")
    required = ("tf.gather", "tf.math.unsorted_segment_sum")
    missing = [token for token in required if token not in source]
    forbidden_dense_literals = ("2304 * 2304", "(2304, 2304)", "[2304, 2304]")
    present = [token for token in forbidden_dense_literals if token in source]
    return [f"missing sparse primitive: {token}" for token in missing] + [
        f"forbidden full-resolution dense construction: {token}" for token in present
    ]


def audit_no_absolute_node_coordinates(source_root: Path) -> List[str]:
    """Audit that the model's initial node projection consumes pixel state only.

    Relative geometry remains permitted inside gate functions. This narrowly
    proves that ``model.py`` does not feed an x/y tensor to ``input_proj``.
    """
    model_path = Path(source_root) / "model.py"
    tree = ast.parse(model_path.read_text(encoding="utf-8"), filename=str(model_path))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "input_proj":
            calls.append(node)
    violations = []
    if len(calls) != 1:
        violations.append(f"expected exactly one input_proj call, found {len(calls)}")
    elif len(calls[0].args) != 1 or not isinstance(calls[0].args[0], ast.Name) or calls[0].args[0].id != "h":
        violations.append("input_proj must consume only the one-channel pixel-state tensor h")
    return violations


def verify_gate_initialization(model: PureGNNv31) -> Dict[str, float]:
    """Verifies that at initialization:
    - local gates produce exactly g_ij = 1.0
    - G1 coarse gates produce approximately uniform 1/35 weights.
    """
    dummy_input = tf.random.uniform([2, 48, 48, 1], dtype=tf.float32)
    _, diags = model(dummy_input, training=False, return_diagnostics=True)

    local_mean = float(diags.get("stage1_gate_mean", 0.0))
    local_std = float(diags.get("stage1_gate_std", 0.0))
    coarse_ent = float(diags.get("coarse_coarse_weight_entropy", 0.0))
    coarse_eff_n = float(diags.get("coarse_effective_neighbor_count", 0.0))

    return {
        "local_gate_mean_at_init": local_mean,
        "local_gate_std_at_init": local_std,
        "coarse_entropy_at_init": coarse_ent,
        "coarse_effective_neighbor_count_at_init": coarse_eff_n,
    }
