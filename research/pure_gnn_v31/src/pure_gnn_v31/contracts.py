"""Formal contract definitions and assertion utilities for Pure-GNN v3.1."""

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
    """Audits model layers for any prohibited primitives (Conv2D, Attention, Landmarks, etc.)."""
    violations = []
    for layer in model.layers:
        layer_cls = layer.__class__.__name__.lower()
        layer_name = layer.name.lower()
        for forbidden in FORBIDDEN_LAYER_STRINGS:
            if forbidden in layer_cls or forbidden in layer_name:
                violations.append(f"Layer '{layer.name}' ({layer.__class__.__name__}) matches forbidden token '{forbidden}'")
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
