"""Weight synchronization and initialization pairing utilities for G0, G0.5, and G1."""

from typing import Dict, List, Set, Tuple
import numpy as np
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31


def get_variable_map(model: tf.keras.Model) -> Dict[str, tf.Variable]:
    """Builds a mapping from canonical variable path to tf.Variable."""
    return {
        f"{getattr(v, 'path', '')}::{getattr(v, 'name', '')}": v
        for v in model.trainable_variables
    }


def audit_and_synchronize_shared_parameters(
    source_model: PureGNNv31,
    target_model: PureGNNv31,
) -> Dict[str, object]:
    """Strictly audits and synchronizes shared parameters between G0.5 and G1.

    Verifies:
    - Identifies exact trainable variable paths in source (G0.5)
    - Identifies target (G1) shared non-gate paths
    - Asserts shared set equality
    - Copies every shared value
    - Asserts every copied tensor equals exactly (max copy error == 0.0)
    - Asserts G1 coarse gate final projections remain zero-initialized
    """
    src_map = get_variable_map(source_model)
    tgt_map = get_variable_map(target_model)

    # Coarse gate variables in G1
    gate_vars = {k: v for k, v in tgt_map.items() if "coarse_gate" in k.lower()}
    # Target shared non-gate variables
    tgt_shared = {k: v for k, v in tgt_map.items() if "coarse_gate" not in k.lower()}

    # Source should have exact same shared set as target shared
    src_keys = set(src_map.keys())
    tgt_shared_keys = set(tgt_shared.keys())

    if src_keys != tgt_shared_keys:
        missing_in_tgt = src_keys - tgt_shared_keys
        missing_in_src = tgt_shared_keys - src_keys
        raise AssertionError(
            f"Shared parameter mismatch between source and target! "
            f"Missing in target: {missing_in_tgt}, Missing in source: {missing_in_src}"
        )

    # Perform exact copy and assert zero copy error
    max_copy_error = 0.0
    shared_param_count = 0
    copied_tensors = 0

    for key in sorted(src_keys):
        src_v = src_map[key]
        tgt_v = tgt_shared[key]

        if src_v.shape != tgt_v.shape:
            raise AssertionError(f"Shape mismatch on {key}: {src_v.shape} != {tgt_v.shape}")

        tgt_v.assign(src_v)
        err = float(tf.reduce_max(tf.abs(src_v - tgt_v)).numpy())
        if err > max_copy_error:
            max_copy_error = err
        shared_param_count += int(np.prod(src_v.shape))
        copied_tensors += 1

    if max_copy_error > 0.0:
        raise AssertionError(f"Non-zero copy error during synchronization: {max_copy_error}")

    # Assert G1 coarse gate final projection remains zero initialized
    for k, v in gate_vars.items():
        if "coarse_gate_dense2" in k.lower():
            gate2_err = float(tf.reduce_max(tf.abs(v)).numpy())
            if gate2_err > 0.0:
                raise AssertionError(f"G1 coarse gate dense2 is not zero-initialized: {gate2_err} in {k}")

    return {
        "source_condition": getattr(source_model, "condition", "UNKNOWN"),
        "target_condition": getattr(target_model, "condition", "UNKNOWN"),
        "exact_shared_tensor_count": copied_tensors,
        "shared_parameter_count": shared_param_count,
        "unmatched_g1_gate_variables": len(gate_vars),
        "max_parameter_copy_error": max_copy_error,
    }


def audit_and_synchronize_non_coarse_parameters(
    source_model: PureGNNv31,
    target_model: PureGNNv31,
) -> Dict[str, object]:
    """Strictly audits and synchronizes non-coarse parameters between G0 and G0.5 (or G1).

    Pairs all structurally equivalent NON-COARSE shared architecture parameters
    where canonical variable paths match:
      - input projection
      - stage 1, 2, 3 local relation blocks
      - coarsening 1, 2, 3 layers
      - readout norm, head dense, classifier

    Does NOT force semantically different coarse blocks (stage 4) to share incompatible weights.
    """
    src_map = get_variable_map(source_model)
    tgt_map = get_variable_map(target_model)

    # Filter for non-coarse variables (excluding stage4_block)
    src_non_coarse = {k: v for k, v in src_map.items() if "stage4_block" not in k.lower()}
    tgt_non_coarse = {k: v for k, v in tgt_map.items() if "stage4_block" not in k.lower()}

    src_keys = set(src_non_coarse.keys())
    tgt_keys = set(tgt_non_coarse.keys())

    if src_keys != tgt_keys:
        missing_in_tgt = src_keys - tgt_keys
        missing_in_src = tgt_keys - src_keys
        raise AssertionError(
            f"Non-coarse parameter mismatch! Missing in target: {missing_in_tgt}, Missing in source: {missing_in_src}"
        )

    max_copy_error = 0.0
    shared_param_count = 0
    copied_tensors = 0

    for key in sorted(src_keys):
        src_v = src_non_coarse[key]
        tgt_v = tgt_non_coarse[key]

        if src_v.shape != tgt_v.shape:
            raise AssertionError(f"Shape mismatch on {key}: {src_v.shape} != {tgt_v.shape}")

        tgt_v.assign(src_v)
        err = float(tf.reduce_max(tf.abs(src_v - tgt_v)).numpy())
        if err > max_copy_error:
            max_copy_error = err
        shared_param_count += int(np.prod(src_v.shape))
        copied_tensors += 1

    if max_copy_error > 0.0:
        raise AssertionError(f"Non-zero copy error during non-coarse synchronization: {max_copy_error}")

    return {
        "source_condition": getattr(source_model, "condition", "UNKNOWN"),
        "target_condition": getattr(target_model, "condition", "UNKNOWN"),
        "non_coarse_shared_tensor_count": copied_tensors,
        "non_coarse_shared_parameter_count": shared_param_count,
        "max_parameter_copy_error": max_copy_error,
    }


def synchronize_shared_parameters(source_model: PureGNNv31, target_model: PureGNNv31) -> int:
    """Wrapper that synchronizes shared parameters and returns tensor count."""
    audit = audit_and_synchronize_shared_parameters(source_model, target_model)
    return int(audit["exact_shared_tensor_count"])


def verify_initialization_equivalence(
    model_g05: PureGNNv31,
    model_g1: PureGNNv31,
    sample_input: tf.Tensor,
    tolerance: float = 1e-5,
) -> Tuple[bool, float]:
    """Tests that after parameter synchronization, G0.5 and G1 produce identical logits within tolerance."""
    logits_g05 = model_g05(sample_input, training=False)
    logits_g1 = model_g1(sample_input, training=False)

    diff = tf.reduce_max(tf.abs(logits_g05 - logits_g1))
    diff_val = float(diff.numpy())
    passed = diff_val <= tolerance
    return passed, diff_val
