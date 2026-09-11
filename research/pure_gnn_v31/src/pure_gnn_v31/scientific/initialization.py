"""Weight synchronization and initialization pairing utilities for G0, G0.5, and G1."""

from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31


def synchronize_shared_parameters(source_model: PureGNNv31, target_model: PureGNNv31) -> int:
    """Synchronizes all shared parameters between source and target PureGNN models.
    
    Specifically used between G0.5 and G1:
    - Input projection: shared
    - Stage 1, 2, 3 local blocks: shared
    - Coarsen 1, 2, 3 layers: shared
    - Stage 4 coarse blocks shared parameters:
        * norm1, norm2, ffn_dense1, ffn_dense2, v_proj
    - Readout and classifier head: shared
    - G1 coarse gate parameters (coarse_gate_*) exist only in G1 and remain at their zero-init.
    
    Returns the number of synchronized variables.
    """
    target_vars = {
        f"{getattr(v, 'path', '')}::{getattr(v, 'name', '')}": v
        for v in target_model.trainable_variables
    }

    synced_count = 0
    for src_v in source_model.trainable_variables:
        key = f"{getattr(src_v, 'path', '')}::{getattr(src_v, 'name', '')}"
        if key in target_vars:
            tgt_v = target_vars[key]
            if tgt_v.shape == src_v.shape:
                tgt_v.assign(src_v)
                synced_count += 1

    return synced_count


def verify_initialization_equivalence(
    model_g05: PureGNNv31,
    model_g1: PureGNNv31,
    sample_input: tf.Tensor,
    tolerance: float = 1e-5,
) -> Tuple[bool, float]:
    """Tests that after parameter synchronization, G0.5 and G1 produce identical logits within tolerance.
    
    Because G1 coarse gate is zero-initialized (giving uniform 1/35 weights),
    its mathematical computation on sample_input is identical to G0.5.
    """
    logits_g05 = model_g05(sample_input, training=False)
    logits_g1 = model_g1(sample_input, training=False)

    diff = tf.reduce_max(tf.abs(logits_g05 - logits_g1))
    diff_val = float(diff.numpy())
    passed = diff_val <= tolerance
    return passed, diff_val
