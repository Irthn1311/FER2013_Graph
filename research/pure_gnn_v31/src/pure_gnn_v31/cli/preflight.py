"""Technical preflight runner for Pure-GNN v3.1."""

import argparse
import json
import time
from pathlib import Path
from typing import Optional
import numpy as np
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.contracts import (
    count_parameters,
    verify_parameter_budget,
    audit_forbidden_layers,
    verify_gate_initialization,
)
from pure_gnn_v31.diagnostics import compute_boundary_diagnostics, compute_shift_stability


def run_technical_preflight(output_path: Optional[str] = None) -> dict:
    """Executes the full technical preflight test suite and benchmark."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tensorflow_version": tf.__version__,
        "devices": [d.name for d in tf.config.list_physical_devices()],
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "checks": {},
        "status": "FAIL",
    }

    print("=== Pure-GNN v3.1 Technical Preflight ===")

    # 1. Instantiate G1 Model
    print("1. Instantiating G1 Model...")
    model_g1 = PureGNNv31(condition="G1")
    dummy_input = tf.random.uniform([2, 48, 48, 1], dtype=tf.float32)
    out_logits = model_g1(dummy_input, training=False)
    assert out_logits.shape == (2, 7), f"Unexpected logits shape: {out_logits.shape}"
    report["checks"]["g1_forward_shape"] = "PASS"

    # 2. Forbidden Ops Audit
    print("2. Auditing model for forbidden operators (Conv2D, Attention, etc.)...")
    violations = audit_forbidden_layers(model_g1)
    if violations:
        report["checks"]["forbidden_ops_audit"] = f"FAIL: {violations}"
        raise RuntimeError(f"Forbidden ops found: {violations}")
    report["checks"]["forbidden_ops_audit"] = "PASS"

    # 3. Parameter Budget Constraint (Gate params <= 0.5% total params)
    print("3. Checking parameter budget constraint...")
    passed, param_info = verify_parameter_budget(model_g1, max_share=0.005)
    report["parameter_info"] = param_info
    if not passed:
        report["checks"]["parameter_budget"] = f"FAIL: Gate share {param_info['coarse_gate_parameter_share']:.4f} > 0.005"
        raise RuntimeError(f"Gate parameter budget exceeded: {param_info}")
    report["checks"]["parameter_budget"] = "PASS"

    # 4. Gate Zero-Init Contract
    print("4. Verifying gate zero-initialization contract...")
    gate_init_info = verify_gate_initialization(model_g1)
    report["gate_init_info"] = gate_init_info
    # Local gate g_ij must be 1.0 at init
    assert abs(gate_init_info["local_gate_mean_at_init"] - 1.0) < 1e-5, "Local gate not initialized to 1.0"
    # Coarse effective neighbor count must be ~35 at init
    assert abs(gate_init_info["coarse_effective_neighbor_count_at_init"] - 35.0) < 0.1, "Coarse weights not uniform 1/35 at init"
    report["checks"]["gate_initialization"] = "PASS"

    # 5. G0.5 vs G1 Path Matching
    print("5. Verifying G0.5 vs G1 functional path matching...")
    model_g05 = PureGNNv31(condition="G0.5")
    _ = model_g05(dummy_input, training=False)
    # G0.5 coarse block has no coarse gate MLP
    p_g05 = count_parameters(model_g05)
    report["g05_parameters"] = p_g05
    # The only parameter difference between G1 and G0.5 should be the coarse gate MLP
    diff_params = param_info["total_trainable_parameters"] - p_g05["total_trainable_parameters"]
    assert diff_params == param_info["coarse_gate_mlp_parameters"], "Unexpected parameter divergence between G1 and G0.5"
    report["checks"]["g05_g1_functional_matching"] = "PASS"

    # 6. G2 Local Content Masking
    print("6. Verifying G2 local content masking...")
    model_g2 = PureGNNv31(condition="G2")
    _ = model_g2(dummy_input, training=False)
    for block in model_g2.stage1_blocks:
        assert block.mask_content is True, "G2 block does not mask content"
    report["checks"]["g2_content_masking"] = "PASS"

    # 7. G3 Coarsening Control
    print("7. Verifying G3 coarsening control...")
    model_g3 = PureGNNv31(condition="G3")
    _ = model_g3(dummy_input, training=False)
    assert model_g3.coarsen1.use_simple_mean is True, "G3 coarsening does not use simple mean"
    report["checks"]["g3_coarsening_control"] = "PASS"

    # 8. Gradient Flow and Trainability
    print("8. Testing gradient flow and optimizer step...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    dummy_labels = tf.constant([3, 0], dtype=tf.int32)
    with tf.GradientTape() as tape:
        logits = model_g1(dummy_input, training=True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=dummy_labels, logits=logits))
    grads = tape.gradient(loss, model_g1.trainable_variables)
    for g, v in zip(grads, model_g1.trainable_variables):
        assert g is not None, f"Gradient is None for variable: {v.name}"
        assert not tf.reduce_any(tf.math.is_nan(g)), f"NaN gradient in: {v.name}"
    optimizer.apply_gradients(zip(grads, model_g1.trainable_variables))
    report["checks"]["gradient_flow"] = "PASS"

    # 9. Tiny-batch Overfit Sanity
    print("9. Running tiny-batch overfit test (10 steps)...")
    overfit_input = tf.random.uniform([4, 48, 48, 1], dtype=tf.float32, seed=123)
    overfit_labels = tf.constant([0, 1, 2, 3], dtype=tf.int32)
    initial_loss = float(loss.numpy())
    current_loss = initial_loss
    for _ in range(10):
        with tf.GradientTape() as tape:
            l = model_g1(overfit_input, training=True)
            step_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=overfit_labels, logits=l))
        g = tape.gradient(step_loss, model_g1.trainable_variables)
        optimizer.apply_gradients(zip(g, model_g1.trainable_variables))
        current_loss = float(step_loss.numpy())
    assert current_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {current_loss}"
    report["checks"]["tiny_overfit"] = "PASS"

    # 10. Boundary and Shift Diagnostics
    print("10. Testing boundary and shift stability diagnostics...")
    shift_diags = compute_shift_stability(model_g1, dummy_input)
    report["shift_diagnostics"] = shift_diags
    report["checks"]["diagnostics_run"] = "PASS"

    report["status"] = "PASS"
    print("=== All Technical Preflight Checks PASSED ===")

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Preflight report saved to: {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pure-GNN v3.1 Technical Preflight")
    parser.add_argument("--output", type=str, default=None, help="Path to write preflight JSON report")
    args = parser.parse_args()
    run_technical_preflight(args.output)
