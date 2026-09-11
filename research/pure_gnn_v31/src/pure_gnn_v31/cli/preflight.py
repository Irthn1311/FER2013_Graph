"""Fail-closed synthetic technical preflight for Pure-GNN v3.1."""

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from pure_gnn_v31.coarse_conditions import CoarseBlock
from pure_gnn_v31.contracts import (
    audit_forbidden_layers,
    audit_forbidden_source,
    audit_no_absolute_node_coordinates,
    audit_sparse_local_source,
    count_parameters,
    verify_gate_initialization,
    verify_parameter_budget,
)
from pure_gnn_v31.diagnostics import (
    collect_graph_block_diagnostics,
    coarsening_shift_probe,
    compute_boundary_diagnostics,
    compute_boundary_gradient_diagnostics,
)
from pure_gnn_v31.graph_index import build_complete_coarse_graph, build_grid_8neighbor_graph
from pure_gnn_v31.local_relation import LocalAdaptiveRelationBlock
from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.technical_checks import (
    g05_g1_initial_equivalence_report,
    permutation_equivariance_report,
    reference_equivalence_report,
)


REFERENCE_TOLERANCE = 1e-5


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, tf.Tensor):
        array = value.numpy()
        return float(array) if array.ndim == 0 else array.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _finish_from_stage1(model: PureGNNv31, features: tf.Tensor) -> tf.Tensor:
    """Continue the unchanged production layers from the Stage-1 boundary."""
    h = model.coarsen1(features)
    for block in model.stage2_blocks:
        h = block(h, graph=model.graph_s2, training=False)
    h = model.coarsen2(h)
    for block in model.stage3_blocks:
        h = block(h, graph=model.graph_s3, training=False)
    h = model.coarsen3(h)
    for block in model.stage4_blocks:
        h = block(h, coarse_graph=model.graph_coarse, training=False)
    h = model.readout_norm(h)
    pooled = tf.concat([tf.reduce_mean(h, axis=1), tf.reduce_max(h, axis=1)], axis=-1)
    return model.classifier(model.head_dense1(pooled))


def _stage1_features(model: PureGNNv31, images: tf.Tensor) -> tf.Tensor:
    h = model.input_proj(tf.reshape(tf.cast(images, tf.float32), [-1, 2304, 1]))
    for block in model.stage1_blocks:
        h = block(h, graph=model.graph_s1, training=False)
    return h


def run_technical_preflight(
    output_path: Optional[str] = None,
    synthetic_overfit_steps: int = 15,
    technical_seed: int = 42,
) -> dict:
    """Execute only synthetic implementation checks; never load FER data."""
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(technical_seed)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tensorflow_version": tf.__version__,
        "devices": [device.name for device in tf.config.list_physical_devices()],
        "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "dtype_policy": tf.keras.mixed_precision.global_policy().name,
        "synthetic_only": True,
        "technical_seed": technical_seed,
        "technical_seed_scope": "synthetic_only",
        "scientific_training_performed": False,
        "validation_or_test_access": False,
        "checks": {},
        "status": "FAIL",
    }

    def persist() -> None:
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")

    def check(name: str, operation: Callable[[], object]):
        try:
            evidence = operation()
            report["checks"][name] = {"status": "PASS", "evidence": _jsonable(evidence)}
            persist()
            return evidence
        except Exception as exc:
            report["checks"][name] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
            persist()
            raise

    model = PureGNNv31(condition="G1")
    dummy = tf.random.stateless_uniform([2, 48, 48, 1], seed=[11, 12])
    logits = model(dummy, training=False)
    check("model_shape", lambda: {"logits_shape": list(logits.shape)} if logits.shape == (2, 7) else (_ for _ in ()).throw(AssertionError(logits.shape)))

    source_root = Path(__file__).resolve().parents[1]
    def forbidden_audit():
        nested = audit_forbidden_layers(model)
        source = audit_forbidden_source(source_root)
        sparse = audit_sparse_local_source(source_root)
        coordinates = audit_no_absolute_node_coordinates(source_root)
        if nested or source or sparse or coordinates:
            raise AssertionError({"nested_layers": nested, "source_ast": source, "sparse_path": sparse, "absolute_node_coordinates": coordinates})
        return {
            "recursive_nested_layer_violations": nested,
            "executable_ast_violations": source,
            "sparse_local_path_violations": sparse,
            "absolute_node_coordinate_violations": coordinates,
            "scope": "nested Keras layers plus executable imports/calls; prose ignored",
        }
    check("forbidden_op_recursive_and_source_audit", forbidden_audit)

    def parameter_check():
        passed, info = verify_parameter_budget(model, max_share=0.005)
        if not passed:
            raise AssertionError(info)
        report["parameter_info"] = info
        return info
    parameter_info = check("parameter_budget", parameter_check)

    def gate_initialization_check():
        info = verify_gate_initialization(model)
        if abs(info["local_gate_mean_at_init"] - 1.0) > REFERENCE_TOLERANCE:
            raise AssertionError(info)
        if abs(info["coarse_effective_neighbor_count_at_init"] - 35.0) > 0.1:
            raise AssertionError(info)
        return info
    check("local_and_coarse_gate_initialization", gate_initialization_check)

    def reference_check():
        info = reference_equivalence_report()
        for key, error in info.items():
            if error > REFERENCE_TOLERANCE:
                raise AssertionError(f"{key}={error} > {REFERENCE_TOLERANCE}")
        return info
    check("dense_reference_equivalence", reference_check)

    def permutation_check():
        info = permutation_equivariance_report()
        for key, error in info.items():
            if error > REFERENCE_TOLERANCE:
                raise AssertionError(f"{key}={error} > {REFERENCE_TOLERANCE}")
        return info
    check("permutation_equivariance", permutation_check)

    def functional_match_check():
        info = g05_g1_initial_equivalence_report()
        if info["max_abs_error"] > REFERENCE_TOLERANCE:
            raise AssertionError(info)
        return info
    check("g05_g1_functional_matching", functional_match_check)

    def mask_and_weight_checks():
        local_graph = build_grid_8neighbor_graph(4, 4)
        masked = LocalAdaptiveRelationBlock(4, gate_hidden_dim=3, mask_content=True)
        h_local = tf.random.stateless_normal([1, 16, 4], seed=[13, 14])
        _, local_diag = masked(h_local, graph=local_graph, return_diagnostics=True)
        content_max = float(tf.reduce_max(tf.abs(local_diag["gate_input_content"])).numpy())
        if content_max != 0.0:
            raise AssertionError(f"G2 gate content is not exact zero: {content_max}")

        graph = build_complete_coarse_graph(6)
        h = tf.random.stateless_normal([2, 36, 4], seed=[15, 16])
        g05 = CoarseBlock(4, condition="G0.5", gate_hidden_dim=2)
        _, g05_diag = g05(h, coarse_graph=graph, return_diagnostics=True)
        expected = 1.0 / 35.0
        weight_error = float(tf.reduce_max(tf.abs(g05_diag["pair_weights"] - expected)).numpy())
        if weight_error != 0.0:
            raise AssertionError(f"G0.5 pair weight error: {weight_error}")
        g1 = CoarseBlock(4, condition="G1", gate_hidden_dim=2)
        _, g1_diag = g1(h, coarse_graph=graph, return_diagnostics=True)
        sum_error = float(tf.reduce_max(tf.abs(g1_diag["receiver_weight_sums"] - 1.0)).numpy())
        if sum_error > REFERENCE_TOLERANCE:
            raise AssertionError(f"G1 receiver sum error: {sum_error}")
        return {
            "g2_content_gate_max_abs": content_max,
            "g05_exact_uniform_weight": expected,
            "g05_weight_max_abs_error": weight_error,
            "g1_receiver_sum_max_abs_error": sum_error,
        }
    check("g2_mask_and_receiver_weights", mask_and_weight_checks)

    def g3_check():
        control = PureGNNv31(condition="G3")
        _ = control(dummy[:1], training=False)
        flags = [control.coarsen1.use_simple_mean, control.coarsen2.use_simple_mean, control.coarsen3.use_simple_mean]
        if flags != [True, True, True]:
            raise AssertionError(flags)
        return {"all_three_coarseners_use_simple_mean": True}
    check("g3_coarsening_switch", g3_check)

    def block_instrumentation():
        _, diagnostics = model(dummy[:1], training=False, return_diagnostics=True)
        traced = collect_graph_block_diagnostics(model, dummy[:1])
        prefixes = [
            *(f"stage1_block{i}" for i in range(2)),
            *(f"stage2_block{i}" for i in range(2)),
            *(f"stage3_block{i}" for i in range(2)),
            *(f"coarse_block{i}" for i in range(2)),
        ]
        missing = []
        for prefix in prefixes:
            for metric in ("feature_variance", "feature_norm", "effective_rank"):
                if f"{prefix}_{metric}" not in diagnostics:
                    missing.append(f"{prefix}_{metric}")
        if missing:
            raise AssertionError(missing)
        missing_gradients = [name for name in prefixes if "gradient_norm" not in traced[name]]
        if missing_gradients:
            raise AssertionError(missing_gradients)
        return {
            "graph_blocks_instrumented": len(prefixes),
            "missing": missing,
            "missing_gradient_norms": missing_gradients,
            "per_block": traced,
        }
    check("all_block_diagnostic_instrumentation", block_instrumentation)

    def batch_independence():
        joint = model(dummy, training=False)
        separate = tf.concat([model(dummy[index:index + 1], training=False) for index in range(2)], axis=0)
        error = float(tf.reduce_max(tf.abs(joint - separate)).numpy())
        if error > REFERENCE_TOLERANCE:
            raise AssertionError(error)
        return {"max_abs_error": error}
    check("batch_independence", batch_independence)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    labels = tf.constant([3, 0], tf.int32)
    before = [variable.numpy().copy() for variable in model.trainable_variables]
    with tf.GradientTape() as tape:
        train_logits = model(dummy, training=True)
        train_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=train_logits)
        )
    gradients = tape.gradient(train_loss, model.trainable_variables)
    def gradient_check():
        missing = [variable.path for gradient, variable in zip(gradients, model.trainable_variables) if gradient is None]
        nonfinite = [
            variable.path for gradient, variable in zip(gradients, model.trainable_variables)
            if gradient is not None and not bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
        ]
        if missing or nonfinite:
            raise AssertionError({"missing": missing, "nonfinite": nonfinite})
        return {"gradient_count": len(gradients), "all_non_none": True, "all_finite": True}
    check("finite_gradients", gradient_check)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    changed = sum(not np.array_equal(old, new.numpy()) for old, new in zip(before, model.trainable_variables))
    check("optimizer_variable_update", lambda: {"changed_trainable_variables": changed} if changed else (_ for _ in ()).throw(AssertionError("No trainable variable changed")))

    overfit_images = tf.random.stateless_uniform([4, 48, 48, 1], seed=[17, 18])
    overfit_labels = tf.constant([0, 1, 2, 3], tf.int32)
    def fixed_loss():
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=overfit_labels, logits=model(overfit_images, training=False)
        ))
    initial_overfit_loss = float(fixed_loss().numpy())
    for _ in range(synthetic_overfit_steps):
        with tf.GradientTape() as tape:
            step_loss = fixed_loss()
        step_gradients = tape.gradient(step_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(step_gradients, model.trainable_variables))
    final_overfit_loss = float(fixed_loss().numpy())
    check("tiny_fixed_batch_overfit", lambda: {
        "steps": synthetic_overfit_steps,
        "initial_overfit_loss": initial_overfit_loss,
        "final_overfit_loss": final_overfit_loss,
        "same_fixed_batch": True,
    } if final_overfit_loss < initial_overfit_loss else (_ for _ in ()).throw(AssertionError(
        f"Fixed-batch loss did not decrease: {initial_overfit_loss} -> {final_overfit_loss}"
    )))

    stage1 = _stage1_features(model, dummy[:1])
    with tf.GradientTape() as boundary_tape:
        boundary_tape.watch(stage1)
        boundary_logits = _finish_from_stage1(model, stage1)
        boundary_objective = tf.reduce_sum(tf.square(boundary_logits))
    stage1_gradient = boundary_tape.gradient(boundary_objective, stage1)
    boundary = compute_boundary_diagnostics(stage1)
    boundary.update(compute_boundary_gradient_diagnostics(stage1_gradient))
    check("stage1_boundary_diagnostics", lambda: boundary)

    coarsening_features = tf.random.stateless_normal([1, 2304, 32], seed=[19, 20])
    shift = check("coarsening_shift_probe", lambda: coarsening_shift_probe(model.coarsen1, coarsening_features))
    report["coarsening_shift_diagnostics"] = shift

    report["status"] = "PASS"
    persist()
    return _jsonable(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pure-GNN v3.1 technical preflight")
    parser.add_argument("--output")
    parser.add_argument("--synthetic-overfit-steps", type=int, default=15)
    parser.add_argument("--technical-seed", type=int, default=42)
    arguments = parser.parse_args()
    run_technical_preflight(
        arguments.output,
        arguments.synthetic_overfit_steps,
        arguments.technical_seed,
    )
