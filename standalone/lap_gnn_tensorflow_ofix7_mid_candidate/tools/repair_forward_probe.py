"""Bounded CPU float32 forward-parity localization for the repair task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from lap_gnn_tf.conversion import load_pytorch_npz
from lap_gnn_tf.graph.batch import load_golden_batch
from lap_gnn_tf.model import build_model


PART_ORDER = ["mouth", "eye", "brow", "nose_cheek", "global"]
GROUP_INDICES = [[5, 6, 7], [0, 1], [2, 3], [4, 8, 9]]


def torch_style_part_pool(
    h, part_soft, node_graph_index, valid_part_mask, num_graphs,
):
    """Diagnostic graph-wise reduction matching the PyTorch loop structure."""

    pooled = {}
    valid = {}
    for name, indices in zip(PART_ORDER[:4], GROUP_INDICES):
        def reduce_graph(graph_id):
            node_mask = tf.equal(node_graph_index, graph_id)
            h_graph = tf.boolean_mask(h, node_mask)
            part_graph = tf.boolean_mask(part_soft, node_mask)
            weights = tf.reduce_max(tf.gather(part_graph, indices, axis=1), axis=1)
            denominator = tf.maximum(
                tf.reduce_sum(weights), tf.cast(1e-6, weights.dtype),
            )
            input_valid = tf.reduce_sum(
                tf.gather(valid_part_mask[graph_id], indices),
            ) > 0.0
            group_valid = tf.logical_and(input_valid, denominator > 1e-5)
            value = tf.reduce_sum(h_graph * weights[:, None], axis=0) / denominator
            return tf.where(group_valid, value, tf.zeros_like(value)), group_valid

        values, flags = tf.map_fn(
            reduce_graph,
            tf.range(num_graphs, dtype=tf.int32),
            fn_output_signature=(
                tf.TensorSpec((96,), tf.float32),
                tf.TensorSpec((), tf.bool),
            ),
        )
        pooled[name] = values
        valid[name] = flags

    def global_graph(graph_id):
        return tf.reduce_mean(
            tf.boolean_mask(h, tf.equal(node_graph_index, graph_id)), axis=0,
        )

    pooled["global"] = tf.map_fn(
        global_graph,
        tf.range(num_graphs, dtype=tf.int32),
        fn_output_signature=tf.TensorSpec((96,), tf.float32),
    )
    valid["global"] = tf.ones((num_graphs,), dtype=tf.bool)
    return pooled, valid


def ordered_float32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, 0x80000000 - bits, bits)


def ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(ordered_float32(left) - ordered_float32(right))


def linear(method: str, x: tf.Tensor, kernel: tf.Tensor, bias: tf.Tensor) -> tf.Tensor:
    if method == "matmul_add":
        return tf.linalg.matmul(x, kernel) + bias
    if method == "raw_matmul_add":
        return tf.raw_ops.MatMul(a=x, b=kernel) + bias
    if method == "einsum_add":
        return tf.einsum("bi,io->bo", x, kernel) + bias
    if method == "tensordot_add":
        return tf.tensordot(x, kernel, axes=[[-1], [0]]) + bias
    if method == "matmul_bias_add":
        return tf.nn.bias_add(tf.linalg.matmul(x, kernel), bias)
    if method == "raw_bias_add":
        return tf.raw_ops.BiasAdd(
            value=tf.raw_ops.MatMul(a=x, b=kernel),
            bias=bias,
            data_format="NHWC",
        )
    raise ValueError(method)


def classifier(model, x: tf.Tensor, method: str) -> tf.Tensor:
    head = model.classifier
    hidden = linear(method, x, head.linear1.kernel, head.linear1.bias)
    hidden = head.norm(hidden)
    hidden = tf.nn.gelu(hidden, approximate=False)
    return linear(method, hidden, head.linear2.kernel, head.linear2.bias)


def delta_summary(actual: np.ndarray, expected: np.ndarray) -> dict:
    delta = np.asarray(actual, np.float32) - np.asarray(expected, np.float32)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(
            np.linalg.norm(delta.astype(np.float64))
            / max(np.linalg.norm(np.asarray(expected, np.float64)), 1e-12)
        ),
        "max_ulp": int(np.max(ulp_distance(actual, expected))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--pooling", choices=("segment", "torch_loop"), default="segment")
    args = parser.parse_args()

    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_jit(False)
    tf.keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    golden = args.package_root / "validation_assets" / "golden"
    batch = load_golden_batch(str(golden / "graph_batch.npz"))
    if args.pooling == "torch_loop":
        import lap_gnn_tf.model.lap_gnn as lap_gnn_module

        lap_gnn_module.part_pool = torch_style_part_pool
    with tf.device("/CPU:0"):
        model = build_model(batch)
        load_pytorch_npz(model, golden / "model_state.npz", strict=True)
        output = model(batch, training=False, collect_intermediates=True)

    expected_logits = np.load(golden / "logits.npy", allow_pickle=False)
    expected_probabilities = np.load(golden / "probabilities.npy", allow_pickle=False)
    expected_classifier_input = np.load(golden / "layer_outputs.npz")["classifier_input"]
    actual_logits = output["logits"].numpy()
    actual_probabilities = output["probabilities"].numpy()
    actual_classifier_input = output["intermediates"]["classifier_input"].numpy()
    abs_delta = np.abs(actual_logits - expected_logits)
    signed_delta = actual_logits - expected_logits
    ulps = ulp_distance(actual_logits, expected_logits)
    flat_order = np.argsort(abs_delta.ravel())[::-1][:20]
    top20 = []
    for flat_index in flat_order:
        sample_index, class_index = np.unravel_index(flat_index, abs_delta.shape)
        top20.append({
            "sample_index": int(sample_index),
            "class_index": int(class_index),
            "pytorch_logit": float(expected_logits[sample_index, class_index]),
            "tensorflow_logit": float(actual_logits[sample_index, class_index]),
            "absolute_difference": float(abs_delta[sample_index, class_index]),
            "signed_difference": float(signed_delta[sample_index, class_index]),
            "ulp_distance": int(ulps[sample_index, class_index]),
        })

    methods = [
        "matmul_add",
        "raw_matmul_add",
        "einsum_add",
        "tensordot_add",
        "matmul_bias_add",
        "raw_bias_add",
    ]
    candidates = {}
    candidate_arrays = {}
    for method in methods:
        eager_golden = classifier(
            model, tf.convert_to_tensor(expected_classifier_input), method,
        ).numpy()
        eager_tensorflow = classifier(
            model, tf.convert_to_tensor(actual_classifier_input), method,
        ).numpy()

        @tf.function(autograph=False)
        def compiled(value):
            return classifier(model, value, method)

        graph_golden = compiled(tf.convert_to_tensor(expected_classifier_input)).numpy()
        graph_tensorflow = compiled(tf.convert_to_tensor(actual_classifier_input)).numpy()
        candidates[method] = {
            "golden_input_eager": delta_summary(eager_golden, expected_logits),
            "tensorflow_input_eager": delta_summary(eager_tensorflow, expected_logits),
            "golden_input_tf_function": delta_summary(graph_golden, expected_logits),
            "tensorflow_input_tf_function": delta_summary(graph_tensorflow, expected_logits),
            "eager_tf_function_golden_input": delta_summary(graph_golden, eager_golden),
            "eager_tf_function_tensorflow_input": delta_summary(graph_tensorflow, eager_tensorflow),
        }
        candidate_arrays[f"{method}_golden_input"] = eager_golden
        candidate_arrays[f"{method}_tensorflow_input"] = eager_tensorflow

    with np.load(golden / "layer_outputs.npz", allow_pickle=False) as expected_layers:
        layers = {
            name: delta_summary(output["intermediates"][name].numpy(), expected_layers[name])
            for name in expected_layers.files
            if name in output["intermediates"]
        }

    readout = model.readout
    major_transformed = output["intermediates"]["micro_major_motif_transformed_tokens"]
    micro_transformed = output["intermediates"]["micro_motif_transformed_tokens"]
    z_major = tf.reduce_mean(major_transformed, axis=1)
    z_micro = tf.reduce_mean(micro_transformed, axis=1)
    micro_project_norm = readout.micro_norm(z_micro)
    micro_projected = readout.micro_project(micro_project_norm)
    gate_concat = tf.concat([z_major, z_micro], axis=1)
    gate_norm = readout.gate_norm(gate_concat)
    gate_linear = readout.gate_in(gate_norm)
    gate_gelu = tf.nn.gelu(gate_linear, approximate=False)
    gate_logits = readout.gate_out(gate_gelu)
    gate = tf.nn.sigmoid(gate_logits)
    z_support = z_major + gate * micro_projected
    residual = tf.reshape(
        tf.stack([output["part_embeddings"][name] for name in PART_ORDER], axis=1),
        (tf.shape(z_major)[0], 480),
    )
    fused = tf.concat([z_support, residual], axis=1)
    projection_norm = readout.projection_norm(fused)
    projection_linear = readout.projection_in(projection_norm)
    projection_gelu = tf.nn.gelu(projection_linear, approximate=False)
    projection_output = readout.projection_out(projection_gelu)

    major_tokens = output["intermediates"]["micro_major_motif_tokens"]
    micro_tokens = output["intermediates"]["micro_motif_tokens"]
    tokens = tf.concat([major_tokens, micro_tokens], axis=1) + readout.token_type_embedding
    cls = tf.tile(readout.cls_token, [tf.shape(tokens)[0], 1, 1])
    tokens_in = tf.concat([cls, tokens], axis=1)
    transformer = readout.transformer
    qkv = tf.linalg.matmul(tokens_in, transformer.self_attn.in_proj_kernel)
    qkv = qkv + transformer.self_attn.in_proj_bias
    q, k, v = tf.split(qkv, 3, axis=-1)
    batch_size = tf.shape(tokens_in)[0]
    token_count = tf.shape(tokens_in)[1]

    def split_heads(value):
        value = tf.reshape(
            value,
            (batch_size, token_count, transformer.self_attn.num_heads, transformer.self_attn.head_dim),
        )
        return tf.transpose(value, (0, 2, 1, 3))

    q_heads = split_heads(q)
    k_heads = split_heads(k)
    v_heads = split_heads(v)
    q_scaled = q_heads * tf.cast(transformer.self_attn.head_dim ** -0.5, q.dtype)
    attention_scores = tf.linalg.matmul(q_scaled, k_heads, transpose_b=True)
    attention_softmax = tf.nn.softmax(attention_scores, axis=-1)
    attention_weighted = tf.linalg.matmul(attention_softmax, v_heads)
    attention_merge = tf.reshape(
        tf.transpose(attention_weighted, (0, 2, 1, 3)),
        (batch_size, token_count, transformer.self_attn.dim),
    )
    attention_output = tf.linalg.matmul(attention_merge, transformer.self_attn.out_kernel)
    attention_output = attention_output + transformer.self_attn.out_bias
    transformer_norm1 = transformer.norm1(tokens_in + attention_output)
    transformer_ff1 = transformer.linear1(transformer_norm1)
    transformer_ff_gelu = tf.nn.gelu(transformer_ff1, approximate=False)
    transformer_ff2 = transformer.linear2(transformer_ff_gelu)
    transformer_norm2 = transformer.norm2(transformer_norm1 + transformer_ff2)
    trace_arrays = {
        "pre_readout_node_representation": output["node_embeddings"].numpy(),
        "classifier_input": actual_classifier_input,
        "logits": actual_logits,
        "major_tokens": major_tokens.numpy(),
        "major_transformed_tokens": major_transformed.numpy(),
        "micro_tokens": micro_tokens.numpy(),
        "micro_transformed_tokens": micro_transformed.numpy(),
        "micro_support_gate": gate.numpy(),
        "z_major": z_major.numpy(),
        "z_micro": z_micro.numpy(),
        "micro_project_norm": micro_project_norm.numpy(),
        "micro_projected": micro_projected.numpy(),
        "gate_concat": gate_concat.numpy(),
        "gate_norm": gate_norm.numpy(),
        "gate_linear": gate_linear.numpy(),
        "gate_gelu": gate_gelu.numpy(),
        "gate_logits": gate_logits.numpy(),
        "z_support": z_support.numpy(),
        "residual_concat": residual.numpy(),
        "fused_readout": fused.numpy(),
        "projection_norm": projection_norm.numpy(),
        "projection_linear": projection_linear.numpy(),
        "projection_gelu": projection_gelu.numpy(),
        "projection_output": projection_output.numpy(),
        "transformer_tokens_in": tokens_in.numpy(),
        "attention_q": q.numpy(),
        "attention_k": k.numpy(),
        "attention_v": v.numpy(),
        "attention_scores": attention_scores.numpy(),
        "attention_softmax": attention_softmax.numpy(),
        "attention_weighted_sum": attention_weighted.numpy(),
        "attention_output": attention_output.numpy(),
        "transformer_norm1": transformer_norm1.numpy(),
        "transformer_ff1": transformer_ff1.numpy(),
        "transformer_ff_gelu": transformer_ff_gelu.numpy(),
        "transformer_ff2": transformer_ff2.numpy(),
        "transformer_norm2": transformer_norm2.numpy(),
    }
    for name, value in output["part_embeddings"].items():
        trace_arrays[f"part_pool_{name}"] = value.numpy()

    result = {
        "tensorflow": tf.__version__,
        "one_dnn_env": os.environ.get("TF_ENABLE_ONEDNN_OPTS", "default"),
        "pooling": args.pooling,
        "eagerly": bool(tf.executing_eagerly()),
        "baseline_logits": delta_summary(actual_logits, expected_logits),
        "baseline_probabilities": delta_summary(actual_probabilities, expected_probabilities),
        "prediction_agreement": float(
            np.mean(actual_logits.argmax(axis=1) == expected_logits.argmax(axis=1))
        ),
        "classifier_input": delta_summary(actual_classifier_input, expected_classifier_input),
        "layers": layers,
        "top20": top20,
        "classifier_candidates": candidates,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output_npz,
        expected_classifier_input=expected_classifier_input,
        tensorflow_classifier_input=actual_classifier_input,
        expected_logits=expected_logits,
        tensorflow_logits=actual_logits,
        **{f"trace_{name}": value for name, value in trace_arrays.items()},
        **candidate_arrays,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
