"""Locked coarse-to-fine micro-motif support readout."""

from __future__ import annotations

import tensorflow as tf

from lap_gnn_tf.model.initializers import (
    MappedLayer,
    StateBinding,
    TorchLayerNorm,
    TorchLinear,
    TorchTransformerEncoderLayer,
)
from lap_gnn_tf.model.motif_layers import PART_ORDER, group_priors, pad_flat_nodes


class MicroMotifSupportReadout(MappedLayer):
    MAJOR_PART_INDEX = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 4]
    MICRO_PART_INDEX = [0, 0, 1, 1, 2, 2, 3, 4]

    def __init__(self, hidden_dim: int = 96, output_dim: int = 480, dropout: float = 0.2):
        super().__init__(name="micro_motif_support")
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dropout_rate = float(dropout)
        self.eps = 1e-6
        self.major_queries = self.add_weight(
            name="major_queries", shape=(12, hidden_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True, dtype=tf.float32,
        )
        self.micro_queries = self.add_weight(
            name="micro_queries", shape=(8, hidden_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True, dtype=tf.float32,
        )
        self.cls_token = self.add_weight(
            name="cls_token", shape=(1, 1, hidden_dim), initializer="zeros", trainable=True, dtype=tf.float32,
        )
        self.token_type_embedding = self.add_weight(
            name="token_type_embedding", shape=(1, 20, hidden_dim), initializer="zeros", trainable=True, dtype=tf.float32,
        )
        self.major_key = TorchLinear(hidden_dim, hidden_dim, "readout.major_key_proj", name="major_key")
        self.major_value = TorchLinear(hidden_dim, hidden_dim, "readout.major_value_proj", name="major_value")
        self.micro_key = TorchLinear(hidden_dim, hidden_dim, "readout.micro_key_proj", name="micro_key")
        self.micro_value = TorchLinear(hidden_dim, hidden_dim, "readout.micro_value_proj", name="micro_value")
        self.transformer = TorchTransformerEncoderLayer(
            hidden_dim, 4, hidden_dim * 2, dropout,
            "readout.encoder.layers.0", name="readout_transformer",
        )
        self.micro_norm = TorchLayerNorm(hidden_dim, "readout.micro_project.0", name="micro_norm")
        self.micro_project = TorchLinear(hidden_dim, hidden_dim, "readout.micro_project.1", name="micro_project")
        self.gate_norm = TorchLayerNorm(hidden_dim * 2, "readout.gate.0", name="gate_norm")
        self.gate_in = TorchLinear(hidden_dim * 2, hidden_dim, "readout.gate.1", name="gate_in")
        self.gate_out = TorchLinear(hidden_dim, hidden_dim, "readout.gate.3", name="gate_out")
        self.projection_norm = TorchLayerNorm(576, "readout.projection.0", name="projection_norm")
        self.projection_in = TorchLinear(576, output_dim, "readout.projection.1", name="projection_in")
        self.projection_out = TorchLinear(output_dim, output_dim, "readout.projection.4", name="projection_out")

    def _branch(self, h_pad, priors_pad, node_valid, valid_groups, queries, key_layer, value_layer, part_index, detail=None):
        keys = key_layer(h_pad)
        values = value_layer(h_pad)
        scores = tf.einsum("kh,bnh->bkn", queries, keys) * tf.cast(self.hidden_dim ** -0.5, h_pad.dtype)
        prior = tf.gather(priors_pad, part_index, axis=1)
        prior = tf.maximum(prior, tf.cast(self.eps, prior.dtype))
        scores = scores + tf.math.log(prior)
        if detail is not None:
            scores = scores + tf.cast(0.05, scores.dtype) * detail[:, None, :]
        scores = tf.where(node_valid[:, None, :], scores, tf.cast(-1e9, scores.dtype))
        shifted = scores - tf.reduce_max(scores, axis=-1, keepdims=True)
        alpha = tf.exp(shifted)
        alpha = alpha / tf.reduce_sum(alpha, axis=-1, keepdims=True)
        motif_valid = tf.cast(tf.gather(valid_groups, part_index, axis=1), alpha.dtype)
        alpha = alpha * motif_valid[:, :, None]
        tokens = tf.linalg.matmul(alpha, values)
        safe_alpha = tf.maximum(alpha, tf.cast(self.eps, alpha.dtype))
        entropy = -tf.reduce_sum(alpha * tf.math.log(safe_alpha), axis=-1)
        peak = tf.reduce_max(alpha, axis=-1)
        mass = tf.reduce_sum(alpha * tf.clip_by_value(prior, 0.0, 1.0), axis=-1)
        detail_mean = (
            tf.reduce_sum(alpha * detail[:, None, :], axis=-1)
            if detail is not None else tf.zeros(tf.shape(alpha)[:2], dtype=alpha.dtype)
        )
        return tokens, entropy, peak, mass, detail_mean

    def call(self, h, node_features, part_soft, node_graph_index, num_graphs, part_embeddings, valid_groups, training: bool = False):
        h_pad, node_valid, _, scatter_indices = pad_flat_nodes(h, node_graph_index, num_graphs)
        priors_flat = group_priors(part_soft)
        priors_pad, _, _, _ = pad_flat_nodes(priors_flat, node_graph_index, num_graphs, max_nodes=tf.shape(h_pad)[1])
        priors_pad = tf.transpose(priors_pad, (0, 2, 1))

        gx = node_features[:, 1]
        gy = node_features[:, 2]
        raw_detail = tf.sqrt(tf.square(gx) + tf.square(gy) + tf.cast(self.eps, gx.dtype))
        detail_mean = tf.math.unsorted_segment_mean(raw_detail, node_graph_index, num_graphs)
        centered = raw_detail - tf.gather(detail_mean, node_graph_index)
        detail_var = tf.math.unsorted_segment_mean(tf.square(centered), node_graph_index, num_graphs)
        detail = centered / tf.maximum(tf.sqrt(tf.gather(detail_var, node_graph_index)), tf.cast(self.eps, gx.dtype))
        detail = tf.clip_by_value(tf.stop_gradient(detail), -2.0, 2.0)
        detail_pad = tf.scatter_nd(scatter_indices, detail, tf.shape(h_pad)[:2])

        valid_matrix = tf.stack([valid_groups[name] for name in PART_ORDER], axis=1)
        major = self._branch(
            h_pad, priors_pad, node_valid, valid_matrix, self.major_queries,
            self.major_key, self.major_value, self.MAJOR_PART_INDEX,
        )
        micro = self._branch(
            h_pad, priors_pad, node_valid, valid_matrix, self.micro_queries,
            self.micro_key, self.micro_value, self.MICRO_PART_INDEX, detail=detail_pad,
        )
        major_tokens, major_entropy, major_peak, major_mass, _ = major
        micro_tokens, micro_entropy, micro_peak, micro_mass, micro_detail = micro
        tokens = tf.concat([major_tokens, micro_tokens], axis=1) + self.token_type_embedding
        cls = tf.tile(self.cls_token, [num_graphs, 1, 1])
        transformed = self.transformer(tf.concat([cls, tokens], axis=1), training=training)
        transformed_all = transformed[:, 1:, :]
        transformed_major = transformed_all[:, :12, :]
        transformed_micro = transformed_all[:, 12:, :]
        z_major = tf.reduce_mean(transformed_major, axis=1)
        z_micro = tf.reduce_mean(transformed_micro, axis=1)
        gate_hidden = tf.nn.gelu(self.gate_in(self.gate_norm(tf.concat([z_major, z_micro], axis=1))), approximate=False)
        gate = tf.nn.sigmoid(self.gate_out(gate_hidden))
        z_support = z_major + gate * self.micro_project(self.micro_norm(z_micro))
        residual = tf.reshape(tf.stack([part_embeddings[name] for name in PART_ORDER], axis=1), (num_graphs, 480))
        fused = tf.concat([z_support, residual], axis=1)
        projected = tf.nn.gelu(self.projection_in(self.projection_norm(fused)), approximate=False)
        if training and self.dropout_rate:
            projected = tf.nn.dropout(projected, rate=self.dropout_rate)
        z_image = self.projection_out(projected)
        return {
            "z_image": z_image,
            "major_tokens": major_tokens,
            "major_transformed_tokens": transformed_major,
            "major_attention_entropy": major_entropy,
            "major_attention_peak": major_peak,
            "major_part_mass": major_mass,
            "micro_tokens": micro_tokens,
            "micro_transformed_tokens": transformed_micro,
            "micro_attention_entropy": micro_entropy,
            "micro_attention_peak": micro_peak,
            "micro_part_mass": micro_mass,
            "micro_detail_score": micro_detail,
            "micro_gate": gate,
        }

    def state_bindings(self) -> list[StateBinding]:
        bindings = [
            StateBinding("readout.major_queries", self.major_queries, "identity"),
            StateBinding("readout.micro_queries", self.micro_queries, "identity"),
            StateBinding("readout.cls_token", self.cls_token, "identity"),
            StateBinding("readout.token_type_embedding", self.token_type_embedding, "identity"),
        ]
        for layer in [
            self.major_key, self.major_value, self.micro_key, self.micro_value,
            self.transformer, self.micro_norm, self.micro_project,
            self.gate_norm, self.gate_in, self.gate_out,
            self.projection_norm, self.projection_in, self.projection_out,
        ]:
            bindings.extend(layer.state_bindings())
        return bindings
