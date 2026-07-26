"""Exact residual edge-gated message-passing layer."""

from __future__ import annotations

import tensorflow as tf

from lap_gnn_tf.model.initializers import MappedLayer, StateBinding, TorchLayerNorm, TorchLinear


class GatedEdgeLayer(MappedLayer):
    def __init__(self, index: int, hidden_dim: int = 96, edge_dim: int = 8, edge_hidden: int = 32, dropout: float = 0.25):
        super().__init__(name=f"gnn_layer_{index + 1}")
        prefix = f"gnn.layers.{index}"
        self.dropout_rate = float(dropout)
        self.edge_linear = TorchLinear(edge_dim, edge_hidden, f"{prefix}.edge_mlp.0", name="edge_linear")
        self.edge_norm = TorchLayerNorm(edge_hidden, f"{prefix}.edge_mlp.1", name="edge_norm")
        self.gate = TorchLinear(edge_hidden, hidden_dim, f"{prefix}.gate", name="gate")
        self.message_in = TorchLinear(hidden_dim + edge_hidden, hidden_dim, f"{prefix}.message.0", name="message_in")
        self.message_out = TorchLinear(hidden_dim, hidden_dim, f"{prefix}.message.3", name="message_out")
        self.norm_msg = TorchLayerNorm(hidden_dim, f"{prefix}.norm_msg", name="norm_msg")
        self.ffn_in = TorchLinear(hidden_dim, hidden_dim * 2, f"{prefix}.ffn.0", name="ffn_in")
        self.ffn_out = TorchLinear(hidden_dim * 2, hidden_dim, f"{prefix}.ffn.3", name="ffn_out")
        self.norm_ffn = TorchLayerNorm(hidden_dim, f"{prefix}.norm_ffn", name="norm_ffn")

    def call(self, h, edge_index, edge_features, training: bool = False, return_intermediates: bool = False):
        src = tf.cast(edge_index[0], tf.int32)
        dst = tf.cast(edge_index[1], tf.int32)
        edge_emb = tf.nn.gelu(self.edge_norm(self.edge_linear(edge_features)), approximate=False)
        if training and self.dropout_rate:
            edge_emb = tf.nn.dropout(edge_emb, rate=self.dropout_rate)
        gate = tf.nn.sigmoid(self.gate(edge_emb))
        message_input = tf.concat([tf.gather(h, src), edge_emb], axis=1)
        message = tf.nn.gelu(self.message_in(message_input), approximate=False)
        if training and self.dropout_rate:
            message = tf.nn.dropout(message, rate=self.dropout_rate)
        message = self.message_out(message) * gate
        node_count = tf.shape(h)[0]
        aggregate_sum = tf.math.unsorted_segment_sum(message, dst, node_count)
        degree = tf.math.unsorted_segment_sum(
            tf.ones((tf.shape(dst)[0], 1), dtype=h.dtype), dst, node_count,
        )
        aggregate = aggregate_sum / tf.maximum(degree, tf.cast(1.0, h.dtype))
        h_msg = self.norm_msg(h + aggregate)
        ffn = tf.nn.gelu(self.ffn_in(h_msg), approximate=False)
        if training and self.dropout_rate:
            ffn = tf.nn.dropout(ffn, rate=self.dropout_rate)
        ffn = self.ffn_out(ffn)
        if training and self.dropout_rate:
            ffn = tf.nn.dropout(ffn, rate=self.dropout_rate)
        output = self.norm_ffn(h_msg + ffn)
        if not return_intermediates:
            return output
        return output, {
            "edge_projection": edge_emb,
            "edge_gate": gate,
            "pre_aggregation_message": message,
            "aggregate_output": aggregate,
            "gated_update": h_msg,
            "residual_output": output,
        }

    def state_bindings(self) -> list[StateBinding]:
        bindings: list[StateBinding] = []
        for layer in [
            self.edge_linear, self.edge_norm, self.gate, self.message_in,
            self.message_out, self.norm_msg, self.ffn_in, self.ffn_out, self.norm_ffn,
        ]:
            bindings.extend(layer.state_bindings())
        return bindings

