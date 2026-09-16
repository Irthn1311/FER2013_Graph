"""Exact residual edge-gated message-passing layer for Pixel-GNN."""

from __future__ import annotations

import tensorflow as tf


class GatedEdgeLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        index: int,
        hidden_dim: int = 96,
        edge_dim: int = 6,
        edge_hidden: int = 32,
        dropout: float = 0.25,
    ):
        super().__init__(name=f"gnn_layer_{index + 1}")
        self.dropout_rate = float(dropout)
        self.edge_linear = tf.keras.layers.Dense(edge_hidden, name="edge_linear")
        self.edge_norm = tf.keras.layers.LayerNormalization(name="edge_norm")
        self.gate = tf.keras.layers.Dense(hidden_dim, name="gate")
        self.message_in = tf.keras.layers.Dense(hidden_dim, name="message_in")
        self.message_out = tf.keras.layers.Dense(hidden_dim, name="message_out")
        self.norm_msg = tf.keras.layers.LayerNormalization(name="norm_msg")
        self.ffn_in = tf.keras.layers.Dense(hidden_dim * 2, activation="gelu", name="ffn_in")
        self.ffn_out = tf.keras.layers.Dense(hidden_dim, name="ffn_out")
        self.norm_ffn = tf.keras.layers.LayerNormalization(name="norm_ffn")

    def call(self, h, edge_index, edge_features, training: bool = False):
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
        ffn = self.ffn_out(self.ffn_in(h_msg))
        if training and self.dropout_rate:
            ffn = tf.nn.dropout(ffn, rate=self.dropout_rate)

        output = self.norm_ffn(h_msg + ffn)
        return output
