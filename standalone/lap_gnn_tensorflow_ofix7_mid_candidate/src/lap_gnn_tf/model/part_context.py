"""Final five-token part/global context injection."""

from __future__ import annotations

import tensorflow as tf

from lap_gnn_tf.model.initializers import (
    MappedLayer,
    StateBinding,
    TorchLayerNorm,
    TorchLinear,
    TorchTransformerEncoderLayer,
)


class PartGlobalContext(MappedLayer):
    GROUP_INDICES = [[5, 6, 7], [0, 1], [2, 3], [4, 8, 9, 10]]

    def __init__(self, hidden_dim: int = 96, dropout: float = 0.25):
        super().__init__(name="part_global_context")
        self.dropout_rate = float(dropout)
        self.context_scale = self.add_weight(
            name="context_scale", shape=(), initializer=tf.keras.initializers.Constant(0.5),
            trainable=True, dtype=tf.float32,
        )
        self.transformer = TorchTransformerEncoderLayer(
            hidden_dim, 4, hidden_dim * 2, dropout,
            "gnn.context_block.context_mixer.layers.0", name="context_transformer",
        )
        self.update_in = TorchLinear(hidden_dim * 2, hidden_dim, "gnn.context_block.context_update.0", name="update_in")
        self.update_out = TorchLinear(hidden_dim, hidden_dim, "gnn.context_block.context_update.3", name="update_out")
        self.norm = TorchLayerNorm(hidden_dim, "gnn.context_block.norm", name="context_norm")

    def call(self, h, part_soft, node_graph_index, num_graphs, training: bool = False):
        priors = [tf.reduce_max(tf.gather(part_soft, indices, axis=1), axis=1) for indices in self.GROUP_INDICES]
        priors.append(tf.ones((tf.shape(part_soft)[0],), dtype=part_soft.dtype))
        priors = tf.stack(priors, axis=1)
        token_sums = tf.math.unsorted_segment_sum(
            h[:, None, :] * priors[:, :, None], node_graph_index, num_graphs,
        )
        denom = tf.math.unsorted_segment_sum(priors, node_graph_index, num_graphs)
        tokens = token_sums / tf.maximum(denom[:, :, None], tf.cast(1e-6, h.dtype))
        mixed = self.transformer(tokens, training=training)
        mixed_nodes = tf.gather(mixed, node_graph_index)
        weights = priors[:, :4]
        weights = weights / tf.maximum(tf.reduce_sum(weights, axis=1, keepdims=True), tf.cast(1e-6, h.dtype))
        local_context = tf.reduce_sum(weights[:, :, None] * mixed_nodes[:, :4, :], axis=1)
        global_context = mixed_nodes[:, 4, :]
        context = 0.5 * (local_context + global_context)
        update = tf.nn.gelu(self.update_in(tf.concat([h, context], axis=1)), approximate=False)
        if training and self.dropout_rate:
            update = tf.nn.dropout(update, rate=self.dropout_rate)
        update = self.update_out(update)
        if training and self.dropout_rate:
            update = tf.nn.dropout(update, rate=self.dropout_rate)
        return self.norm(h + tf.cast(self.context_scale, h.dtype) * update)

    def state_bindings(self) -> list[StateBinding]:
        bindings = [StateBinding("gnn.context_block.context_scale", self.context_scale, "identity")]
        for layer in [self.transformer, self.update_in, self.update_out, self.norm]:
            bindings.extend(layer.state_bindings())
        return bindings
