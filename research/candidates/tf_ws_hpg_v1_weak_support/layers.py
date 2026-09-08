"""Registered sparse PNA-lite relation blocks and hierarchy pooling."""

from __future__ import annotations

import tensorflow as tf

from .graph import sparse_pna_lite


SUPPORT_BETA = 0.25


class _SparseRelationCore(tf.keras.layers.Layer):
    def __init__(self, width: int, dropout_rate: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        self.width = int(width)
        self.dropout_rate = float(dropout_rate)
        self.graph_norm = tf.keras.layers.LayerNormalization(name="graph_norm")
        self.gate_hidden = tf.keras.layers.Dense(self.width // 2, activation="gelu", name="gate_hidden")
        self.gate_output = tf.keras.layers.Dense(1, activation="sigmoid", name="gate_output")
        self.message_projection = tf.keras.layers.Dense(self.width, activation="gelu", name="message_projection")
        self.node_mixer = tf.keras.layers.Dense(self.width, name="node_mixer")
        self.graph_dropout = tf.keras.layers.Dropout(self.dropout_rate, name="graph_dropout")
        self.ffn_norm = tf.keras.layers.LayerNormalization(name="ffn_norm")
        self.ffn_in = tf.keras.layers.Dense(2 * self.width, activation="gelu", name="ffn_in")
        self.ffn_dropout_in = tf.keras.layers.Dropout(self.dropout_rate, name="ffn_dropout_in")
        self.ffn_out = tf.keras.layers.Dense(self.width, name="ffn_out")
        self.ffn_dropout_out = tf.keras.layers.Dropout(self.dropout_rate, name="ffn_dropout_out")

    def _call_core(self, nodes, coordinates, edge_index, training, support=None, return_debug=False):
        nodes = tf.convert_to_tensor(nodes)
        batch_size, node_count = tf.shape(nodes)[0], tf.shape(nodes)[1]
        normalized = self.graph_norm(nodes)
        flat = tf.reshape(normalized, [-1, self.width])
        source, destination = edge_index[0], edge_index[1]
        z_source, z_destination = tf.gather(flat, source), tf.gather(flat, destination)
        difference = z_source - z_destination
        coordinates = tf.broadcast_to(coordinates[None, :, :], [batch_size, node_count, 2])
        flat_coordinates = tf.reshape(coordinates, [-1, 2])
        coordinate_difference = tf.gather(flat_coordinates, source) - tf.gather(flat_coordinates, destination)
        distance = tf.norm(coordinate_difference, axis=-1, keepdims=True)
        geometry = tf.concat([coordinate_difference, distance], axis=-1)
        gate_input = tf.concat([tf.abs(difference), geometry], axis=-1)
        learned_gate = self.gate_output(self.gate_hidden(gate_input))
        content_input = tf.concat([z_source, difference], axis=-1)
        content = self.message_projection(content_input)
        support_gate = tf.ones_like(learned_gate)
        if support is not None:
            flat_support = tf.reshape(support, [-1, 1])
            support_gate = SUPPORT_BETA + (1.0 - SUPPORT_BETA) * tf.sqrt(
                tf.maximum(tf.gather(flat_support, source) * tf.gather(flat_support, destination), 0.0)
            )
        messages = learned_gate * support_gate * content
        mean, maximum, std = sparse_pna_lite(messages, destination, batch_size * node_count)
        mixed = self.node_mixer(tf.concat([flat, mean, maximum, std], axis=-1))
        updated = nodes + self.graph_dropout(tf.reshape(mixed, tf.shape(nodes)), training=training)
        ffn = self.ffn_in(self.ffn_norm(updated))
        ffn = self.ffn_dropout_in(ffn, training=training)
        ffn = self.ffn_out(ffn)
        output = updated + self.ffn_dropout_out(ffn, training=training)
        if return_debug:
            return output, {
                "geometry": geometry,
                "gate_input": gate_input,
                "content_input": content_input,
                "learned_gate": learned_gate,
                "support_gate": support_gate,
                "messages": messages,
                "mean": mean,
                "max": maximum,
                "std": std,
            }
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"width": self.width, "dropout_rate": self.dropout_rate})
        return config


@tf.keras.utils.register_keras_serializable(package="fer2013_graph_research")
class FineWeakSupportRelationBlock(_SparseRelationCore):
    """Fine-only block whose API requires the weak-support score."""

    def call(self, nodes, coordinates, edge_index, support, training=False, return_debug=False):
        return self._call_core(nodes, coordinates, edge_index, training, support, return_debug)


@tf.keras.utils.register_keras_serializable(package="fer2013_graph_research")
class SupportFreeRelationBlock(_SparseRelationCore):
    """Mid/coarse block whose forward API cannot accept support."""

    def call(self, nodes, coordinates, edge_index, training=False, return_debug=False):
        return self._call_core(nodes, coordinates, edge_index, training, None, return_debug)


@tf.keras.utils.register_keras_serializable(package="fer2013_graph_research")
class MeanMaxPool2x2(tf.keras.layers.Layer):
    def __init__(self, input_grid: int, input_width: int, output_width: int, **kwargs):
        super().__init__(**kwargs)
        self.input_grid = int(input_grid)
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.projection = tf.keras.layers.Dense(self.output_width, name="projection")
        self.norm = tf.keras.layers.LayerNormalization(name="norm")

    def grouped_statistics(self, nodes):
        batch_size = tf.shape(nodes)[0]
        output_grid = self.input_grid // 2
        grid = tf.reshape(nodes, [batch_size, output_grid, 2, output_grid, 2, self.input_width])
        mean = tf.reduce_mean(grid, axis=[2, 4])
        maximum = tf.reduce_max(grid, axis=[2, 4])
        return tf.reshape(mean, [batch_size, output_grid**2, self.input_width]), tf.reshape(maximum, [batch_size, output_grid**2, self.input_width])

    def call(self, nodes):
        mean, maximum = self.grouped_statistics(nodes)
        return tf.nn.gelu(self.norm(self.projection(tf.concat([mean, maximum], axis=-1))))

    def get_config(self):
        config = super().get_config()
        config.update({"input_grid": self.input_grid, "input_width": self.input_width, "output_width": self.output_width})
        return config
