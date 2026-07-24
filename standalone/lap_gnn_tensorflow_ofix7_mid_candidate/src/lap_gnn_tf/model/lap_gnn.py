"""Complete TensorFlow/Keras OFIX7-mid LAP-GNN model."""

from __future__ import annotations

import tensorflow as tf

from lap_gnn_tf.model.classifier import Classifier
from lap_gnn_tf.model.edge_context import EdgeContextEncoder
from lap_gnn_tf.model.initializers import StateBinding, TorchLayerNorm, TorchLinear
from lap_gnn_tf.model.motif_layers import part_pool
from lap_gnn_tf.model.readout import MicroMotifSupportReadout


class PixelEncoder(tf.keras.layers.Layer):
    def __init__(self, dropout: float = 0.2):
        super().__init__(name="pixel_encoder")
        self.dropout_rate = float(dropout)
        self.linear1 = TorchLinear(37, 96, "encoder.net.0", name="linear1")
        self.norm1 = TorchLayerNorm(96, "encoder.net.1", name="norm1")
        self.linear2 = TorchLinear(96, 96, "encoder.net.4", name="linear2")
        self.norm2 = TorchLayerNorm(96, "encoder.net.5", name="norm2")

    def call(self, x, training: bool = False):
        x = tf.nn.gelu(self.norm1(self.linear1(x)), approximate=False)
        if training and self.dropout_rate:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        return tf.nn.gelu(self.norm2(self.linear2(x)), approximate=False)

    def state_bindings(self) -> list[StateBinding]:
        bindings: list[StateBinding] = []
        for layer in [self.linear1, self.norm1, self.linear2, self.norm2]:
            bindings.extend(layer.state_bindings())
        return bindings


@tf.keras.utils.register_keras_serializable(package="lap_gnn_tf")
class LapGNN(tf.keras.Model):
    def __init__(self, **kwargs):
        kwargs.pop("name", None)
        super().__init__(name="lap_gnn_ofix7_mid", **kwargs)
        self.encoder = PixelEncoder()
        self.gnn = EdgeContextEncoder()
        self.readout = MicroMotifSupportReadout()
        self.classifier = Classifier()

    def build(self, input_shape):
        """Register the structured graph input without creating new state."""
        super().build(input_shape)

    def call(self, batch: dict[str, tf.Tensor], training: bool = False, collect_intermediates: bool = False):
        node_features = tf.cast(batch["node_features"], tf.float32)
        edge_features = tf.cast(batch["edge_features"], tf.float32)
        edge_index = tf.cast(batch["edge_index"], tf.int64)
        node_graph_index = tf.cast(batch["node_graph_index"], tf.int32)
        part_soft = tf.cast(batch["part_soft"], tf.float32)
        valid_part_mask = tf.cast(batch["valid_part_mask"], tf.float32)
        num_graphs = tf.shape(batch["labels"])[0]
        h = self.encoder(node_features, training=training)
        intermediates = {"input_projection": h}
        if collect_intermediates:
            h, gnn_outputs = self.gnn.encode(
                h, edge_index, edge_features, node_graph_index, num_graphs,
                part_soft, training=training, collect=True,
            )
            intermediates.update(gnn_outputs)
        else:
            h = self.gnn.encode(
                h, edge_index, edge_features, node_graph_index, num_graphs,
                part_soft, training=training, collect=False,
            )
        pooled, valid = part_pool(h, part_soft, node_graph_index, valid_part_mask, num_graphs)
        readout = self.readout(
            h, node_features, part_soft, node_graph_index, num_graphs,
            pooled, valid, training=training,
        )
        z_image = readout["z_image"]
        logits = self.classifier(z_image, training=training)
        probabilities = tf.nn.softmax(logits, axis=-1)
        if collect_intermediates:
            intermediates.update({
                "micro_major_motif_tokens": readout["major_tokens"],
                "micro_major_motif_transformed_tokens": readout["major_transformed_tokens"],
                "micro_motif_tokens": readout["micro_tokens"],
                "micro_motif_transformed_tokens": readout["micro_transformed_tokens"],
                "micro_support_gate": readout["micro_gate"],
                "pooled_graph_embedding": z_image,
                "classifier_input": z_image,
            })
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": tf.argmax(logits, axis=1, output_type=tf.int64),
            "z_image": z_image,
            "node_embeddings": h,
            "part_embeddings": pooled,
            "intermediates": intermediates,
        }

    def state_bindings(self) -> list[StateBinding]:
        bindings: list[StateBinding] = []
        for layer in [self.encoder, self.gnn, self.readout, self.classifier]:
            bindings.extend(layer.state_bindings())
        return bindings

    def mapped_trainable_variables(self) -> dict[str, StateBinding]:
        bindings = self.state_bindings()
        return {binding.source_key: binding for binding in bindings}

    def get_config(self):
        return super().get_config()


def build_model(golden_batch: dict[str, tf.Tensor] | None = None) -> LapGNN:
    model = LapGNN()
    if golden_batch is not None:
        model(golden_batch, training=False)
    return model
