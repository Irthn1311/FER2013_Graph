"""Shared pixel MLP -> unchanged gated layers -> global mean -> linear."""

import tensorflow as tf

from lap_gnn_tf.model.gated_edge_layer import GatedEdgeLayer
from lap_gnn_tf.model.initializers import TorchLinear


class PixelMLP(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="pixel_mlp_5_to_96")
        self.linear = TorchLinear(5, 96, "pixel_encoder.linear", name="linear_5_to_96")

    def call(self, x, training=False):
        h = tf.nn.gelu(self.linear(x), approximate=False)
        return tf.nn.dropout(h, rate=0.2) if training else h


class PixelGatedEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="pixel_gated_gnn")
        self.layers_ = [GatedEdgeLayer(i, hidden_dim=96, edge_dim=6, edge_hidden=32, dropout=0.25) for i in range(3)]

    def call(self, h, edge_index, edge_features, training=False):
        for layer in self.layers_:
            h = layer(h, edge_index, edge_features, training=training)
        return h


class GlobalMeanPooling(tf.keras.layers.Layer):
    def call(self, h, node_graph_index, num_graphs):
        return tf.math.unsorted_segment_mean(h, tf.cast(node_graph_index, tf.int32), num_graphs)


@tf.keras.utils.register_keras_serializable(package="pixel_gnn_only")
class PixelGNNOnly(tf.keras.Model):
    def __init__(self, **kwargs):
        kwargs.pop("name", None)
        super().__init__(name="pixel_gnn_only", **kwargs)
        self.encoder = PixelMLP()
        self.gnn = PixelGatedEncoder()
        self.pool = GlobalMeanPooling(name="global_mean_pooling")
        self.classifier = TorchLinear(96, 7, "classifier", name="linear_96_to_7")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, batch, training=False):
        tf.debugging.assert_equal(tf.shape(batch["node_features"])[1], 5)
        tf.debugging.assert_equal(batch["graph_node_counts"], tf.constant(2304, tf.int64))
        h = self.encoder(tf.cast(batch["node_features"], tf.float32), training=training)
        h = self.gnn(h, batch["edge_index"], tf.cast(batch["edge_features"], tf.float32), training=training)
        pooled = self.pool(h, batch["node_graph_index"], tf.shape(batch["graph_node_counts"])[0])
        logits = self.classifier(pooled)
        return {"logits": logits, "probabilities": tf.nn.softmax(logits, axis=-1),
                "predictions": tf.argmax(logits, axis=1, output_type=tf.int64),
                "node_embeddings": h, "z_image": pooled}

    def get_config(self):
        return super().get_config()
