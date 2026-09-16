"""Pixel-GNN Only Model (Gated Edge GNN x 3 + Global Mean Pooling)."""

from __future__ import annotations

import tensorflow as tf

from lap_gnn_tf.model.gated_edge_layer import GatedEdgeLayer
from lap_gnn_tf.model.initializers import TorchLinear
from pixel_gnn.grid import StaticGridTopology


class PixelMLP(tf.keras.layers.Layer):
    def __init__(self, hidden_dim: int = 96, dropout: float = 0.2):
        super().__init__(name="pixel_mlp")
        self.linear = TorchLinear(5, hidden_dim, "pixel_encoder.linear", name="linear_5_to_hidden")
        self.dropout_rate = float(dropout)

    def call(self, x, training=False):
        h = tf.nn.gelu(self.linear(x), approximate=False)
        return tf.nn.dropout(h, rate=self.dropout_rate) if training else h


class PixelGatedEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dim: int = 96, edge_dim: int = 6, edge_hidden: int = 32, num_layers: int = 3, dropout: float = 0.25):
        super().__init__(name="pixel_gated_gnn")
        self.layers_ = [
            GatedEdgeLayer(i, hidden_dim=hidden_dim, edge_dim=edge_dim, edge_hidden=edge_hidden, dropout=dropout)
            for i in range(num_layers)
        ]

    def call(self, h, edge_index, edge_features, training=False):
        for layer in self.layers_:
            h = layer(h, edge_index, edge_features, training=training)
        return h


@tf.keras.utils.register_keras_serializable(package="pixel_gnn")
class PixelGNNOnly(tf.keras.Model):
    """Reference Pixel-GNN Only architecture (3 GatedEdgeLayers -> GlobalMean -> Linear)."""

    def __init__(
        self,
        hidden_dim: int = 96,
        num_layers: int = 3,
        edge_dim: int = 6,
        edge_hidden: int = 32,
        dropout: float = 0.2,
        gnn_dropout: float = 0.25,
        num_classes: int = 7,
        **kwargs,
    ):
        super().__init__(name="pixel_gnn_only", **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.edge_dim = int(edge_dim)
        self.num_classes = int(num_classes)

        self.encoder = PixelMLP(hidden_dim=self.hidden_dim, dropout=dropout)
        self.gnn = PixelGatedEncoder(
            hidden_dim=self.hidden_dim,
            edge_dim=self.edge_dim,
            edge_hidden=edge_hidden,
            num_layers=self.num_layers,
            dropout=gnn_dropout,
        )
        self.classifier = TorchLinear(self.hidden_dim, self.num_classes, "classifier", name="linear_hidden_to_classes")

    def call(self, batch, training=False):
        # Support both flat graph representation and batch 3D representation
        if "edge_index" in batch and "edge_features" in batch:
            # Flat graph mode
            h = self.encoder(tf.cast(batch["node_features"], tf.float32), training=training)
            h = self.gnn(h, batch["edge_index"], tf.cast(batch["edge_features"], tf.float32), training=training)
            node_count = tf.shape(batch["graph_node_counts"])[0] if "graph_node_counts" in batch else 1
            z = tf.math.unsorted_segment_mean(h, tf.cast(batch["node_graph_index"], tf.int32), node_count)
        else:
            # 3D representation: reconstruct grid edges
            x = batch["node_features"]  # [B, 2304, 5]
            batch_size = tf.shape(x)[0]
            x_flat = tf.reshape(x, [-1, 5])
            h = self.encoder(x_flat, training=training)

            # Use static grid edges
            grid = StaticGridTopology.get_instance()
            edge_index, edge_attr = grid.get_flat_batch_edges(batch_size)
            h = self.gnn(h, edge_index, edge_attr, training=training)

            h_3d = tf.reshape(h, [batch_size, 2304, self.hidden_dim])
            z = tf.reduce_mean(h_3d, axis=1)

        logits = self.classifier(z)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int64)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "z_image": z,
            "node_embeddings": h,
            "motif_assignment": None,
            "motif_prototypes": None,
            "motif_attention_weights": None,
            "motif_usage": None,
        }
