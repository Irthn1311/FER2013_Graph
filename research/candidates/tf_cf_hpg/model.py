"""CF-HPG v1.0 model: raw patches, generic hierarchy, and graph mixing."""

from __future__ import annotations

import tensorflow as tf

from .graph import HybridGraphBuilder, max_relative_aggregate, normalized_grid_centers


IMAGE_SIZE = 48
PATCH_SIZE = 6
FINE_GRID_SIZE = 8
FINE_NODE_COUNT = 64
RAW_PATCH_WIDTH = 36
FINE_WIDTH = 96
COARSE_GRID_SIZE = 4
COARSE_NODE_COUNT = 16
COARSE_WIDTH = 128
NUM_CLASSES = 7


def patchify_and_scale(images: tf.Tensor) -> tf.Tensor:
    """Scale raw grayscale pixels and rearrange them into row-major patches."""

    images = tf.cast(tf.convert_to_tensor(images), tf.float32)
    tf.debugging.assert_rank(images, 4)
    tf.debugging.assert_equal(tf.shape(images)[1:], [IMAGE_SIZE, IMAGE_SIZE, 1])
    images = images / 127.5 - 1.0
    batch_size = tf.shape(images)[0]
    patches = tf.reshape(
        images,
        [batch_size, FINE_GRID_SIZE, PATCH_SIZE, FINE_GRID_SIZE, PATCH_SIZE, 1],
    )
    patches = tf.transpose(patches, [0, 1, 3, 2, 4, 5])
    return tf.reshape(patches, [batch_size, FINE_NODE_COUNT, RAW_PATCH_WIDTH])


def pool_2x2_mean(nodes: tf.Tensor) -> tf.Tensor:
    """Parameter-free 8x8 to 4x4 non-overlapping arithmetic-mean hierarchy."""

    nodes = tf.convert_to_tensor(nodes)
    tf.debugging.assert_rank(nodes, 3)
    tf.debugging.assert_equal(tf.shape(nodes)[1], FINE_NODE_COUNT)
    batch_size = tf.shape(nodes)[0]
    width = tf.shape(nodes)[2]
    grid = tf.reshape(nodes, [batch_size, 4, 2, 4, 2, width])
    pooled = tf.reduce_mean(grid, axis=[2, 4])
    return tf.reshape(pooled, [batch_size, COARSE_NODE_COUNT, width])


@tf.keras.utils.register_keras_serializable(package="fer2013_graph_research")
class MaxRelativeGraphBlock(tf.keras.layers.Layer):
    """Pre-normalized residual max-relative graph block."""

    def __init__(self, width: int, dropout_rate: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.width = int(width)
        self.dropout_rate = float(dropout_rate)
        self.graph_norm = tf.keras.layers.LayerNormalization(name="graph_norm")
        self.graph_dense_in = tf.keras.layers.Dense(self.width, name="graph_dense_in")
        self.graph_activation = tf.keras.layers.Activation("gelu")
        self.graph_dense_out = tf.keras.layers.Dense(self.width, name="graph_dense_out")
        self.graph_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.ffn_norm = tf.keras.layers.LayerNormalization(name="ffn_norm")
        self.ffn_dense_in = tf.keras.layers.Dense(self.width * 2, name="ffn_dense_in")
        self.ffn_activation = tf.keras.layers.Activation("gelu")
        self.ffn_dense_out = tf.keras.layers.Dense(self.width, name="ffn_dense_out")
        self.ffn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, nodes, adjacency, training=False):
        normalized = self.graph_norm(nodes)
        relative = max_relative_aggregate(normalized, adjacency)
        mixed = tf.concat([normalized, relative], axis=-1)
        mixed = self.graph_dense_in(mixed)
        mixed = self.graph_activation(mixed)
        mixed = self.graph_dense_out(mixed)
        nodes = nodes + self.graph_dropout(mixed, training=training)
        feed_forward = self.ffn_dense_in(self.ffn_norm(nodes))
        feed_forward = self.ffn_activation(feed_forward)
        feed_forward = self.ffn_dense_out(feed_forward)
        return nodes + self.ffn_dropout(feed_forward, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"width": self.width, "dropout_rate": self.dropout_rate})
        return config


@tf.keras.utils.register_keras_serializable(package="fer2013_graph_research")
class CFHPG(tf.keras.Model):
    """Registered Generation-2 convolution-free hierarchical patch graph."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_projection = tf.keras.layers.Dense(FINE_WIDTH, name="raw_projection")
        self.position_projection = tf.keras.layers.Dense(
            FINE_WIDTH, name="position_projection"
        )
        self.patch_norm = tf.keras.layers.LayerNormalization(name="patch_norm")
        self.patch_activation = tf.keras.layers.Activation("gelu")
        self.patch_dropout = tf.keras.layers.Dropout(0.10)
        self.fine_graph_builder = HybridGraphBuilder(8, 4, name="fine_hybrid_graph")
        self.fine_blocks = [
            MaxRelativeGraphBlock(FINE_WIDTH, 0.15, name=f"fine_block_{index + 1}")
            for index in range(2)
        ]
        self.coarse_projection = tf.keras.layers.Dense(
            COARSE_WIDTH, name="coarse_projection"
        )
        self.coarse_norm = tf.keras.layers.LayerNormalization(name="coarse_norm")
        self.coarse_activation = tf.keras.layers.Activation("gelu")
        self.coarse_graph_builder = HybridGraphBuilder(
            4, 4, name="coarse_hybrid_graph"
        )
        self.coarse_blocks = [
            MaxRelativeGraphBlock(COARSE_WIDTH, 0.15, name=f"coarse_block_{index + 1}")
            for index in range(2)
        ]
        self.readout_norm = tf.keras.layers.LayerNormalization(name="readout_norm")
        self.readout_dense = tf.keras.layers.Dense(128, name="readout_dense")
        self.readout_activation = tf.keras.layers.Activation("gelu")
        self.readout_dropout = tf.keras.layers.Dropout(0.20)
        self.classifier = tf.keras.layers.Dense(NUM_CLASSES, name="classifier")

    def call(self, images, training=False, return_debug=False):
        patches = patchify_and_scale(images)
        positions = normalized_grid_centers(FINE_GRID_SIZE, patches.dtype)
        positions = tf.broadcast_to(positions[None, :, :], [tf.shape(patches)[0], 64, 2])
        nodes = self.raw_projection(patches) + self.position_projection(positions)
        nodes = self.patch_norm(nodes)
        nodes = self.patch_activation(nodes)
        nodes = self.patch_dropout(nodes, training=training)

        fine_adjacency = self.fine_graph_builder(nodes)
        for block in self.fine_blocks:
            nodes = block(nodes, fine_adjacency, training=training)

        nodes = pool_2x2_mean(nodes)
        nodes = self.coarse_projection(nodes)
        nodes = self.coarse_norm(nodes)
        nodes = self.coarse_activation(nodes)

        coarse_adjacency = self.coarse_graph_builder(nodes)
        for block in self.coarse_blocks:
            nodes = block(nodes, coarse_adjacency, training=training)

        pooled = tf.concat(
            [tf.reduce_mean(nodes, axis=1), tf.reduce_max(nodes, axis=1)], axis=-1
        )
        pooled = self.readout_norm(pooled)
        pooled = self.readout_dense(pooled)
        pooled = self.readout_activation(pooled)
        pooled = self.readout_dropout(pooled, training=training)
        logits = self.classifier(pooled)
        if return_debug:
            return logits, {
                "fine_adjacency": fine_adjacency,
                "coarse_adjacency": coarse_adjacency,
                "final_nodes": nodes,
            }
        return logits

    def get_config(self):
        return super().get_config()


def build_cf_hpg_v1() -> CFHPG:
    model = CFHPG(name="cf_hpg_v1")
    model(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1], tf.float32), training=False)
    return model
