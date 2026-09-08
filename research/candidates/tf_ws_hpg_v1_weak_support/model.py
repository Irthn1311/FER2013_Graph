"""WS-HPG v1.0 model: weak support only in the sparse fine stage."""

from __future__ import annotations

import tensorflow as tf

from .graph import (
    batch_shared_edges,
    hybrid_spatial_cosine_edges,
    normalized_grid_coordinates,
    spatial_8_edge_index,
)
from .layers import FineWeakSupportRelationBlock, MeanMaxPool2x2, SupportFreeRelationBlock
from .support import patch_support_scores


IMAGE_SIZE = 48
PATCH_SIZE = 3
FINE_GRID = 16
FINE_NODES = 256
FINE_WIDTH = 64
MID_GRID = 8
MID_NODES = 64
MID_WIDTH = 96
COARSE_GRID = 4
COARSE_NODES = 16
COARSE_WIDTH = 128
TOP_K = 4
NUM_CLASSES = 7
EXPECTED_PARAMETER_COUNT = 707_213
EXPECTED_TRAINABLE_VARIABLE_COUNT = 118


def patchify(images):
    images = tf.cast(tf.convert_to_tensor(images), tf.float32)
    tf.debugging.assert_equal(tf.shape(images)[1:], [48, 48, 1])
    patches = tf.image.extract_patches(images, [1, 3, 3, 1], [1, 3, 3, 1], [1, 1, 1, 1], "VALID")
    return tf.reshape(patches, [tf.shape(images)[0], FINE_NODES, 9])


@tf.keras.utils.register_keras_serializable(package="fer2013_graph_research")
class WSHPGWeakSupport(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token_dense_1 = tf.keras.layers.Dense(64, activation="gelu", name="token_dense_1")
        self.token_dense_2 = tf.keras.layers.Dense(64, name="token_dense_2")
        self.position_projection = tf.keras.layers.Dense(64, name="position_projection")
        self.token_norm = tf.keras.layers.LayerNormalization(name="token_norm")
        self.token_dropout = tf.keras.layers.Dropout(0.10, name="token_dropout")
        self.fine_blocks = [FineWeakSupportRelationBlock(64, name=f"fine_block_{i + 1}") for i in range(2)]
        self.pool_1 = MeanMaxPool2x2(16, 64, 96, name="pool_1")
        self.mid_blocks = [SupportFreeRelationBlock(96, name=f"mid_block_{i + 1}") for i in range(2)]
        self.pool_2 = MeanMaxPool2x2(8, 96, 128, name="pool_2")
        self.coarse_blocks = [SupportFreeRelationBlock(128, name=f"coarse_block_{i + 1}") for i in range(2)]
        self.readout_norm = tf.keras.layers.LayerNormalization(name="readout_norm")
        self.readout_dense = tf.keras.layers.Dense(128, activation="gelu", name="readout_dense")
        self.readout_dropout = tf.keras.layers.Dropout(0.20, name="readout_dropout")
        self.classifier = tf.keras.layers.Dense(7, name="classifier")
        self._fine_spatial = spatial_8_edge_index(16)
        self._fine_coordinates = normalized_grid_coordinates(16)
        self._mid_coordinates = normalized_grid_coordinates(8)
        self._coarse_coordinates = normalized_grid_coordinates(4)

    def tokenize(self, images, training=False):
        patches = patchify(images)
        position = tf.broadcast_to(self._fine_coordinates[None, :, :], [tf.shape(patches)[0], 256, 2])
        nodes = self.token_dense_2(self.token_dense_1(patches)) + self.position_projection(position)
        return self.token_dropout(tf.nn.gelu(self.token_norm(nodes)), training=training)

    def call(self, inputs, training=False, support_override="normal", return_debug=False):
        if not isinstance(inputs, dict) or set(inputs) != {"images", "support"}:
            raise ValueError("inputs must contain exactly images and support")
        if support_override not in ("normal", "ones"):
            raise ValueError("support_override must be 'normal' or 'ones'")
        images = inputs["images"]
        pixel_support = tf.cast(inputs["support"], tf.float32)
        nodes = self.tokenize(images, training=training)
        fine_support = patch_support_scores(pixel_support)
        if support_override == "ones":
            fine_support = tf.ones_like(fine_support)
        fine_edges = batch_shared_edges(self._fine_spatial, tf.shape(nodes)[0], 256)
        fine_debug = []
        for block in self.fine_blocks:
            if return_debug:
                nodes, block_debug = block(nodes, self._fine_coordinates, fine_edges, fine_support, training=training, return_debug=True)
                fine_debug.append(block_debug)
            else:
                nodes = block(nodes, self._fine_coordinates, fine_edges, fine_support, training=training)
        fine_nodes = nodes
        nodes = self.pool_1(nodes)
        # Scientific boundary: support has no downstream API or data path.
        mid_edge_counts = []
        for block in self.mid_blocks:
            edges = hybrid_spatial_cosine_edges(nodes, 8, TOP_K)
            mid_edge_counts.append(tf.shape(edges)[1])
            nodes = block(nodes, self._mid_coordinates, edges, training=training)
        mid_nodes = nodes
        nodes = self.pool_2(nodes)
        coarse_edge_counts = []
        for block in self.coarse_blocks:
            edges = hybrid_spatial_cosine_edges(nodes, 4, TOP_K)
            coarse_edge_counts.append(tf.shape(edges)[1])
            nodes = block(nodes, self._coarse_coordinates, edges, training=training)
        coarse_nodes = nodes
        readout = tf.concat([tf.reduce_mean(nodes, axis=1), tf.reduce_max(nodes, axis=1)], axis=-1)
        logits = self.classifier(self.readout_dropout(self.readout_dense(self.readout_norm(readout)), training=training))
        if return_debug:
            return logits, {
                "token_nodes": self.tokenize(images, training=False),
                "fine_nodes": fine_nodes,
                "mid_nodes": mid_nodes,
                "coarse_nodes": coarse_nodes,
                "fine_edge_count": tf.shape(fine_edges)[1],
                "mid_edge_counts": tf.stack(mid_edge_counts),
                "coarse_edge_counts": tf.stack(coarse_edge_counts),
                "fine_block_debug": fine_debug,
            }
        return logits


def build_ws_hpg_v1_weak_support():
    model = WSHPGWeakSupport(name="ws_hpg_v1_weak_support")
    model({"images": tf.zeros([1, 48, 48, 1]), "support": tf.ones([1, 48, 48, 1])})
    if model.count_params() != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"parameter identity mismatch: {model.count_params()} != {EXPECTED_PARAMETER_COUNT}")
    if len(model.trainable_variables) != EXPECTED_TRAINABLE_VARIABLE_COUNT:
        raise RuntimeError(f"trainable-variable identity mismatch: {len(model.trainable_variables)} != {EXPECTED_TRAINABLE_VARIABLE_COUNT}")
    return model
