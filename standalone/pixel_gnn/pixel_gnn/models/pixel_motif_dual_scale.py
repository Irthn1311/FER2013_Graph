"""Hierarchical Pixel-to-Motif Dual-Scale Graph Neural Network for FER.

Architecture Pipeline:
  1. Pixel Node [B, 2304, 5] -> Projected [B, 2304, D]
  2. Multi-Head Local Neighbor Attention GNN (3 layers, 4 heads, edge-aware):
     Captures multi-directional pixel relationships (horizontal, vertical, contrast).
  3. Spatial-Aware Learned Clustering (2304 pixels -> K=32 Motifs):
     Coarsened Motif Graph A_motif [B, 32, 32].
  4. Motif GNN (2 layers):
     Relational reasoning between facial landmark regions.
  5. Dual-Scale Readout:
     - Macro Scale: Motif Attention Pooling z_motif [B, D]
     - Micro Scale: Pixel Global Attention Pooling z_pixel [B, D]
     - Fusion: Concat[z_motif, z_pixel] -> GELU -> LayerNorm -> Dropout -> z_fused [B, D]
  6. Classifier [B, 7]
"""

from __future__ import annotations

import math
import numpy as np
import tensorflow as tf

from pixel_gnn.grid import StaticGridTopology
from pixel_gnn.models.pixel_neighbor_motif import (
    SpatialLearnedMotifClustering,
    MotifGNNLayer,
    MultiHeadMotifGNNLayer,
    MotifAttentionPooling,
)


class MultiHeadLocalNeighborAttentionLayer(tf.keras.layers.Layer):
    """Vectorized Multi-Head Attention over static 8-neighborhood with edge context.
    
    Splits hidden dimension D into H heads of dimension d_k.
    Each head computes:
        score_ij^h = (Q_i^h + E_k,ij^h) . K_j^h / sqrt(d_k)
        message_i^h = sum_j alpha_ij^h * (V_j^h + E_v,ij^h)
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        num_heads: int = 4,
        edge_dim: int = 3,
        dropout: float = 0.15,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        self.head_dim = self.hidden_dim // self.num_heads
        self.edge_dim = int(edge_dim)
        self.dropout_rate = float(dropout)

        # Projections for Q, K, V
        self.q_dense = tf.keras.layers.Dense(self.hidden_dim, name="q_dense")
        self.k_dense = tf.keras.layers.Dense(self.hidden_dim, name="k_dense")
        self.v_dense = tf.keras.layers.Dense(self.hidden_dim, name="v_dense")

        # Edge context projections (for static edge features [2304, 8, 3])
        self.edge_k_dense = tf.keras.layers.Dense(self.hidden_dim, name="edge_k_dense")
        self.edge_v_dense = tf.keras.layers.Dense(self.hidden_dim, name="edge_v_dense")

        # Output projection for concatenated heads
        self.out_dense = tf.keras.layers.Dense(self.hidden_dim, name="out_dense")

        # Normalization and residual
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # Feed-Forward Network
        self.ffn_dense1 = tf.keras.layers.Dense(self.hidden_dim * 2, activation="gelu", name="ffn_dense1")
        self.ffn_dense2 = tf.keras.layers.Dense(self.hidden_dim, name="ffn_dense2")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.ffn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, h, training: bool = False):
        grid = StaticGridTopology.get_instance()
        neighbors_idx = grid.neighbors_idx        # [2304, 8]
        neighbor_valid = grid.neighbor_valid      # [2304, 8]
        static_edge = grid.static_edge_features    # [2304, 8, 3]

        batch_size = tf.shape(h)[0]
        num_nodes = tf.shape(h)[1]  # 2304

        # 1. Compute Q, K, V at node level: [B, 2304, H, head_dim]
        q = tf.reshape(self.q_dense(h), [batch_size, num_nodes, self.num_heads, self.head_dim])
        k = tf.reshape(self.k_dense(h), [batch_size, num_nodes, self.num_heads, self.head_dim])
        v = tf.reshape(self.v_dense(h), [batch_size, num_nodes, self.num_heads, self.head_dim])

        # 2. Gather neighbor keys & values: [B, 2304, 8, H, head_dim]
        k_neighbors = tf.gather(k, neighbors_idx, axis=1)
        v_neighbors = tf.gather(v, neighbors_idx, axis=1)

        # 3. Static edge projections: [1, 2304, 8, H, head_dim]
        e_k = tf.expand_dims(
            tf.reshape(self.edge_k_dense(static_edge), [num_nodes, 8, self.num_heads, self.head_dim]),
            axis=0,
        )
        e_v = tf.expand_dims(
            tf.reshape(self.edge_v_dense(static_edge), [num_nodes, 8, self.num_heads, self.head_dim]),
            axis=0,
        )

        # 4. Expand query: [B, 2304, 1, H, head_dim]
        q_exp = tf.expand_dims(q, axis=2)

        # 5. Multi-head scaled dot-product attention with edge modulation
        # (q + e_k) * k_neighbors: [B, 2304, 8, H]
        scale = tf.cast(math.sqrt(self.head_dim), tf.float32)
        scores = tf.reduce_sum((q_exp + e_k) * k_neighbors, axis=-1) / scale

        # 6. Mask boundary / invalid neighbors
        mask = tf.expand_dims(
            tf.broadcast_to(tf.expand_dims(neighbor_valid, axis=0), [batch_size, num_nodes, 8]),
            axis=-1,
        )  # [B, 2304, 8, 1]
        masked_scores = tf.where(mask, scores, tf.constant(-1e9, dtype=scores.dtype))

        # 7. Softmax across 8 neighbors per head: [B, 2304, 8, H]
        alpha = tf.nn.softmax(masked_scores, axis=2)
        alpha = tf.where(mask, alpha, tf.zeros_like(alpha))
        if training and self.dropout_rate > 0.0:
            alpha = self.attn_dropout(alpha, training=training)

        # 8. Aggregate values: sum_j alpha_ij * (v_j + e_v): [B, 2304, H, head_dim]
        alpha_exp = tf.expand_dims(alpha, axis=-1)  # [B, 2304, 8, H, 1]
        v_total = v_neighbors + e_v                  # [B, 2304, 8, H, head_dim]
        head_messages = tf.reduce_sum(alpha_exp * v_total, axis=2)

        # 9. Concat heads & project: [B, 2304, D]
        concat_messages = tf.reshape(head_messages, [batch_size, num_nodes, self.hidden_dim])
        projected = self.out_dense(concat_messages)

        # 10. Residual + LayerNorm
        h_res = self.norm1(h + projected)

        # 11. Feed-Forward Block + Residual + LayerNorm
        ffn = self.ffn_dense2(self.ffn_dense1(h_res))
        if training and self.dropout_rate > 0.0:
            ffn = self.ffn_dropout(ffn, training=training)
        h_out = self.norm2(h_res + ffn)

        return h_out


class GlobalPixelAttentionPooling(tf.keras.layers.Layer):
    """Global attention pooling over 2304 pixel nodes to extract micro-scale facial context."""

    def __init__(self, hidden_dim: int = 96, name: str | None = None):
        super().__init__(name=name)
        self.hidden_dim = int(hidden_dim)
        mid_dim = max(self.hidden_dim // 2, 16)
        self.w1 = tf.keras.layers.Dense(mid_dim, activation="tanh", name="w1")
        self.w2 = tf.keras.layers.Dense(1, name="w2")

    def call(self, h):
        # h: [B, 2304, D]
        # Attention scores: [B, 2304, 1]
        scores = self.w2(self.w1(h))
        weights = tf.nn.softmax(scores, axis=1)  # [B, 2304, 1]
        pooled = tf.reduce_sum(weights * h, axis=1)  # [B, D]
        return pooled, weights


class PixelMotifDualScaleModel(tf.keras.Model):
    """Hierarchical Pixel-to-Motif GNN with Multi-Head Attention and Dual-Scale Readout."""

    def __init__(
        self,
        hidden_dim: int = 96,
        num_attention_layers: int = 3,
        num_heads: int = 4,
        num_motifs: int = 36,
        spatial_span: float = 0.58,
        use_motif_graph: bool = True,
        num_motif_gnn_layers: int = 2,
        num_motif_heads: int = 4,
        temperature: float = 0.1,
        dropout: float = 0.15,
        num_classes: int = 7,
        **kwargs,
    ):
        super().__init__(name="pixel_motif_dual_scale", **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_attention_layers = int(num_attention_layers)
        self.num_heads = int(num_heads)
        self.num_motifs = int(num_motifs)
        self.spatial_span = float(spatial_span)
        self.use_motif_graph = bool(use_motif_graph)
        self.num_motif_gnn_layers = int(num_motif_gnn_layers)
        self.num_motif_heads = int(num_motif_heads)
        self.temperature = float(temperature)
        self.dropout_rate = float(dropout)
        self.num_classes = int(num_classes)

        # 1. Pixel node projection [5 -> D]
        self.node_proj = tf.keras.layers.Dense(self.hidden_dim, name="node_projection")
        self.node_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # 2. Multi-Head Local Neighbor Attention layers
        self.pixel_layers = [
            MultiHeadLocalNeighborAttentionLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                edge_dim=3,
                dropout=self.dropout_rate,
                name=f"multihead_pixel_layer_{i}",
            )
            for i in range(self.num_attention_layers)
        ]

        # 3. Spatial Motif Clustering [2304 -> K]
        self.clustering_layer = SpatialLearnedMotifClustering(
            num_motifs=self.num_motifs,
            hidden_dim=self.hidden_dim,
            temperature=self.temperature,
            use_spatial=True,
            spatial_span=self.spatial_span,
            name="spatial_motif_clustering",
        )

        # 4. Motif GNN layers (macro structural reasoning with multi-head attention)
        if self.num_motif_heads > 1:
            self.motif_gnn_layers = [
                MultiHeadMotifGNNLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_motif_heads,
                    dropout=self.dropout_rate,
                    name=f"multihead_motif_gnn_{i}",
                )
                for i in range(self.num_motif_gnn_layers)
            ]
        else:
            self.motif_gnn_layers = [
                MotifGNNLayer(
                    hidden_dim=self.hidden_dim,
                    dropout=self.dropout_rate,
                    name=f"motif_gnn_{i}",
                )
                for i in range(self.num_motif_gnn_layers)
            ]

        # 5. Dual-Scale Readout:
        # 5a. Macro scale: Motif Attention Pooling
        self.motif_pooling = MotifAttentionPooling(
            hidden_dim=self.hidden_dim,
            name="motif_attention_pooling",
        )
        # 5b. Micro scale: Pixel Global Attention Pooling
        self.pixel_pooling = GlobalPixelAttentionPooling(
            hidden_dim=self.hidden_dim,
            name="pixel_global_pooling",
        )

        # 5c. Dual-Scale Fusion MLP: [2*D -> D]
        self.fusion_dense = tf.keras.layers.Dense(self.hidden_dim, name="fusion_dense")
        self.fusion_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="fusion_norm")
        self.fusion_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # 6. Final Classifier: [D -> 7]
        self.classifier = tf.keras.layers.Dense(self.num_classes, name="classifier")

    def call(self, batch, training: bool = False):
        x = batch["node_features"]
        if len(x.shape) == 2:
            x = tf.reshape(x, [-1, 2304, 5])

        # Step 1: Node projection
        h = tf.nn.gelu(self.node_proj(x))
        if training and self.dropout_rate > 0.0:
            h = self.node_dropout(h, training=training)

        # Step 2: Multi-Head Pixel GNN message passing
        for layer in self.pixel_layers:
            h = layer(h, training=training)

        pixel_embeddings = h  # [B, 2304, D]

        # Step 3: Spatial Motif Clustering
        motif_reps, assignment, p_norm, A_motif = self.clustering_layer(pixel_embeddings)
        motif_usage = tf.reduce_mean(assignment, axis=[0, 1])

        # Step 4: Motif GNN reasoning on coarsened graph
        z = motif_reps
        last_motif_attn = None
        for m_gnn in self.motif_gnn_layers:
            z, last_motif_attn = m_gnn(z, A_motif, training=training)

        # Step 5: Dual-Scale Readout
        # 5a. Macro: Motif pooled vector [B, D]
        z_motif, motif_attn_weights = self.motif_pooling(z)

        # 5b. Micro: Pixel pooled vector [B, D]
        z_pixel, pixel_attn_weights = self.pixel_pooling(pixel_embeddings)

        # 5c. Dual-scale fusion
        z_fused = tf.concat([z_motif, z_pixel], axis=-1)  # [B, 2*D]
        z_fused = tf.nn.gelu(self.fusion_dense(z_fused))
        z_fused = self.fusion_norm(z_fused)
        if training and self.dropout_rate > 0.0:
            z_fused = self.fusion_dropout(z_fused, training=training)

        # Step 6: Classification
        logits = self.classifier(z_fused)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int64)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "z_image": z_fused,
            "z_motif": z_motif,
            "z_pixel": z_pixel,
            "motif_assignment": assignment,
            "motif_prototypes": p_norm,
            "motif_spatial_centers": self.clustering_layer.spatial_centers,
            "A_motif": A_motif,
            "motif_gnn_attention": last_motif_attn,
            "motif_attention_weights": motif_attn_weights,
            "pixel_attention_weights": pixel_attn_weights,
            "motif_usage": motif_usage,
        }
