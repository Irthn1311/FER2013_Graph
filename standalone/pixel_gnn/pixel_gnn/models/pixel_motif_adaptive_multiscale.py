"""Pure Pixel Adaptive Multi-Scale Hierarchical Motif GNN for Facial Expression Recognition.

Pure Pixel Architecture Pipeline (No CNNs, No Sliding Windows):
  1. Pure Pixel Node Input [B, 2304, node_dim] -> Projected [B, 2304, D] (node_dim=7 or 9)
  2. Multi-Head Local Neighbor Attention GNN (3 layers, 4 heads, edge-aware over 8-grid).
  3. Tier 1: Adaptive Anisotropic Micro-Motif Clustering (2304 pixels -> K1=49 Micro Motifs):
     Learns per-motif spatial centers (cx_k, cy_k) and anisotropic spatial spreads (sigma_x, sigma_y).
     Outputs coarsened Micro Motif Graph A_micro [B, 49, 49] and Micro Reps Z_micro [B, 49, D].
  4. Micro Motif GNN (2 layers):
     Relational message passing between local facial micro regions.
  5. Tier 2: Adaptive Anisotropic Macro-Motif Clustering (49 Micro Motifs -> K2=9 Macro Landmarks):
     Coarsens micro motifs into 9 major facial landmark structures (Brows, Eyes, Nose, Mouth, Cheeks, Jaw).
     Outputs Macro Motif Graph A_macro [B, 9, 9] and Macro Reps Z_macro [B, 9, D].
  6. Macro Motif GNN (2 layers):
     High-level structural reasoning across major facial landmarks.
  7. Multi-Scale Tri-Scale Readout:
     - Macro Scale: Motif Attention Pooling z_macro [B, D]
     - Micro Scale: Motif Attention Pooling z_micro [B, D]
     - Pixel Scale: Global Pixel Attention Pooling z_pixel [B, D]
     - Tri-Scale Fusion: Concat[z_macro, z_micro, z_pixel] -> GELU -> LayerNorm -> Dropout -> z_fused [B, D]
  8. Classifier [B, 7]
"""

from __future__ import annotations

import math
import numpy as np
import tensorflow as tf

from pixel_gnn.grid import StaticGridTopology
from pixel_gnn.models.pixel_motif_dual_scale import (
    MultiHeadLocalNeighborAttentionLayer,
    GlobalPixelAttentionPooling,
)
from pixel_gnn.models.pixel_neighbor_motif import (
    init_grid_centers,
    MultiHeadMotifGNNLayer,
    MotifAttentionPooling,
)


class AdaptiveAnisotropicMotifClustering(tf.keras.layers.Layer):
    """Adaptive Motif Clustering with per-motif learnable 2D anisotropic spatial covariance.
    
    Assigns nodes to K motifs based on:
        similarity = <h_i, p_k> / tau - [ (x_i - c_{x,k})^2 / (2 * sigma_{x,k}^2) + (y_i - c_{y,k})^2 / (2 * sigma_{y,k}^2) ] / tau
    
    Allows horizontal stretching for mouth/brow motifs and compact radial clustering for eye motifs.
    """

    def __init__(
        self,
        num_motifs: int = 49,
        hidden_dim: int = 96,
        temperature: float = 0.1,
        spatial_span: float = 0.58,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.num_motifs = int(num_motifs)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.spatial_span = float(spatial_span)

    def build(self, input_shape):
        # Semantic feature prototypes [K, D]
        self.prototypes = self.add_weight(
            name="motif_prototypes",
            shape=(self.num_motifs, self.hidden_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=tf.float32,
        )

        # Learnable 2D spatial centers [K, 2] initialized quasi-uniformly
        init_centers = init_grid_centers(self.num_motifs, span=self.spatial_span)
        self.spatial_centers = self.add_weight(
            name="motif_spatial_centers",
            shape=(self.num_motifs, 2),
            initializer=tf.keras.initializers.Constant(init_centers),
            trainable=True,
            dtype=tf.float32,
        )

        # Learnable anisotropic log-spreads [K, 2] initialized to softplus(0.5) ~ 0.97 radius
        self.spatial_log_sigma = self.add_weight(
            name="spatial_log_sigma",
            shape=(self.num_motifs, 2),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, h, coords: tf.Tensor | None = None):
        if coords is None:
            grid = StaticGridTopology.get_instance()
            coords = grid.normalized_coords  # [2304, 2]

        # 1. Feature cosine similarity: [B, N, K]
        h_norm = tf.math.l2_normalize(h, axis=-1, epsilon=1e-6)                # [B, N, D]
        p_norm = tf.math.l2_normalize(self.prototypes, axis=-1, epsilon=1e-6)  # [K, D]
        sim_feat = tf.einsum("bnd,kd->bnk", h_norm, p_norm)                    # [B, N, K]

        # 2. Anisotropic spatial Mahalanobis-like distance penalty
        # sigmas: [K, 2] > 0.02
        sigmas = tf.nn.softplus(self.spatial_log_sigma) + 0.02                 # [K, 2]
        
        # diff: coords [N, 1, 2] - centers [1, K, 2] -> [N, K, 2]
        coords_exp = tf.expand_dims(coords, axis=1)                            # [N, 1, 2]
        centers_exp = tf.expand_dims(self.spatial_centers, axis=0)             # [1, K, 2]
        diff = coords_exp - centers_exp                                        # [N, K, 2]
        
        # Scaled squared distance: sum_d (diff^2 / (2 * sigma_d^2))
        inv_two_sigma_sq = 1.0 / (2.0 * tf.square(sigmas) + 1e-6)              # [K, 2]
        dist_mah = tf.reduce_sum(tf.square(diff) * tf.expand_dims(inv_two_sigma_sq, axis=0), axis=-1)  # [N, K]
        dist_batch = tf.expand_dims(dist_mah, axis=0)                          # [1, N, K]

        # 3. Soft assignment logits & Softmax S: [B, N, K]
        logits = (sim_feat - dist_batch) / self.temperature
        assignment = tf.nn.softmax(logits, axis=-1)                            # [B, N, K]

        # 4. Coarsened Motif Representations Z: [B, K, D]
        motif_unnorm = tf.einsum("bnk,bnd->bkd", assignment, h)               # [B, K, D]
        motif_weights = tf.reduce_sum(assignment, axis=1, keepdims=True)       # [B, 1, K]
        motif_weights = tf.transpose(motif_weights, [0, 2, 1])                 # [B, K, 1]
        motif_reps = motif_unnorm / (motif_weights + 1e-6)                     # [B, K, D]

        return motif_reps, assignment, p_norm, dist_batch


class PixelMotifAdaptiveMultiScaleModel(tf.keras.Model):
    """Pure Pixel 2-Tier Hierarchical Adaptive Motif GNN with InfoNCE Loss support."""

    def __init__(
        self,
        hidden_dim: int = 96,
        num_attention_layers: int = 3,
        num_heads: int = 4,
        num_micro_motifs: int = 49,
        num_macro_motifs: int = 9,
        spatial_span: float = 0.58,
        num_motif_gnn_layers: int = 2,
        num_motif_heads: int = 4,
        ffn_expansion: int = 2,
        temperature: float = 0.1,
        dropout: float = 0.20,
        num_classes: int = 7,
        **kwargs,
    ):
        super().__init__(name="pixel_motif_adaptive_multiscale", **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_attention_layers = int(num_attention_layers)
        self.num_heads = int(num_heads)
        self.num_micro_motifs = int(num_micro_motifs)
        self.num_macro_motifs = int(num_macro_motifs)
        self.spatial_span = float(spatial_span)
        self.num_motif_gnn_layers = int(num_motif_gnn_layers)
        self.num_motif_heads = int(num_motif_heads)
        self.ffn_expansion = int(ffn_expansion)
        self.temperature = float(temperature)
        self.dropout_rate = float(dropout)
        self.num_classes = int(num_classes)

        # 1. Pure Pixel Node projection [node_dim -> D]
        self.node_proj = tf.keras.layers.Dense(self.hidden_dim, name="node_projection")
        self.node_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # 2. Multi-Head Local Pixel Neighbor Attention layers over 8-grid
        self.pixel_layers = [
            MultiHeadLocalNeighborAttentionLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                edge_dim=3,
                dropout=self.dropout_rate,
                ffn_expansion=self.ffn_expansion,
                name=f"multihead_pixel_layer_{i}",
            )
            for i in range(self.num_attention_layers)
        ]

        # 3. Tier 1: Adaptive Anisotropic Micro-Motif Clustering (2304 pixels -> K1=49)
        self.micro_clustering = AdaptiveAnisotropicMotifClustering(
            num_motifs=self.num_micro_motifs,
            hidden_dim=self.hidden_dim,
            temperature=self.temperature,
            spatial_span=self.spatial_span,
            name="adaptive_micro_clustering",
        )

        # 4. Micro Motif GNN layers (2 layers)
        self.micro_gnn_layers = [
            MultiHeadMotifGNNLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_motif_heads,
                dropout=self.dropout_rate,
                ffn_expansion=self.ffn_expansion,
                name=f"micro_motif_gnn_{i}",
            )
            for i in range(self.num_motif_gnn_layers)
        ]

        # 5. Tier 2: Adaptive Anisotropic Macro-Motif Clustering (49 Micro Motifs -> K2=9 Macro Landmarks)
        self.macro_clustering = AdaptiveAnisotropicMotifClustering(
            num_motifs=self.num_macro_motifs,
            hidden_dim=self.hidden_dim,
            temperature=self.temperature,
            spatial_span=self.spatial_span,
            name="adaptive_macro_clustering",
        )

        # 6. Macro Motif GNN layers (2 layers)
        self.macro_gnn_layers = [
            MultiHeadMotifGNNLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_motif_heads,
                dropout=self.dropout_rate,
                ffn_expansion=self.ffn_expansion,
                name=f"macro_motif_gnn_{i}",
            )
            for i in range(self.num_motif_gnn_layers)
        ]

        # 7. Multi-Scale Tri-Scale Readout:
        self.macro_pooling = MotifAttentionPooling(
            hidden_dim=self.hidden_dim,
            name="macro_attention_pooling",
        )
        self.micro_pooling = MotifAttentionPooling(
            hidden_dim=self.hidden_dim,
            name="micro_attention_pooling",
        )
        self.pixel_pooling = GlobalPixelAttentionPooling(
            hidden_dim=self.hidden_dim,
            name="pixel_global_pooling",
        )

        # Fusion MLP: [3*D -> D]
        self.fusion_dense = tf.keras.layers.Dense(self.hidden_dim, name="fusion_dense")
        self.fusion_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="fusion_norm")
        self.fusion_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # Final Classifier: [D -> 7]
        self.classifier = tf.keras.layers.Dense(self.num_classes, name="classifier")

    def call(self, batch, training: bool = False):
        x = batch["node_features"]
        if len(x.shape) == 2:
            feat_dim = tf.shape(x)[-1]
            x = tf.reshape(x, [-1, 2304, feat_dim])

        # Step 1: Pixel Node Projection
        h = tf.nn.gelu(self.node_proj(x))
        if training and self.dropout_rate > 0.0:
            h = self.node_dropout(h, training=training)

        # Step 2: Pixel GNN Message Passing over 8-grid
        for layer in self.pixel_layers:
            h = layer(h, training=training)
        pixel_embeddings = h  # [B, 2304, D]

        # Step 3: Tier 1 Micro Motif Clustering (2304 pixels -> K1=49 Micro Motifs)
        grid = StaticGridTopology.get_instance()
        coords_pixel = grid.normalized_coords  # [2304, 2]
        
        z_micro_raw, assign_micro, p_micro_norm, _ = self.micro_clustering(pixel_embeddings, coords=coords_pixel)
        micro_usage = tf.reduce_mean(assign_micro, axis=[0, 1])

        # Coarsen Micro Graph Adjacency: A_micro = S1^T A_pixel S1 [B, 49, 49]
        neighbors_idx = grid.neighbors_idx      # [2304, 8]
        neighbor_valid = grid.neighbor_valid    # [2304, 8]
        S1_nbr = tf.gather(assign_micro, neighbors_idx, axis=1)  # [B, 2304, 8, 49]
        valid_mask = tf.cast(neighbor_valid, assign_micro.dtype) # [2304, 8]
        S1_agg = tf.einsum("bnik,ni->bnk", S1_nbr, valid_mask)
        A_micro = tf.einsum("bni,bnj->bij", assign_micro, S1_agg)

        # Step 4: Micro Motif GNN Reasoning
        z_micro = z_micro_raw
        for m_gnn in self.micro_gnn_layers:
            z_micro, _ = m_gnn(z_micro, A_micro, training=training)

        # Step 5: Tier 2 Macro Motif Clustering (49 Micro Motifs -> K2=9 Macro Landmarks)
        coords_micro = self.micro_clustering.spatial_centers  # [49, 2]
        z_macro_raw, assign_macro, p_macro_norm, _ = self.macro_clustering(z_micro, coords=coords_micro)

        # Coarsen Macro Graph Adjacency: A_macro = S2^T A_micro S2 [B, 9, 9]
        A_macro = tf.einsum("bni,bnj->bij", assign_macro, tf.einsum("bnm,bmk->bnk", A_micro, assign_macro))

        # Step 6: Macro Motif GNN Reasoning
        z_macro = z_macro_raw
        last_macro_attn = None
        for M_gnn in self.macro_gnn_layers:
            z_macro, last_macro_attn = M_gnn(z_macro, A_macro, training=training)

        # Step 7: Tri-Scale Readout & Fusion
        z_macro_pooled, macro_attn_weights = self.macro_pooling(z_macro)
        z_micro_pooled, micro_attn_weights = self.micro_pooling(z_micro)
        z_pixel_pooled, pixel_attn_weights = self.pixel_pooling(pixel_embeddings)

        z_tri_fused = tf.concat([z_macro_pooled, z_micro_pooled, z_pixel_pooled], axis=-1)  # [B, 3*D]
        z_fused = tf.nn.gelu(self.fusion_dense(z_tri_fused))
        z_fused = self.fusion_norm(z_fused)
        if training and self.dropout_rate > 0.0:
            z_fused = self.fusion_dropout(z_fused, training=training)

        # Step 8: Classification
        logits = self.classifier(z_fused)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int64)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "z_image": z_fused,
            "z_motif": z_macro_pooled,
            "z_motif_raw": z_macro_raw,
            "z_micro": z_micro_pooled,
            "z_micro_raw": z_micro_raw,
            "z_pixel": z_pixel_pooled,
            "motif_assignment": assign_micro,
            "macro_assignment": assign_macro,
            "motif_prototypes": p_macro_norm,
            "micro_prototypes": p_micro_norm,
            "macro_prototypes": p_macro_norm,
            "motif_spatial_centers": self.micro_clustering.spatial_centers,
            "macro_spatial_centers": self.macro_clustering.spatial_centers,
            "A_motif": A_micro,
            "A_macro": A_macro,
            "motif_gnn_attention": last_macro_attn,
            "motif_attention_weights": macro_attn_weights,
            "pixel_attention_weights": pixel_attn_weights,
            "motif_usage": micro_usage,
        }
