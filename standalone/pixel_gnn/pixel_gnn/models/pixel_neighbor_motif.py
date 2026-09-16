"""Pure Pixel Neighbor Attention + Learned Motif Prototypes Model."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from pixel_gnn.grid import StaticGridTopology


def init_grid_centers(num_motifs: int, span: float = 0.58) -> np.ndarray:
    """Initialize K motif spatial centers symmetrically across [-span, span]^2.
    
    Default span=0.58 maps normalized coordinates to pixel indices [10, 37] on a 48x48 image,
    tightly focusing motif capacity on core facial structures (brows, eyes, nose, mouth)
    and avoiding background / hair boundary artifacts.
    """
    if num_motifs == 16:
        grid_h, grid_w = 4, 4
        ys = np.linspace(-span, span, grid_h, dtype=np.float32)
        xs = np.linspace(-span, span, grid_w, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        centers = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        return centers.astype(np.float32)
    elif num_motifs == 32:
        # 6x6 grid with the 4 extreme outer corners removed -> 32 motifs in an octagonal face oval
        grid_h, grid_w = 6, 6
        ys = np.linspace(-span, span, grid_h, dtype=np.float32)
        xs = np.linspace(-span, span, grid_w, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        centers_all = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        corner_indices = {0, 5, 30, 35}
        centers_32 = [centers_all[i] for i in range(36) if i not in corner_indices]
        return np.array(centers_32, dtype=np.float32)
    elif num_motifs == 36:
        # Perfectly symmetric 6x6 facial grid
        grid_h, grid_w = 6, 6
        ys = np.linspace(-span, span, grid_h, dtype=np.float32)
        xs = np.linspace(-span, span, grid_w, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        centers = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        return centers.astype(np.float32)
    elif num_motifs == 64:
        grid_h, grid_w = 8, 8
        ys = np.linspace(-span, span, grid_h, dtype=np.float32)
        xs = np.linspace(-span, span, grid_w, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        centers = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        return centers.astype(np.float32)
    else:
        grid_w = int(np.ceil(np.sqrt(num_motifs)))
        grid_h = int(np.ceil(num_motifs / grid_w))
        ys = np.linspace(-span, span, grid_h, dtype=np.float32)
        xs = np.linspace(-span, span, grid_w, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        centers = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        return centers[:num_motifs].astype(np.float32)


class LocalNeighborAttentionLayer(tf.keras.layers.Layer):
    """Vectorized local attention over static 8-neighborhood with edge-aware message passing."""

    def __init__(self, hidden_dim: int = 64, edge_dim: int = 3, dropout: float = 0.1, name: str = None):
        super().__init__(name=name)
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.dropout_rate = float(dropout)
        self.score_hidden = max(self.hidden_dim // 2, 16)

        # Fast projected attention (W [h_i, h_j, e] = W_i(h_i) + W_j(h_j) + W_e(e))
        self.score_proj_i = tf.keras.layers.Dense(self.score_hidden, name="score_proj_i")
        self.score_proj_j = tf.keras.layers.Dense(self.score_hidden, name="score_proj_j")
        self.score_proj_e = tf.keras.layers.Dense(self.score_hidden, name="score_proj_e")
        self.score_out = tf.keras.layers.Dense(1, name="score_out")

        # Edge-aware value projection
        self.value_dense = tf.keras.layers.Dense(self.hidden_dim, name="value_dense")
        self.value_edge_dense = tf.keras.layers.Dense(self.hidden_dim, name="value_edge_dense")

        # Normalization and residual
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)

        # Lightweight FFN
        self.ffn_dense1 = tf.keras.layers.Dense(self.hidden_dim * 2, activation="gelu", name="ffn_dense1")
        self.ffn_dense2 = tf.keras.layers.Dense(self.hidden_dim, name="ffn_dense2")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, h, training: bool = False):
        grid = StaticGridTopology.get_instance()
        neighbors_idx = grid.neighbors_idx      # [2304, 8]
        neighbor_valid = grid.neighbor_valid    # [2304, 8]
        static_edge = grid.static_edge_features  # [2304, 8, 3]

        batch_size = tf.shape(h)[0]
        num_nodes = tf.shape(h)[1]  # 2304

        # 1. Projections computed at node level (2304 nodes, NOT tiled) -> 8x fewer operations!
        proj_i = self.score_proj_i(h)  # [B, 2304, score_hidden]
        proj_j = self.score_proj_j(h)  # [B, 2304, score_hidden]

        # 2. Gather neighbor projections: [B, 2304, 8, score_hidden]
        proj_j_gathered = tf.gather(proj_j, neighbors_idx, axis=1)

        # 3. Expand self node projection: [B, 2304, 1, score_hidden]
        proj_i_expanded = tf.expand_dims(proj_i, axis=2)

        # 4. Static edge projection: [1, 2304, 8, score_hidden]
        proj_e = tf.expand_dims(self.score_proj_e(static_edge), axis=0)

        # 5. Additive attention score with GELU activation
        pair_features = tf.nn.gelu(proj_i_expanded + proj_j_gathered + proj_e)  # [B, 2304, 8, score_hidden]
        scores = tf.squeeze(self.score_out(pair_features), axis=-1)            # [B, 2304, 8]

        # 6. Mask boundary / invalid neighbors
        mask_tiled = tf.broadcast_to(tf.expand_dims(neighbor_valid, axis=0), [batch_size, num_nodes, 8])
        masked_scores = tf.where(mask_tiled, scores, tf.constant(-1e9, dtype=scores.dtype))

        # 7. Softmax attention weights over 8 neighbors
        alpha = tf.nn.softmax(masked_scores, axis=-1)  # [B, 2304, 8]
        alpha = tf.where(mask_tiled, alpha, tf.zeros_like(alpha))
        if training and self.dropout_rate > 0.0:
            alpha = self.dropout1(alpha, training=training)

        # 8. True edge-aware value projection: v_ij = W_v(h_j) + W_ve(e_ij)
        v_nodes = self.value_dense(h)                                         # [B, 2304, D]
        v_neighbors = tf.gather(v_nodes, neighbors_idx, axis=1)               # [B, 2304, 8, D]
        v_edges = tf.expand_dims(self.value_edge_dense(static_edge), axis=0)  # [1, 2304, 8, D]
        v_total = v_neighbors + v_edges                                       # [B, 2304, 8, D]

        # 9. Aggregate message: sum_j alpha_ij * (W_v(h_j) + W_ve(e_ij))
        message = tf.reduce_sum(tf.expand_dims(alpha, axis=-1) * v_total, axis=2)  # [B, 2304, D]

        # 9. Residual + LayerNorm
        h_res = self.norm1(h + message)

        # 10. Lightweight Feed-Forward Block
        ffn = self.ffn_dense2(self.ffn_dense1(h_res))
        if training and self.dropout_rate > 0.0:
            ffn = self.dropout2(ffn, training=training)
        h_out = self.norm2(h_res + ffn)

        return h_out


class SpatialLearnedMotifClustering(tf.keras.layers.Layer):
    """Learned motif clustering with spatial awareness and vectorized graph coarsening.
    
    Assigns 2304 pixel nodes to K motifs based on:
        similarity = <h_i, p_k> / tau - gamma * ||(x_i, y_i) - c_k||^2 / tau
    
    Outputs:
        motif_reps: [B, K, D]
        assignment (S): [B, 2304, K]
        p_norm: [K, D]
        A_motif: [B, K, K] coarsened adjacency (S^T A_pixel S)
    """

    def __init__(
        self,
        num_motifs: int = 32,
        hidden_dim: int = 64,
        temperature: float = 0.1,
        use_spatial: bool = True,
        spatial_span: float = 0.58,
        name: str = None,
    ):
        super().__init__(name=name)
        self.num_motifs = int(num_motifs)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.use_spatial = bool(use_spatial)
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

        # Learnable spatial scale penalty gamma
        self.spatial_scale = self.add_weight(
            name="spatial_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, h):
        grid = StaticGridTopology.get_instance()
        coords = grid.normalized_coords  # [2304, 2]

        # 1. Feature cosine similarity: [B, 2304, K]
        h_norm = tf.math.l2_normalize(h, axis=-1, epsilon=1e-6)                # [B, 2304, D]
        p_norm = tf.math.l2_normalize(self.prototypes, axis=-1, epsilon=1e-6)  # [K, D]
        sim_feat = tf.einsum("bnd,kd->bnk", h_norm, p_norm)                    # [B, 2304, K]

        # 2. Spatial distance penalty: ||pos_i - center_k||^2
        if self.use_spatial:
            coords_exp = tf.expand_dims(coords, axis=1)                        # [2304, 1, 2]
            centers_exp = tf.expand_dims(self.spatial_centers, axis=0)         # [1, K, 2]
            dist_sq = tf.reduce_sum(tf.square(coords_exp - centers_exp), axis=-1)  # [2304, K]
            dist_batch = tf.expand_dims(dist_sq, axis=0)                       # [1, 2304, K]
            gamma = tf.nn.softplus(self.spatial_scale)
            logits = (sim_feat - gamma * dist_batch) / self.temperature
        else:
            logits = sim_feat / self.temperature

        # 3. Soft assignment S: [B, 2304, K]
        assignment = tf.nn.softmax(logits, axis=-1)                            # [B, 2304, K]

        # 4. Motif representations Z: [B, K, D]
        motif_unnorm = tf.einsum("bnk,bnd->bkd", assignment, h)               # [B, K, D]
        motif_weights = tf.reduce_sum(assignment, axis=1, keepdims=True)       # [B, 1, K]
        motif_weights = tf.transpose(motif_weights, [0, 2, 1])                 # [B, K, 1]
        motif_reps = motif_unnorm / (motif_weights + 1e-6)                     # [B, K, D]

        # 5. Vectorized Motif Graph Coarsening: A_motif = S^T A_pixel S: [B, K, K]
        neighbors_idx = grid.neighbors_idx        # [2304, 8]
        neighbor_valid = grid.neighbor_valid      # [2304, 8]

        # Gather S on neighbor indices: [B, 2304, 8, K]
        S_nbr = tf.gather(assignment, neighbors_idx, axis=1)
        mask = tf.expand_dims(tf.expand_dims(neighbor_valid, axis=0), axis=-1)  # [1, 2304, 8, 1]
        S_nbr_masked = tf.where(mask, S_nbr, tf.zeros_like(S_nbr))              # [B, 2304, 8, K]

        # S_agg = A_pixel @ S: [B, 2304, K]
        S_agg = tf.reduce_sum(S_nbr_masked, axis=2)

        # A_motif = S^T @ (A_pixel @ S): [B, K, K]
        A_motif = tf.einsum("bni,bnj->bij", assignment, S_agg)

        return motif_reps, assignment, p_norm, A_motif


# Alias for backward compatibility if imported elsewhere
LearnedMotifPrototypeLayer = SpatialLearnedMotifClustering


class MotifGNNLayer(tf.keras.layers.Layer):
    """Edge-aware Motif Graph Attention Layer operating on the coarsened K-node graph.
    
    Attention between motifs incorporates topological connectivity bias:
        score_kl = (q_k * k_l) / sqrt(d) + W_e * A_motif_norm_kl
    Topologically connected motifs receive a learned affinity boost, while distant
    motifs (A_norm = 0) have bias = 0 and interact purely via semantic dot-product attention.
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1, name: str = None):
        super().__init__(name=name)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout)

        self.q_dense = tf.keras.layers.Dense(self.hidden_dim, name="motif_q")
        self.k_dense = tf.keras.layers.Dense(self.hidden_dim, name="motif_k")
        self.v_dense = tf.keras.layers.Dense(self.hidden_dim, name="motif_v")
        self.edge_proj = tf.keras.layers.Dense(
            1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Constant(0.5),
            name="motif_edge_proj",
        )

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="motif_norm1")
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)

        self.ffn1 = tf.keras.layers.Dense(self.hidden_dim * 2, activation="gelu", name="motif_ffn1")
        self.ffn2 = tf.keras.layers.Dense(self.hidden_dim, name="motif_ffn2")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="motif_norm2")
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, z, A_motif, training: bool = False):
        # z: [B, K, D], A_motif: [B, K, K]
        q = self.q_dense(z)  # [B, K, D]
        k = self.k_dense(z)  # [B, K, D]
        v = self.v_dense(z)  # [B, K, D]

        scale = 1.0 / tf.math.sqrt(tf.cast(self.hidden_dim, tf.float32))
        node_sim = tf.matmul(q, k, transpose_b=True) * scale  # [B, K, K]

        # Normalize A_motif to row-stochastic [0, 1]
        deg = tf.reduce_sum(A_motif, axis=-1, keepdims=True) + 1e-6  # [B, K, 1]
        A_norm = A_motif / deg                                       # [B, K, K]
        
        # Linear edge bias: connected motifs get an affinity boost, distant motifs get 0 bias
        edge_bias = tf.squeeze(self.edge_proj(tf.expand_dims(A_norm, axis=-1)), axis=-1)  # [B, K, K]

        scores = node_sim + edge_bias  # [B, K, K]
        attn_weights = tf.nn.softmax(scores, axis=-1)  # [B, K, K]

        if training and self.dropout_rate > 0.0:
            attn_weights = self.dropout1(attn_weights, training=training)

        msg = tf.matmul(attn_weights, v)  # [B, K, D]
        z_res = self.norm1(z + msg)

        ffn = self.ffn2(self.ffn1(z_res))
        if training and self.dropout_rate > 0.0:
            ffn = self.dropout2(ffn, training=training)
        z_out = self.norm2(z_res + ffn)

        return z_out, attn_weights


class MotifAttentionPooling(tf.keras.layers.Layer):
    """Lightweight attention pooling over K motif representations."""

    def __init__(self, hidden_dim: int = 64, name: str = None):
        super().__init__(name=name)
        self.score_linear = tf.keras.layers.Dense(1, name="motif_score_linear")

    def call(self, motif_reps):
        scores = self.score_linear(motif_reps)
        beta = tf.nn.softmax(scores, axis=1)
        graph_embedding = tf.reduce_sum(beta * motif_reps, axis=1)
        return graph_embedding, tf.squeeze(beta, axis=-1)


@tf.keras.utils.register_keras_serializable(package="pixel_gnn")
class PixelNeighborMotifModel(tf.keras.Model):
    """End-to-end Hierarchical Architecture:
    
    Pixel Node [B, 2304, 5]
    -> Edge-aware Pixel GNN [B, 2304, 64]
    -> Spatial-aware Learned Clustering [B, 2304, K] -> Z [B, K, 64]
    -> Coarsened Motif Graph A_motif [B, K, K]
    -> Edge-aware Motif GNN [B, K, 64]
    -> Motif Attention Pooling [B, 64]
    -> Classifier [B, 7]
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_attention_layers: int = 1,
        num_motifs: int = 32,
        use_motif_graph: bool = True,
        num_motif_gnn_layers: int = 1,
        temperature: float = 0.1,
        pooling_type: str = "motif",
        dropout: float = 0.1,
        num_classes: int = 7,
        **kwargs,
    ):
        super().__init__(name="pixel_neighbor_motif", **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_attention_layers = int(num_attention_layers)
        self.num_motifs = int(num_motifs)
        self.use_motif_graph = bool(use_motif_graph)
        self.num_motif_gnn_layers = int(num_motif_gnn_layers)
        self.temperature = float(temperature)
        self.pooling_type = str(pooling_type)
        self.dropout_rate = float(dropout)
        self.num_classes = int(num_classes)

        # 1. Pixel node input projection
        self.node_proj = tf.keras.layers.Dense(self.hidden_dim, name="node_projection")
        self.node_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # 2. Edge-aware Pixel GNN layers
        self.attention_layers = [
            LocalNeighborAttentionLayer(
                hidden_dim=self.hidden_dim,
                edge_dim=3,
                dropout=self.dropout_rate,
                name=f"neighbor_attention_{i}",
            )
            for i in range(self.num_attention_layers)
        ]

        # 3. Spatial Motif Clustering
        if self.pooling_type == "motif":
            self.clustering_layer = SpatialLearnedMotifClustering(
                num_motifs=self.num_motifs,
                hidden_dim=self.hidden_dim,
                temperature=self.temperature,
                use_spatial=True,
                name="spatial_motif_clustering",
            )
            # 4. Motif GNN layers (if use_motif_graph is True)
            if self.use_motif_graph and self.num_motif_gnn_layers > 0:
                self.motif_gnn_layers = [
                    MotifGNNLayer(
                        hidden_dim=self.hidden_dim,
                        dropout=self.dropout_rate,
                        name=f"motif_gnn_{i}",
                    )
                    for i in range(self.num_motif_gnn_layers)
                ]
            else:
                self.motif_gnn_layers = []

            # 5. Global Motif Pooling
            self.motif_pooling = MotifAttentionPooling(
                hidden_dim=self.hidden_dim,
                name="motif_attention_pooling",
            )
        else:
            self.clustering_layer = None
            self.motif_gnn_layers = []
            self.motif_pooling = None

        # 6. Classifier
        self.classifier = tf.keras.layers.Dense(self.num_classes, name="classifier")

    def call(self, batch, training: bool = False):
        x = batch["node_features"]
        if len(x.shape) == 2:
            x = tf.reshape(x, [-1, 2304, 5])

        # Step 1: Project pixel node features
        h = tf.nn.gelu(self.node_proj(x), approximate=False)
        if training and self.dropout_rate > 0.0:
            h = self.node_dropout(h, training=training)

        # Step 2: Edge-aware Pixel GNN message passing
        for att_layer in self.attention_layers:
            h = att_layer(h, training=training)

        node_embeddings = h

        # Step 3 & 4: Motif Clustering & Motif Graph
        if self.pooling_type == "motif":
            motif_reps, assignment, p_norm, A_motif = self.clustering_layer(node_embeddings)
            motif_usage = tf.reduce_mean(assignment, axis=[0, 1])

            # Pass through Motif GNN layers if enabled
            z = motif_reps
            last_motif_attn = None
            for m_gnn in self.motif_gnn_layers:
                z, last_motif_attn = m_gnn(z, A_motif, training=training)

            # Step 5: Global pooling
            z_image, beta = self.motif_pooling(z)
        else:
            z_image = tf.reduce_mean(node_embeddings, axis=1)
            assignment = None
            p_norm = None
            A_motif = None
            last_motif_attn = None
            beta = None
            motif_usage = None

        # Step 6: Classification
        logits = self.classifier(z_image)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int64)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "z_image": z_image,
            "node_embeddings": node_embeddings,
            "motif_assignment": assignment,
            "motif_prototypes": p_norm,
            "motif_spatial_centers": self.clustering_layer.spatial_centers if self.clustering_layer else None,
            "A_motif": A_motif,
            "motif_gnn_attention": last_motif_attn,
            "motif_attention_weights": beta,
            "motif_usage": motif_usage,
        }
