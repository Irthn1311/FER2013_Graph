"""Pure Pixel Neighbor Attention + Learned Motif Prototypes Model."""

from __future__ import annotations

import tensorflow as tf

from pixel_gnn.grid import StaticGridTopology


class LocalNeighborAttentionLayer(tf.keras.layers.Layer):
    """Vectorized local attention over static 8-neighborhood with fast projected additive attention."""

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

        # Value projection
        self.value_dense = tf.keras.layers.Dense(self.hidden_dim, name="value_dense")

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

        # 8. Value projection computed at node level (2304 nodes, NOT tiled) -> 8x faster!
        v_nodes = self.value_dense(h)                            # [B, 2304, D]
        v_neighbors = tf.gather(v_nodes, neighbors_idx, axis=1)  # [B, 2304, 8, D]
        message = tf.reduce_sum(tf.expand_dims(alpha, axis=-1) * v_neighbors, axis=2)  # [B, 2304, D]

        # 9. Residual + LayerNorm
        h_res = self.norm1(h + message)

        # 10. Lightweight Feed-Forward Block
        ffn = self.ffn_dense2(self.ffn_dense1(h_res))
        if training and self.dropout_rate > 0.0:
            ffn = self.dropout2(ffn, training=training)
        h_out = self.norm2(h_res + ffn)

        return h_out


class LearnedMotifPrototypeLayer(tf.keras.layers.Layer):
    """Learned motif prototypes and soft assignment mechanics."""

    def __init__(self, num_motifs: int = 32, hidden_dim: int = 64, temperature: float = 0.1, name: str = None):
        super().__init__(name=name)
        self.num_motifs = int(num_motifs)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)

    def build(self, input_shape):
        self.prototypes = self.add_weight(
            name="motif_prototypes",
            shape=(self.num_motifs, self.hidden_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, h):
        h_norm = tf.math.l2_normalize(h, axis=-1, epsilon=1e-6)              # [B, 2304, D]
        p_norm = tf.math.l2_normalize(self.prototypes, axis=-1, epsilon=1e-6)  # [K, D]

        similarity = tf.einsum("bnd,kd->bnk", h_norm, p_norm)
        assignment = tf.nn.softmax(similarity / self.temperature, axis=-1)  # [B, 2304, K]

        motif_unnorm = tf.einsum("bnk,bnd->bkd", assignment, h)
        motif_weights = tf.reduce_sum(assignment, axis=1, keepdims=True)     # [B, 1, K]
        motif_weights = tf.transpose(motif_weights, [0, 2, 1])               # [B, K, 1]

        motif_reps = motif_unnorm / (motif_weights + 1e-6)                   # [B, K, D]

        return motif_reps, assignment, p_norm


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
    """End-to-end model: Pixel Node -> Neighbor Attention -> Motif Prototypes -> Classifier."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_attention_layers: int = 1,
        num_motifs: int = 32,
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
        self.temperature = float(temperature)
        self.pooling_type = str(pooling_type)
        self.dropout_rate = float(dropout)
        self.num_classes = int(num_classes)

        self.node_proj = tf.keras.layers.Dense(self.hidden_dim, name="node_projection")
        self.node_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.attention_layers = [
            LocalNeighborAttentionLayer(
                hidden_dim=self.hidden_dim,
                edge_dim=3,
                dropout=self.dropout_rate,
                name=f"neighbor_attention_{i}",
            )
            for i in range(self.num_attention_layers)
        ]

        if self.pooling_type == "motif":
            self.motif_layer = LearnedMotifPrototypeLayer(
                num_motifs=self.num_motifs,
                hidden_dim=self.hidden_dim,
                temperature=self.temperature,
                name="learned_motifs",
            )
            self.motif_pooling = MotifAttentionPooling(
                hidden_dim=self.hidden_dim,
                name="motif_attention_pooling",
            )
        else:
            self.motif_layer = None
            self.motif_pooling = None

        self.classifier = tf.keras.layers.Dense(self.num_classes, name="classifier")

    def call(self, batch, training: bool = False):
        x = batch["node_features"]
        if len(x.shape) == 2:
            x = tf.reshape(x, [-1, 2304, 5])

        h = tf.nn.gelu(self.node_proj(x), approximate=False)
        if training and self.dropout_rate > 0.0:
            h = self.node_dropout(h, training=training)

        for att_layer in self.attention_layers:
            h = att_layer(h, training=training)

        node_embeddings = h

        if self.pooling_type == "motif":
            motif_reps, assignment, p_norm = self.motif_layer(node_embeddings)
            z_image, beta = self.motif_pooling(motif_reps)
            motif_usage = tf.reduce_mean(assignment, axis=[0, 1])
        else:
            z_image = tf.reduce_mean(node_embeddings, axis=1)
            assignment = None
            p_norm = None
            beta = None
            motif_usage = None

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
            "motif_attention_weights": beta,
            "motif_usage": motif_usage,
        }
