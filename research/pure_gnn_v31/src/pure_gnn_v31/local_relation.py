"""Local Adaptive Relation Block for Pure-GNN v3.1."""

from typing import Dict, Optional
import tensorflow as tf


class LocalAdaptiveRelationBlock(tf.keras.layers.Layer):
    """Local Adaptive Relation Block operating on an 8-neighbor spatial grid graph.
    
    Formula for spatial edge i -> j (receiver i, sender j):
        r_ij = h_j - h_i
        s_ij = MLP_gate([LN(h_i), LN(r_ij), delta_x, delta_y])
        g_ij = 1 + tanh(s_ij)
        m_ij = g_ij * (W_direction[dir_id] h_j + W_relation (h_j - h_i))
        m_i = (1 / degree(i)) * sum_{j in N(i)} m_ij
        h_mid = H + m_i
        h_out = h_mid + FFN(LN(h_mid))
    
    When mask_content=True (G2 condition):
        LN(h_i) and LN(r_ij) are masked to zero before entering MLP_gate.
    """

    def __init__(
        self,
        channels: int,
        gate_hidden_dim: int = 16,
        ffn_expansion: int = 4,
        mask_content: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        self.gate_hidden_dim = gate_hidden_dim
        self.ffn_expansion = ffn_expansion
        self.mask_content = mask_content

        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="norm1")
        self.norm_diff = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="norm_diff")

        # Local Gate MLP: input dimension is 2 * channels + 2
        self.gate_dense1 = tf.keras.layers.Dense(
            gate_hidden_dim,
            activation="gelu",
            kernel_initializer="glorot_uniform",
            name="gate_dense1",
        )
        # Final layer MUST be zero-initialized: s_ij = 0 -> g_ij = 1 + tanh(0) = 1.0
        self.gate_dense2 = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="gate_dense2",
        )

        # Directional weights for 8 relative directions: shape (8, channels, channels)
        self.w_dir = self.add_weight(
            name="w_direction",
            shape=(8, channels, channels),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Relational linear transformation
        self.w_rel = tf.keras.layers.Dense(
            channels,
            use_bias=False,
            kernel_initializer="glorot_uniform",
            name="w_relation",
        )

        # FFN
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="norm2")
        self.ffn_dense1 = tf.keras.layers.Dense(
            channels * ffn_expansion,
            activation="gelu",
            kernel_initializer="glorot_uniform",
            name="ffn_dense1",
        )
        self.ffn_dense2 = tf.keras.layers.Dense(
            channels,
            activation=None,
            kernel_initializer="glorot_uniform",
            name="ffn_dense2",
        )

    def call(
        self,
        h: tf.Tensor,
        graph: Optional[Dict[str, tf.Tensor]] = None,
        training: Optional[bool] = None,
        return_diagnostics: bool = False,
    ):
        """Forward pass.
        
        Args:
            h: (B, N, C) node feature tensor
            graph: dict with src_indices (E,), dst_indices (E,), delta_x (E, 1),
                   delta_y (E, 1), direction_id (E,), degrees (N, 1), num_nodes N
            training: bool
            return_diagnostics: if True, returns (h_out, diag_dict)
        """
        num_nodes = graph["num_nodes"]
        src = graph["src_indices"]
        dst = graph["dst_indices"]
        dx = graph["delta_x"]
        dy = graph["delta_y"]
        dir_id = graph["direction_id"]
        degrees = graph["degrees"]

        # Pre-norm on node states
        h_norm = self.norm1(h)

        # Gather sender and receiver node states
        # Shape: (B, E, C)
        h_j = tf.gather(h_norm, src, axis=1)
        h_i = tf.gather(h_norm, dst, axis=1)
        r_ij = h_j - h_i
        r_ij_norm = self.norm_diff(r_ij)

        # Content inputs for local gate
        if self.mask_content:
            # G2 condition: mask content inputs to exact zeros
            h_i_gate = tf.zeros_like(h_i)
            r_ij_gate = tf.zeros_like(r_ij_norm)
        else:
            h_i_gate = h_i
            r_ij_gate = r_ij_norm

        # Broadcast dx, dy to batch: (B, E, 1)
        b_size = tf.shape(h)[0]
        dx_b = tf.broadcast_to(dx[None, :, :], [b_size, tf.shape(dx)[0], 1])
        dy_b = tf.broadcast_to(dy[None, :, :], [b_size, tf.shape(dy)[0], 1])

        # Gate input: [h_i, r_ij, dx, dy] -> shape (B, E, 2*C + 2)
        gate_input = tf.concat([h_i_gate, r_ij_gate, dx_b, dy_b], axis=-1)
        s_ij = self.gate_dense2(self.gate_dense1(gate_input))  # (B, E, 1)
        g_ij = 1.0 + tf.tanh(s_ij)  # (B, E, 1) in (0, 2)

        # Directional weights for each edge: (E, C, C)
        edge_w_dir = tf.gather(self.w_dir, dir_id)
        # Message direction term: einsum over channel dimension
        # h_j is (B, E, C), edge_w_dir is (E, C, C) -> (B, E, C)
        dir_term = tf.einsum("bec,ecd->bed", h_j, edge_w_dir)
        rel_term = self.w_rel(r_ij)

        m_ij = g_ij * (dir_term + rel_term)  # (B, E, C)

        # Aggregation: sum messages per receiver node i and divide by real degree
        # Transpose to (E, B, C) for unsorted_segment_sum
        m_ij_t = tf.transpose(m_ij, [1, 0, 2])
        m_i_sum_t = tf.math.unsorted_segment_sum(m_ij_t, dst, num_segments=num_nodes)
        m_i_sum = tf.transpose(m_i_sum_t, [1, 0, 2])  # (B, N, C)

        # Real degree normalization
        deg_expanded = degrees[None, :, :]  # (1, N, 1)
        m_i = m_i_sum / deg_expanded  # (B, N, C)

        # Pre-norm residual update
        h_mid = h + m_i

        # Pointwise FFN with pre-norm
        h_mid_norm = self.norm2(h_mid)
        ffn_out = self.ffn_dense2(self.ffn_dense1(h_mid_norm))
        h_out = h_mid + ffn_out

        if return_diagnostics:
            g_flat = tf.reshape(g_ij, [-1])
            diag = {
                "gate_mean": tf.reduce_mean(g_flat),
                "gate_std": tf.math.reduce_std(g_flat),
                "gate_near_min": tf.reduce_mean(tf.cast(g_flat < 0.1, tf.float32)),
                "gate_near_max": tf.reduce_mean(tf.cast(g_flat > 1.9, tf.float32)),
            }
            return h_out, diag

        return h_out
