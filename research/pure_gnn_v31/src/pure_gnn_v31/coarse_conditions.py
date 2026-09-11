"""Coarse relational blocks and condition implementations (G0, G0.5, G1, G2, G3) for Pure-GNN v3.1."""

from typing import Dict, Optional, Tuple
import tensorflow as tf


class CoarseBlock(tf.keras.layers.Layer):
    """Unified Coarse Block supporting G0, G0.5, and G1/G2/G3 conditions.
    
    Functional-Path Matching Contract:
      - Between G0.5 and G1:
        * Value projection V(h_j - h_i) is structurally identical.
        * Update, pre-norm LayerNorm, and FFN are structurally identical.
        * Complete graph with 1260 edges is identical.
        * In G0.5, normalized edge weight is exactly 1/35.
        * In G1, normalized edge weight is learned a_bar_ij, zero-initialized to ~1/35.
      - In G0 (local only):
        * No cross-node message passing; pure residual node FFN blocks.
    """

    def __init__(
        self,
        channels: int = 128,
        condition: str = "G1",
        gate_hidden_dim: int = 4,
        ffn_expansion: int = 4,
        epsilon: float = 1e-6,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        self.condition = condition.upper()
        self.gate_hidden_dim = gate_hidden_dim
        self.ffn_expansion = ffn_expansion
        self.epsilon = epsilon

        if self.condition not in ("G0", "G0.5", "G1", "G2", "G3"):
            raise ValueError(f"Unknown condition: {self.condition}. Expected G0, G0.5, G1, G2, G3.")

        # Norms
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="norm1")
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="norm2")

        # FFN
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

        if self.condition != "G0":
            # Receiver-relative value function V(h_j - h_i)
            self.v_proj = tf.keras.layers.Dense(
                channels,
                use_bias=False,
                kernel_initializer="glorot_uniform",
                name="v_proj",
            )

        if self.condition in ("G1", "G2", "G3"):
            # Learned pair-specific relational gate MLP
            # Input: LN(h_i) [C] + LN(h_j - h_i) [C] + p_ij [3] -> 2*C + 3
            self.coarse_gate_norm_diff = tf.keras.layers.LayerNormalization(
                axis=-1, epsilon=1e-5, name="coarse_gate_norm_diff"
            )
            self.coarse_gate_dense1 = tf.keras.layers.Dense(
                gate_hidden_dim,
                activation="gelu",
                kernel_initializer="glorot_uniform",
                name="coarse_gate_dense1",
            )
            # Final projection ZERO initialized: s_ij = 0 -> sigmoid(0) = 0.5
            self.coarse_gate_dense2 = tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer="zeros",
                bias_initializer="zeros",
                name="coarse_gate_dense2",
            )

    def call(
        self,
        h: tf.Tensor,
        coarse_graph: Optional[Dict[str, tf.Tensor]] = None,
        training: Optional[bool] = None,
        return_diagnostics: bool = False,
    ):
        """Forward pass.
        
        Args:
            h: (B, 36, C) node feature tensor
            coarse_graph: dict with src_indices (1260,), dst_indices (1260,), p_ij (1260, 3)
            training: bool
            return_diagnostics: bool
        """
        # G0: local-only node-wise residual FFN
        if self.condition == "G0":
            h_norm = self.norm1(h)
            ffn_out = self.ffn_dense2(self.ffn_dense1(h_norm))
            h_out = h + ffn_out
            if return_diagnostics:
                return h_out, {}
            return h_out

        # Non-local relational communication (G0.5, G1, G2, G3)
        num_nodes = coarse_graph["num_nodes"]  # 36
        src = coarse_graph["src_indices"]      # 1260
        dst = coarse_graph["dst_indices"]      # 1260
        p_ij = coarse_graph["p_ij"]            # (1260, 3)

        # The pair value is defined on raw receiver-relative state.  Normalized
        # copies exist only for the learned gate.
        raw_h_i = tf.gather(h, dst, axis=1)  # (B, E, C)
        raw_h_j = tf.gather(h, src, axis=1)  # (B, E, C)
        raw_diff_ij = raw_h_j - raw_h_i
        q_ij = self.v_proj(raw_diff_ij)  # (B, E, C)
        # This shared normalization is built in both G0.5 and G1.  It affects
        # only the G1 gate input and never the pair value q_ij.
        h_i_norm = self.norm1(raw_h_i)

        diagnostics = {}

        if self.condition == "G0.5":
            # EXACT UNIFORM WEIGHT: 1 / 35
            edges_per_node = tf.cast(coarse_graph["edges_per_node"], tf.float32)
            uniform_weight = 1.0 / edges_per_node  # 1 / 35
            m_ij = q_ij * uniform_weight

            # Aggregate per receiver node i: sum_{j != i} (1/35) * q_ij
            m_ij_t = tf.transpose(m_ij, [1, 0, 2])
            m_i_t = tf.math.unsorted_segment_sum(m_ij_t, dst, num_segments=num_nodes)
            m_i = tf.transpose(m_i_t, [1, 0, 2])  # (B, 36, C)

            if return_diagnostics:
                diagnostics["coarse_weight_entropy"] = tf.constant(3.555348, dtype=tf.float32)
                diagnostics["effective_neighbor_count"] = tf.constant(35.0, dtype=tf.float32)
                diagnostics["coarse_weight_min"] = uniform_weight
                diagnostics["coarse_weight_max"] = uniform_weight
                diagnostics["receiver_weight_sums"] = tf.ones(
                    [tf.shape(h)[0], num_nodes, 1], dtype=q_ij.dtype
                )
                diagnostics["pair_weights"] = tf.ones_like(q_ij[..., :1]) * uniform_weight
                diagnostics["pair_values"] = q_ij

        else:
            # G1, G2, G3: Learned pair-specific gate
            diff_ij_norm = self.coarse_gate_norm_diff(raw_diff_ij)
            b_size = tf.shape(h)[0]
            p_ij_b = tf.broadcast_to(p_ij[None, :, :], [b_size, tf.shape(p_ij)[0], 3])

            gate_input = tf.concat([h_i_norm, diff_ij_norm, p_ij_b], axis=-1)  # (B, E, 2*C + 3)
            s_ij = self.coarse_gate_dense2(self.coarse_gate_dense1(gate_input))  # (B, 1260, 1)
            a_ij = tf.sigmoid(s_ij)  # (B, 1260, 1)

            # Receiver normalization: sum_{k != i} a_ik
            # Transpose to (1260, B, 1) for unsorted_segment_sum
            a_ij_t = tf.transpose(a_ij, [1, 0, 2])
            a_sum_t = tf.math.unsorted_segment_sum(a_ij_t, dst, num_segments=num_nodes)  # (36, B, 1)
            a_sum = tf.transpose(a_sum_t, [1, 0, 2])  # (B, 36, 1)

            # Gather sum for each edge's receiver
            receiver_sum = tf.gather(a_sum, dst, axis=1)  # (B, 1260, 1)
            a_bar_ij = a_ij / (receiver_sum + self.epsilon)  # (B, 1260, 1)

            # Message: a_bar_ij * q_ij
            m_ij = a_bar_ij * q_ij  # (B, 1260, C)

            # Aggregate per receiver node i: sum_{j != i} a_bar_ij * q_ij
            m_ij_t = tf.transpose(m_ij, [1, 0, 2])
            m_i_t = tf.math.unsorted_segment_sum(m_ij_t, dst, num_segments=num_nodes)
            m_i = tf.transpose(m_i_t, [1, 0, 2])  # (B, 36, C)

            if return_diagnostics:
                # Diagnostics: weight entropy per receiver
                # a_bar_ij is (B, 1260, 1). Compute entropy: -sum a_bar * log(a_bar + eps)
                log_a_bar = tf.math.log(a_bar_ij + 1e-12)
                ent_edge = -a_bar_ij * log_a_bar
                ent_edge_t = tf.transpose(ent_edge, [1, 0, 2])
                node_ent_t = tf.math.unsorted_segment_sum(ent_edge_t, dst, num_segments=num_nodes)
                node_ent = tf.transpose(node_ent_t, [1, 0, 2])  # (B, 36, 1)
                mean_ent = tf.reduce_mean(node_ent)
                diagnostics["coarse_weight_entropy"] = mean_ent
                diagnostics["effective_neighbor_count"] = tf.exp(mean_ent)
                diagnostics["coarse_weight_min"] = tf.reduce_min(a_bar_ij)
                diagnostics["coarse_weight_max"] = tf.reduce_max(a_bar_ij)
                diagnostics["receiver_weight_sums"] = tf.transpose(
                    tf.math.unsorted_segment_sum(
                        tf.transpose(a_bar_ij, [1, 0, 2]), dst, num_segments=num_nodes
                    ),
                    [1, 0, 2],
                )
                diagnostics["pair_weights"] = a_bar_ij
                diagnostics["pair_values"] = q_ij
                diagnostics["gate_input_content"] = tf.concat(
                    [h_i_norm, diff_ij_norm], axis=-1
                )

        # Pre-norm residual update
        h_mid = h + m_i

        # Pointwise FFN
        h_mid_norm = self.norm2(h_mid)
        ffn_out = self.ffn_dense2(self.ffn_dense1(h_mid_norm))
        h_out = h_mid + ffn_out

        if return_diagnostics:
            return h_out, diagnostics
        return h_out
