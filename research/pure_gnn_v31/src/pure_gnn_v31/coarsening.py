"""Fixed Anti-Aliased and Control Graph Coarsening layers for Pure-GNN v3.1."""

from typing import Dict, Optional, Tuple
import numpy as np
import tensorflow as tf


def compute_aa_weights_and_indices(
    in_height: int,
    in_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes sparse indices and normalized coefficients for 3x3 binomial AA filter with stride 2.
    
    Binomial 1D kernel: [1, 2, 1] for offsets [-1, 0, 1].
    Renormalizes available kernel coefficients at image boundaries:
      - Interior nodes: sum = 16
      - Edge nodes: sum = 12
      - Corner nodes: sum = 9
    
    Subsamples with stride 2: out_height = in_height // 2, out_width = in_width // 2.
    Output node k = r_out * out_width + c_out corresponds to center (2*r_out, 2*c_out) in input.
    
    Returns:
        out_indices: (M,) int32, output coarse node index
        in_indices: (M,) int32, input fine node index
        weights: (M,) float32, normalized anti-aliased kernel weight
    """
    out_height = in_height // 2
    out_width = in_width // 2

    b1d = { -1: 1.0, 0: 2.0, 1: 1.0 }

    out_idx_list = []
    in_idx_list = []
    w_list = []

    for r_out in range(out_height):
        r_in = 2 * r_out
        for c_out in range(out_width):
            c_in = 2 * c_out
            k_out = r_out * out_width + c_out

            # Step 1: find valid neighbors and compute sum for boundary renormalization
            valid_neighbors = []
            weight_sum = 0.0
            for dr in (-1, 0, 1):
                nr = r_in + dr
                if not (0 <= nr < in_height):
                    continue
                w_r = b1d[dr]
                for dc in (-1, 0, 1):
                    nc = c_in + dc
                    if not (0 <= nc < in_width):
                        continue
                    w_c = b1d[dc]
                    w = w_r * w_c
                    weight_sum += w
                    valid_neighbors.append((nr * in_width + nc, w))

            # Step 2: normalize weights and append
            for in_node_idx, raw_w in valid_neighbors:
                norm_w = raw_w / weight_sum
                out_idx_list.append(k_out)
                in_idx_list.append(in_node_idx)
                w_list.append(norm_w)

    return (
        np.array(out_idx_list, dtype=np.int32),
        np.array(in_idx_list, dtype=np.int32),
        np.array(w_list, dtype=np.float32),
    )


def compute_simple_mean_indices(
    in_height: int,
    in_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes indices and uniform 1/4 weights for 2x2 deterministic local mean (G3 control)."""
    out_height = in_height // 2
    out_width = in_width // 2

    out_idx_list = []
    in_idx_list = []
    w_list = []

    for r_out in range(out_height):
        r_in = 2 * r_out
        for c_out in range(out_width):
            c_in = 2 * c_out
            k_out = r_out * out_width + c_out
            for dr in (0, 1):
                nr = r_in + dr
                for dc in (0, 1):
                    nc = c_in + dc
                    k_in = nr * in_width + nc
                    out_idx_list.append(k_out)
                    in_idx_list.append(k_in)
                    w_list.append(0.25)

    return (
        np.array(out_idx_list, dtype=np.int32),
        np.array(in_idx_list, dtype=np.int32),
        np.array(w_list, dtype=np.float32),
    )


class FixedAntiAliasedCoarsening(tf.keras.layers.Layer):
    """Fixed non-trainable anti-aliased binomial 3x3 coarsening with stride 2 and boundary renormalization.
    
    Followed by pointwise learned channel projection.
    """

    def __init__(
        self,
        in_height: int,
        in_width: int,
        in_channels: int,
        out_channels: int,
        use_simple_mean: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.in_height = in_height
        self.in_width = in_width
        self.out_height = in_height // 2
        self.out_width = in_width // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_simple_mean = use_simple_mean

        self.num_out_nodes = self.out_height * self.out_width

        if use_simple_mean:
            out_idx, in_idx, w = compute_simple_mean_indices(in_height, in_width)
        else:
            out_idx, in_idx, w = compute_aa_weights_and_indices(in_height, in_width)

        self.out_idx = tf.constant(out_idx, dtype=tf.int32)
        self.in_idx = tf.constant(in_idx, dtype=tf.int32)
        self.weights_aa = tf.constant(w[:, None], dtype=tf.float32)  # (M, 1)

        # Pointwise learned channel projection
        self.proj = tf.keras.layers.Dense(
            out_channels,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            name="channel_proj",
        )

    def call(self, h: tf.Tensor) -> tf.Tensor:
        """Forward pass.
        
        Args:
            h: (B, N_in, C_in)
        Returns:
            h_coarse: (B, N_out, C_out)
        """
        # Gather input nodes for each kernel tap: (B, M, C_in)
        h_gathered = tf.gather(h, self.in_idx, axis=1)

        # Multiply by fixed non-trainable scalar kernel weight: (B, M, C_in)
        w_expanded = self.weights_aa[None, :, :]  # (1, M, 1)
        h_weighted = h_gathered * w_expanded

        # Sum per output coarse node
        # Transpose to (M, B, C_in) for unsorted_segment_sum
        h_weighted_t = tf.transpose(h_weighted, [1, 0, 2])
        h_coarse_t = tf.math.unsorted_segment_sum(
            h_weighted_t,
            self.out_idx,
            num_segments=self.num_out_nodes,
        )
        h_smoothed = tf.transpose(h_coarse_t, [1, 0, 2])  # (B, N_out, C_in)

        # Pointwise channel projection: (B, N_out, C_out)
        h_out = self.proj(h_smoothed)
        return h_out
