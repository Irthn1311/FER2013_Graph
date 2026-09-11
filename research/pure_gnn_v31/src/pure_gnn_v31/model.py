"""Full Pure-GNN v3.1 Model and Classifier Architecture."""

from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf

from pure_gnn_v31.graph_index import (
    build_grid_8neighbor_graph,
    build_complete_coarse_graph,
)
from pure_gnn_v31.local_relation import LocalAdaptiveRelationBlock
from pure_gnn_v31.coarsening import FixedAntiAliasedCoarsening
from pure_gnn_v31.coarse_conditions import CoarseBlock
from pure_gnn_v31.diagnostics import tensor_feature_diagnostics


class PureGNNv31(tf.keras.Model):
    """Pure-GNN v3.1 Architecture.
    
    Operates directly from raw 48x48 pixel intensities using only graph-local
    and relational graph operations.
    Contains NO Conv2D, NO CNN, NO MediaPipe, NO facial landmarks, NO Transformers.
    """

    def __init__(
        self,
        condition: str = "G1",
        channels: Tuple[int, int, int, int] = (32, 64, 96, 128),
        blocks_per_stage: Tuple[int, int, int, int] = (2, 2, 2, 2),
        local_gate_hidden_dim: int = 16,
        coarse_gate_hidden_dim: int = 4,
        ffn_expansion: int = 4,
        num_classes: int = 7,
        dropout_rate: float = 0.1,
        name: Optional[str] = "PureGNNv31",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.condition = condition.upper()
        self.channels = channels
        self.blocks_per_stage = blocks_per_stage
        self.local_gate_hidden_dim = local_gate_hidden_dim
        self.coarse_gate_hidden_dim = coarse_gate_hidden_dim
        self.ffn_expansion = ffn_expansion
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Fixed graph topologies
        self.graph_s1 = build_grid_8neighbor_graph(48, 48)
        self.graph_s2 = build_grid_8neighbor_graph(24, 24)
        self.graph_s3 = build_grid_8neighbor_graph(12, 12)
        self.graph_coarse = build_complete_coarse_graph(6)

        c1, c2, c3, c4 = channels
        b1, b2, b3, b4 = blocks_per_stage

        # Input pointwise projection: 1 -> 32
        self.input_proj = tf.keras.layers.Dense(
            c1,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            name="input_proj",
        )

        # Condition G2 flag: mask content inputs to local gate
        mask_local_content = (self.condition == "G2")
        # Condition G3 flag: simple local mean downsampling
        use_simple_mean_coarsening = (self.condition == "G3")

        # Stage 1: 48x48, c1
        self.stage1_blocks = [
            LocalAdaptiveRelationBlock(
                channels=c1,
                gate_hidden_dim=local_gate_hidden_dim,
                ffn_expansion=ffn_expansion,
                mask_content=mask_local_content,
                name=f"stage1_block{i}",
            )
            for i in range(b1)
        ]

        # Coarsening 1: 48x48 -> 24x24, c1 -> c2
        self.coarsen1 = FixedAntiAliasedCoarsening(
            in_height=48,
            in_width=48,
            in_channels=c1,
            out_channels=c2,
            use_simple_mean=use_simple_mean_coarsening,
            name="coarsen1",
        )

        # Stage 2: 24x24, c2
        self.stage2_blocks = [
            LocalAdaptiveRelationBlock(
                channels=c2,
                gate_hidden_dim=local_gate_hidden_dim,
                ffn_expansion=ffn_expansion,
                mask_content=mask_local_content,
                name=f"stage2_block{i}",
            )
            for i in range(b2)
        ]

        # Coarsening 2: 24x24 -> 12x12, c2 -> c3
        self.coarsen2 = FixedAntiAliasedCoarsening(
            in_height=24,
            in_width=24,
            in_channels=c2,
            out_channels=c3,
            use_simple_mean=use_simple_mean_coarsening,
            name="coarsen2",
        )

        # Stage 3: 12x12, c3
        self.stage3_blocks = [
            LocalAdaptiveRelationBlock(
                channels=c3,
                gate_hidden_dim=local_gate_hidden_dim,
                ffn_expansion=ffn_expansion,
                mask_content=mask_local_content,
                name=f"stage3_block{i}",
            )
            for i in range(b3)
        ]

        # Coarsening 3: 12x12 -> 6x6, c3 -> c4
        self.coarsen3 = FixedAntiAliasedCoarsening(
            in_height=12,
            in_width=12,
            in_channels=c3,
            out_channels=c4,
            use_simple_mean=use_simple_mean_coarsening,
            name="coarsen3",
        )

        # Stage 4: 6x6, c4
        self.stage4_blocks = [
            CoarseBlock(
                channels=c4,
                condition=self.condition,
                gate_hidden_dim=coarse_gate_hidden_dim,
                ffn_expansion=ffn_expansion,
                name=f"stage4_block{i}",
            )
            for i in range(b4)
        ]

        # Final Readout & Classification Head
        # Concat mean and max over 36 nodes -> 2 * c4 = 256
        self.readout_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, name="readout_norm"
        )
        self.head_dense1 = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_initializer="glorot_uniform",
            name="head_dense1",
        )
        self.head_dropout = tf.keras.layers.Dropout(dropout_rate, name="head_dropout")
        self.classifier = tf.keras.layers.Dense(
            num_classes,
            activation=None,
            kernel_initializer="glorot_uniform",
            name="classifier",
        )

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        return_diagnostics: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, Dict[str, tf.Tensor]]]:
        """Forward pass.
        
        Args:
            inputs: Tensor of shape (B, 48, 48), (B, 48, 48, 1), or (B, 2304, 1)
            training: bool
            return_diagnostics: bool
        Returns:
            logits: (B, 7) or (logits, diagnostics_dict)
        """
        # Ensure inputs are shaped (B, 2304, 1)
        if len(inputs.shape) == 4:  # (B, 48, 48, 1)
            h = tf.reshape(inputs, [-1, 2304, 1])
        elif len(inputs.shape) == 3:
            if inputs.shape[1] == 48 and inputs.shape[2] == 48:  # (B, 48, 48)
                h = tf.reshape(inputs, [-1, 2304, 1])
            else:  # (B, 2304, 1)
                h = inputs
        else:
            raise ValueError(f"Unexpected input shape: {inputs.shape}")

        h = tf.cast(h, tf.float32)

        # Initial node projection
        h = self.input_proj(h)  # (B, 2304, 32)

        diagnostics = {}

        def record_block(prefix, values, features):
            for key, value in values.items():
                if value.shape.rank == 0:
                    diagnostics[f"{prefix}_{key}"] = value
            for key, value in tensor_feature_diagnostics(features).items():
                diagnostics[f"{prefix}_{key}"] = value

        # Stage 1: 48x48
        for i, block in enumerate(self.stage1_blocks):
            if return_diagnostics:
                h, diag = block(h, graph=self.graph_s1, training=training, return_diagnostics=True)
                record_block(f"stage1_block{i}", diag, h)
                if i == 0:
                    for key in ("gate_mean", "gate_std", "gate_near_min", "gate_near_max"):
                        diagnostics[f"stage1_{key}"] = diag[key]
            else:
                h = block(h, graph=self.graph_s1, training=training)

        # Coarsen 1: 48x48 -> 24x24
        h = self.coarsen1(h)  # (B, 576, 64)

        # Stage 2: 24x24
        for i, block in enumerate(self.stage2_blocks):
            if return_diagnostics:
                h, diag = block(h, graph=self.graph_s2, training=training, return_diagnostics=True)
                record_block(f"stage2_block{i}", diag, h)
            else:
                h = block(h, graph=self.graph_s2, training=training)

        # Coarsen 2: 24x24 -> 12x12
        h = self.coarsen2(h)  # (B, 144, 96)

        # Stage 3: 12x12
        for i, block in enumerate(self.stage3_blocks):
            if return_diagnostics:
                h, diag = block(h, graph=self.graph_s3, training=training, return_diagnostics=True)
                record_block(f"stage3_block{i}", diag, h)
            else:
                h = block(h, graph=self.graph_s3, training=training)

        # Coarsen 3: 12x12 -> 6x6
        h = self.coarsen3(h)  # (B, 36, 128)

        # Stage 4: 6x6 Coarse relational
        for i, block in enumerate(self.stage4_blocks):
            if return_diagnostics:
                h, diag = block(h, coarse_graph=self.graph_coarse, training=training, return_diagnostics=True)
                record_block(f"coarse_block{i}", diag, h)
                if i == 0:
                    for key in (
                        "coarse_weight_entropy",
                        "effective_neighbor_count",
                        "coarse_weight_min",
                        "coarse_weight_max",
                    ):
                        if key in diag:
                            diagnostics[f"coarse_{key}"] = diag[key]
            else:
                h = block(h, coarse_graph=self.graph_coarse, training=training)

        # Final Readout
        h_norm = self.readout_norm(h)  # (B, 36, 128)
        mean_pooled = tf.reduce_mean(h_norm, axis=1)  # (B, 128)
        max_pooled = tf.reduce_max(h_norm, axis=1)    # (B, 128)
        z = tf.concat([mean_pooled, max_pooled], axis=-1)  # (B, 256)

        # Classifier
        hidden = self.head_dense1(z)
        hidden = self.head_dropout(hidden, training=training)
        logits = self.classifier(hidden)  # (B, 7)

        if return_diagnostics:
            return logits, diagnostics

        return logits
