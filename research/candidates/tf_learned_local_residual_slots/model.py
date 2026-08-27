"""Step-10 candidate with four graph-local learned residual slots.

Only the source of the first four readout residual embeddings changes. The
frozen graph encoder, context encoder, support/motif readout, official global
residual, and classifier are reused without modification.
"""

from __future__ import annotations

import tensorflow as tf

from lap_gnn_tf.model.lap_gnn import LapGNN
from lap_gnn_tf.model.motif_layers import PART_ORDER, part_pool


HIDDEN_DIM = 96
NUM_LOCAL_SLOTS = 4
NEW_TRAINABLE_SCALARS = NUM_LOCAL_SLOTS * HIDDEN_DIM


@tf.keras.utils.register_keras_serializable(package="fer2013_graph_research")
class LearnedLocalResidualSlotPool(tf.keras.layers.Layer):
    """Pool exactly four query-ordered slots independently within each graph."""

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_slots: int = NUM_LOCAL_SLOTS,
        **kwargs,
    ):
        if int(hidden_dim) != HIDDEN_DIM:
            raise ValueError(f"Step 10 requires hidden_dim={HIDDEN_DIM}")
        if int(num_slots) != NUM_LOCAL_SLOTS:
            raise ValueError(f"Step 10 requires num_slots={NUM_LOCAL_SLOTS}")
        kwargs.pop("name", None)
        kwargs.pop("dtype", None)
        super().__init__(
            name="learned_local_residual_slot_pool",
            dtype=tf.float32,
            **kwargs,
        )
        self.hidden_dim = HIDDEN_DIM
        self.num_slots = NUM_LOCAL_SLOTS
        self.Q = self.add_weight(
            name="Q",
            shape=(NUM_LOCAL_SLOTS, HIDDEN_DIM),
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, h, node_graph_index, num_graphs):
        h = tf.cast(tf.convert_to_tensor(h), tf.float32)
        node_graph_index = tf.cast(
            tf.reshape(tf.convert_to_tensor(node_graph_index), (-1,)), tf.int32
        )
        num_graphs = tf.cast(num_graphs, tf.int32)

        tf.debugging.assert_rank(h, 2)
        tf.debugging.assert_equal(tf.shape(h)[1], HIDDEN_DIM)
        tf.debugging.assert_equal(tf.shape(h)[0], tf.shape(node_graph_index)[0])
        tf.debugging.assert_positive(num_graphs)
        tf.debugging.assert_greater_equal(node_graph_index, 0)
        tf.debugging.assert_less(node_graph_index, num_graphs)
        graph_node_counts = tf.math.unsorted_segment_sum(
            tf.ones_like(node_graph_index), node_graph_index, num_graphs
        )
        tf.debugging.assert_positive(
            graph_node_counts, message="Every graph must contain at least one node"
        )

        scale = tf.math.rsqrt(tf.cast(HIDDEN_DIM, h.dtype))
        scores = tf.einsum("kh,nh->nk", self.Q, h) * scale
        graph_max = tf.math.unsorted_segment_max(
            scores, node_graph_index, num_graphs
        )
        shifted = scores - tf.gather(graph_max, node_graph_index)
        unnormalized = tf.exp(shifted)
        graph_denominator = tf.math.unsorted_segment_sum(
            unnormalized, node_graph_index, num_graphs
        )
        attention_weights = unnormalized / tf.gather(
            graph_denominator, node_graph_index
        )

        weighted_nodes = attention_weights[:, :, None] * h[:, None, :]
        slot_embeddings = tf.math.unsorted_segment_sum(
            weighted_nodes, node_graph_index, num_graphs
        )
        safe_attention = tf.maximum(
            attention_weights, tf.cast(tf.keras.backend.epsilon(), h.dtype)
        )
        entropy_terms = -attention_weights * tf.math.log(safe_attention)
        attention_entropy = tf.math.unsorted_segment_sum(
            entropy_terms, node_graph_index, num_graphs
        )
        attention_peak = tf.math.unsorted_segment_max(
            attention_weights, node_graph_index, num_graphs
        )
        return {
            "slot_embeddings": slot_embeddings,
            "attention_weights": attention_weights,
            "attention_entropy": attention_entropy,
            "attention_peak": attention_peak,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_dim": self.hidden_dim, "num_slots": self.num_slots}
        )
        return config


@tf.keras.utils.register_keras_serializable(package="fer2013_graph_research")
class LearnedLocalResidualSlotLapGNN(LapGNN):
    """Frozen LAP-GNN with only its four local residual sources replaced."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.learned_local_residual_slots = LearnedLocalResidualSlotPool()

    def call(
        self,
        batch: dict[str, tf.Tensor],
        training: bool = False,
        collect_intermediates: bool = False,
    ):
        node_features = tf.cast(batch["node_features"], tf.float32)
        edge_features = tf.cast(batch["edge_features"], tf.float32)
        edge_index = tf.cast(batch["edge_index"], tf.int64)
        node_graph_index = tf.cast(batch["node_graph_index"], tf.int32)
        part_soft = tf.cast(batch["part_soft"], tf.float32)
        valid_part_mask = tf.cast(batch["valid_part_mask"], tf.float32)
        num_graphs = tf.shape(batch["labels"])[0]

        h = self.encoder(node_features, training=training)
        intermediates = {"input_projection": h}
        if collect_intermediates:
            h, gnn_outputs = self.gnn.encode(
                h,
                edge_index,
                edge_features,
                node_graph_index,
                num_graphs,
                part_soft,
                training=training,
                collect=True,
            )
            intermediates.update(gnn_outputs)
        else:
            h = self.gnn.encode(
                h,
                edge_index,
                edge_features,
                node_graph_index,
                num_graphs,
                part_soft,
                training=training,
                collect=False,
            )

        official_pooled, official_valid_groups = part_pool(
            h,
            part_soft,
            node_graph_index,
            valid_part_mask,
            num_graphs,
        )
        slot_diagnostics = self.learned_local_residual_slots(
            h, node_graph_index, num_graphs
        )
        raw_slot_embeddings = slot_diagnostics["slot_embeddings"]
        residual_slot_embeddings = tf.cast(
            raw_slot_embeddings, official_pooled["global"].dtype
        )
        learned_slots = tf.unstack(
            residual_slot_embeddings,
            num=NUM_LOCAL_SLOTS,
            axis=1,
        )

        # The frozen readout requires PART_ORDER dictionary keys. These four
        # entries are positional interface aliases only; they carry no anatomy.
        residual_part_embeddings = {
            key: learned_slots[index]
            for index, key in enumerate(PART_ORDER[:NUM_LOCAL_SLOTS])
        }
        residual_part_embeddings["global"] = official_pooled["global"]
        residual_stack = tf.stack(
            [residual_part_embeddings[key] for key in PART_ORDER], axis=1
        )

        readout = self.readout(
            h,
            node_features,
            part_soft,
            node_graph_index,
            num_graphs,
            residual_part_embeddings,
            official_valid_groups,
            training=training,
        )
        z_image = readout["z_image"]
        logits = self.classifier(z_image, training=training)
        probabilities = tf.nn.softmax(logits, axis=-1)
        if collect_intermediates:
            intermediates.update(
                {
                    "micro_major_motif_tokens": readout["major_tokens"],
                    "micro_major_motif_transformed_tokens": readout[
                        "major_transformed_tokens"
                    ],
                    "micro_motif_tokens": readout["micro_tokens"],
                    "micro_motif_transformed_tokens": readout[
                        "micro_transformed_tokens"
                    ],
                    "micro_support_gate": readout["micro_gate"],
                    "pooled_graph_embedding": z_image,
                    "classifier_input": z_image,
                }
            )
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": tf.argmax(logits, axis=1, output_type=tf.int64),
            "z_image": z_image,
            "node_embeddings": h,
            "learned_local_residual_slots": raw_slot_embeddings,
            "learned_local_attention_weights": slot_diagnostics[
                "attention_weights"
            ],
            "learned_local_attention_entropy": slot_diagnostics[
                "attention_entropy"
            ],
            "learned_local_attention_peak": slot_diagnostics["attention_peak"],
            "official_global_residual": official_pooled["global"],
            "residual_stack": residual_stack,
            "residual_flat": tf.reshape(
                residual_stack, (num_graphs, NUM_LOCAL_SLOTS * HIDDEN_DIM + HIDDEN_DIM)
            ),
            "intermediates": intermediates,
        }

    def get_config(self):
        return super().get_config()


def build_candidate_model(
    golden_batch: dict[str, tf.Tensor] | None = None,
) -> LearnedLocalResidualSlotLapGNN:
    model = LearnedLocalResidualSlotLapGNN()
    if golden_batch is not None:
        model(golden_batch, training=False)
    return model
