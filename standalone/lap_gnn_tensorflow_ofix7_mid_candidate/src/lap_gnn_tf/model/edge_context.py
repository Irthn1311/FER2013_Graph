"""Three-layer edge-context encoder."""

from __future__ import annotations

import tensorflow as tf

from lap_gnn_tf.model.gated_edge_layer import GatedEdgeLayer
from lap_gnn_tf.model.initializers import MappedLayer, StateBinding
from lap_gnn_tf.model.part_context import PartGlobalContext


class EdgeContextEncoder(MappedLayer):
    def __init__(self, hidden_dim: int = 96, edge_dim: int = 8, dropout: float = 0.25):
        super().__init__(name="edge_context_gnn")
        self.layers_ = [
            GatedEdgeLayer(index, hidden_dim, edge_dim, 32, dropout)
            for index in range(3)
        ]
        self.context = PartGlobalContext(hidden_dim, dropout)

    def call(
        self, h, edge_index, edge_features, node_graph_index, num_graphs,
        part_soft, training: bool = False, collect: bool = False,
    ):
        outputs = {}
        for index, layer in enumerate(self.layers_):
            if collect:
                h, details = layer(h, edge_index, edge_features, training=training, return_intermediates=True)
                outputs[f"gnn_layer_{index + 1}"] = h
                for name, value in details.items():
                    outputs[f"gnn_layer_{index + 1}_{name}"] = value
            else:
                h = layer(h, edge_index, edge_features, training=training)
        h = self.context(
            h, part_soft=part_soft, node_graph_index=node_graph_index,
            num_graphs=num_graphs, training=training,
        )
        outputs["pre_readout_node_representation"] = h
        return (h, outputs) if collect else h

    def encode(self, h, edge_index, edge_features, node_graph_index, num_graphs, part_soft, training=False, collect=False):
        return self.call(
            h, edge_index, edge_features, node_graph_index, num_graphs,
            part_soft, training=training, collect=collect,
        )

    def state_bindings(self) -> list[StateBinding]:
        bindings: list[StateBinding] = []
        for layer in self.layers_:
            bindings.extend(layer.state_bindings())
        bindings.extend(self.context.state_bindings())
        return bindings
