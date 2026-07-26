"""Locked 480 -> 192 -> 7 classifier."""

from __future__ import annotations

import tensorflow as tf

from lap_gnn_tf.model.initializers import MappedLayer, StateBinding, TorchLayerNorm, TorchLinear


class Classifier(MappedLayer):
    def __init__(self, dropout: float = 0.2):
        super().__init__(name="classifier")
        self.dropout_rate = float(dropout)
        self.linear1 = TorchLinear(480, 192, "classifier.net.0", name="linear1")
        self.norm = TorchLayerNorm(192, "classifier.net.1", name="norm")
        self.linear2 = TorchLinear(192, 7, "classifier.net.4", name="linear2")

    def call(self, x, training: bool = False):
        hidden = tf.nn.gelu(self.norm(self.linear1(x)), approximate=False)
        if training and self.dropout_rate:
            hidden = tf.nn.dropout(hidden, rate=self.dropout_rate)
        return self.linear2(hidden)

    def state_bindings(self) -> list[StateBinding]:
        bindings: list[StateBinding] = []
        for layer in [self.linear1, self.norm, self.linear2]:
            bindings.extend(layer.state_bindings())
        return bindings

