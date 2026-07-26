"""PyTorch-compatible Keras primitives with explicit state mappings."""

from __future__ import annotations

from dataclasses import dataclass
import math

import tensorflow as tf


@dataclass(frozen=True)
class StateBinding:
    source_key: str
    variable: tf.Variable
    transform: str


class MappedLayer(tf.keras.layers.Layer):
    def state_bindings(self) -> list[StateBinding]:
        bindings: list[StateBinding] = []
        for layer in self._flatten_layers(include_self=False, recursive=False):
            if isinstance(layer, MappedLayer):
                bindings.extend(layer.state_bindings())
        return bindings


class TorchLinear(MappedLayer):
    def __init__(self, in_features: int, out_features: int, source_prefix: str, **kwargs):
        super().__init__(**kwargs)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.source_prefix = source_prefix
        bound = 1.0 / math.sqrt(self.in_features)
        self.kernel = self.add_weight(
            name="kernel", shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.RandomUniform(-bound, bound),
            trainable=True, dtype=tf.float32,
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.out_features,),
            initializer=tf.keras.initializers.RandomUniform(-bound, bound),
            trainable=True, dtype=tf.float32,
        )

    def call(self, x):
        return tf.nn.bias_add(tf.linalg.matmul(x, self.kernel), self.bias)

    def state_bindings(self) -> list[StateBinding]:
        return [
            StateBinding(f"{self.source_prefix}.weight", self.kernel, "transpose"),
            StateBinding(f"{self.source_prefix}.bias", self.bias, "identity"),
        ]


class TorchLayerNorm(MappedLayer):
    def __init__(self, dim: int, source_prefix: str, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.source_prefix = source_prefix
        self.epsilon = float(epsilon)
        self.gamma = self.add_weight(
            name="gamma", shape=(self.dim,), initializer="ones",
            trainable=True, dtype=tf.float32,
        )
        self.beta = self.add_weight(
            name="beta", shape=(self.dim,), initializer="zeros",
            trainable=True, dtype=tf.float32,
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
        normalized = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
        return normalized * self.gamma + self.beta

    def state_bindings(self) -> list[StateBinding]:
        return [
            StateBinding(f"{self.source_prefix}.weight", self.gamma, "identity"),
            StateBinding(f"{self.source_prefix}.bias", self.beta, "identity"),
        ]


class TorchMultiheadAttention(MappedLayer):
    def __init__(self, dim: int, num_heads: int, source_prefix: str, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.source_prefix = source_prefix
        self.dropout_rate = float(dropout)
        self.in_proj_kernel = self.add_weight(
            name="in_proj_kernel", shape=(self.dim, self.dim * 3),
            initializer=tf.keras.initializers.GlorotUniform(), trainable=True, dtype=tf.float32,
        )
        self.in_proj_bias = self.add_weight(
            name="in_proj_bias", shape=(self.dim * 3,),
            initializer="zeros", trainable=True, dtype=tf.float32,
        )
        self.out_kernel = self.add_weight(
            name="out_kernel", shape=(self.dim, self.dim),
            initializer=tf.keras.initializers.RandomUniform(
                -1.0 / math.sqrt(self.dim), 1.0 / math.sqrt(self.dim),
            ),
            trainable=True, dtype=tf.float32,
        )
        self.out_bias = self.add_weight(
            name="out_bias", shape=(self.dim,),
            initializer="zeros", trainable=True, dtype=tf.float32,
        )

    def call(self, x, training: bool = False):
        qkv = tf.linalg.matmul(x, self.in_proj_kernel) + self.in_proj_bias
        q, k, v = tf.split(qkv, 3, axis=-1)
        batch = tf.shape(x)[0]
        length = tf.shape(x)[1]

        def split_heads(tensor):
            tensor = tf.reshape(tensor, (batch, length, self.num_heads, self.head_dim))
            return tf.transpose(tensor, (0, 2, 1, 3))

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        q = q * tf.cast(self.head_dim ** -0.5, q.dtype)
        scores = tf.linalg.matmul(q, k, transpose_b=True)
        attention = tf.nn.softmax(scores, axis=-1)
        if training and self.dropout_rate:
            attention = tf.nn.dropout(attention, rate=self.dropout_rate)
        context = tf.linalg.matmul(attention, v)
        context = tf.transpose(context, (0, 2, 1, 3))
        context = tf.reshape(context, (batch, length, self.dim))
        return tf.linalg.matmul(context, self.out_kernel) + self.out_bias

    def state_bindings(self) -> list[StateBinding]:
        return [
            StateBinding(f"{self.source_prefix}.in_proj_weight", self.in_proj_kernel, "transpose"),
            StateBinding(f"{self.source_prefix}.in_proj_bias", self.in_proj_bias, "identity"),
            StateBinding(f"{self.source_prefix}.out_proj.weight", self.out_kernel, "transpose"),
            StateBinding(f"{self.source_prefix}.out_proj.bias", self.out_bias, "identity"),
        ]


class TorchTransformerEncoderLayer(MappedLayer):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        source_prefix: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout_rate = float(dropout)
        self.self_attn = TorchMultiheadAttention(
            dim, num_heads, f"{source_prefix}.self_attn", dropout=dropout, name="self_attn",
        )
        self.linear1 = TorchLinear(dim, ff_dim, f"{source_prefix}.linear1", name="linear1")
        self.linear2 = TorchLinear(ff_dim, dim, f"{source_prefix}.linear2", name="linear2")
        self.norm1 = TorchLayerNorm(dim, f"{source_prefix}.norm1", name="norm1")
        self.norm2 = TorchLayerNorm(dim, f"{source_prefix}.norm2", name="norm2")

    def call(self, x, training: bool = False):
        attn = self.self_attn(x, training=training)
        if training and self.dropout_rate:
            attn = tf.nn.dropout(attn, rate=self.dropout_rate)
        x = self.norm1(x + attn)
        hidden = tf.nn.gelu(self.linear1(x), approximate=False)
        if training and self.dropout_rate:
            hidden = tf.nn.dropout(hidden, rate=self.dropout_rate)
        hidden = self.linear2(hidden)
        if training and self.dropout_rate:
            hidden = tf.nn.dropout(hidden, rate=self.dropout_rate)
        return self.norm2(x + hidden)

    def state_bindings(self) -> list[StateBinding]:
        bindings: list[StateBinding] = []
        for layer in [self.self_attn, self.linear1, self.linear2, self.norm1, self.norm2]:
            bindings.extend(layer.state_bindings())
        return bindings
