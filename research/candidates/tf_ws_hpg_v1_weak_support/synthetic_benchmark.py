"""Synthetic-only runtime smoke benchmark for WS-HPG v1.0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import tensorflow as tf

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.candidates.tf_ws_hpg_v1_weak_support.model import (  # noqa: E402
    build_ws_hpg_v1_weak_support,
)


def _sync(value):
    return float(tf.reduce_sum(value).numpy())


def run_synthetic_benchmark(batch_sizes=(8, 16, 32, 64), repeats=1):
    tf.keras.utils.set_random_seed(1403)
    model = build_ws_hpg_v1_weak_support()

    @tf.function(reduce_retracing=True)
    def forward(inputs):
        return model(inputs, training=False)

    @tf.function(reduce_retracing=True)
    def forward_backward(inputs):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(logits))
        gradients = tape.gradient(loss, model.trainable_variables)
        return loss, gradients

    devices = tf.config.list_physical_devices("GPU")
    results = []
    for batch_size in batch_sizes:
        images = tf.random.stateless_uniform([batch_size, 48, 48, 1], [batch_size, 1])
        support = tf.ones_like(images)
        inputs = {"images": images, "support": support}
        _sync(forward(inputs))
        loss, gradients = forward_backward(inputs)
        _sync(loss)
        start = time.perf_counter()
        for _ in range(repeats):
            logits = forward(inputs)
            _sync(logits)
        forward_seconds = (time.perf_counter() - start) / repeats
        start = time.perf_counter()
        for _ in range(repeats):
            loss, gradients = forward_backward(inputs)
            _sync(loss)
            _sync(tf.add_n([tf.reduce_sum(g) for g in gradients if g is not None]))
        backward_seconds = (time.perf_counter() - start) / repeats
        _, debug = model(inputs, training=False, return_debug=True)
        memory = None
        if devices:
            try:
                memory = tf.config.experimental.get_memory_info("GPU:0")["peak"]
            except (ValueError, RuntimeError):
                memory = None
        results.append(
            {
                "batch_size": batch_size,
                "forward_seconds": forward_seconds,
                "forward_backward_seconds": backward_seconds,
                "forward_examples_per_second": batch_size / forward_seconds,
                "forward_backward_examples_per_second": batch_size / backward_seconds,
                "fine_edges": int(debug["fine_edge_count"].numpy()),
                "mid_edges_per_block": [int(v) for v in debug["mid_edge_counts"].numpy()],
                "coarse_edges_per_block": [int(v) for v in debug["coarse_edge_counts"].numpy()],
                "peak_accelerator_bytes": memory,
            }
        )
    return {"tensorflow": tf.__version__, "accelerators": [d.name for d in devices], "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(run_synthetic_benchmark(args.batch_sizes, args.repeats), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
