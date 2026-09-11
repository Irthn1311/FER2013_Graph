"""Synchronous synthetic inference and optimizer-step runtime benchmarks."""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31


def _synchronize(value: tf.Tensor) -> None:
    """Materialize a scalar so elapsed time includes device completion."""
    _ = float(tf.reduce_sum(value).numpy())


def _gpu_memory() -> Optional[Dict[str, int]]:
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
    except (ValueError, RuntimeError):
        return None
    return {"current_bytes": int(info["current"]), "peak_bytes": int(info["peak"])}


def _reset_peak_memory() -> None:
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except (ValueError, RuntimeError):
        pass


def _measure(callable_step, batch_size: int, warmup: int, steps: int) -> Dict[str, float]:
    for _ in range(warmup):
        _synchronize(callable_step())
    _reset_peak_memory()
    start = time.perf_counter()
    for _ in range(steps):
        _synchronize(callable_step())
    elapsed = time.perf_counter() - start
    return {
        "steps": steps,
        "ms_per_step": elapsed * 1000.0 / steps,
        "examples_per_second": batch_size * steps / elapsed,
        "gpu_memory": _gpu_memory(),
    }


def benchmark_batch_sizes(
    batch_sizes: Sequence[int] = (16, 32, 64),
    condition: str = "G1",
    num_warmup: int = 5,
    num_steps: int = 10,
    output_path: Optional[str] = None,
    technical_seed: int = 42,
) -> Dict:
    """Benchmark float32 forward and forward/backward/update on synthetic data."""
    results = {
        "status": "FAIL",
        "condition": condition,
        "synthetic_only": True,
        "technical_seed": technical_seed,
        "dtype_policy": tf.keras.mixed_precision.global_policy().name,
        "tensorflow_version": tf.__version__,
        "devices": [device.name for device in tf.config.list_physical_devices()],
        "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "benchmarks": {},
    }
    for batch_size in batch_sizes:
        key = f"B{batch_size}"
        try:
            model = PureGNNv31(condition=condition)
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            images = tf.random.stateless_uniform(
                [batch_size, 48, 48, 1], seed=[technical_seed, batch_size], dtype=tf.float32
            )
            labels = tf.math.mod(tf.range(batch_size), 7)

            @tf.function(reduce_retracing=True)
            def inference_step():
                return model(images, training=False)

            @tf.function(reduce_retracing=True)
            def training_step():
                with tf.GradientTape() as tape:
                    logits = model(images, training=True)
                    loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=labels, logits=logits
                        )
                    )
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss

            results["benchmarks"][key] = {
                "status": "COMPLETE",
                "batch_size": batch_size,
                "inference": _measure(inference_step, batch_size, num_warmup, num_steps),
                "training_step": _measure(training_step, batch_size, num_warmup, num_steps),
            }
        except (tf.errors.ResourceExhaustedError, MemoryError) as exc:
            results["benchmarks"][key] = {
                "status": "NOT_FEASIBLE",
                "batch_size": batch_size,
                "error": str(exc),
            }
            if batch_size == 32:
                break
        finally:
            tf.keras.backend.clear_session()

    if results["benchmarks"].get("B32", {}).get("status") == "COMPLETE":
        results["status"] = "PASS"
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
