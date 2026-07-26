"""Runtime-only resource controls and telemetry."""

from __future__ import annotations

import os
import platform
import time
from dataclasses import asdict, dataclass, field

import psutil
import tensorflow as tf


@dataclass
class ResourceControls:
    intra_op_threads: int = 0
    inter_op_threads: int = 0
    graph_workers: int = 2
    tf_data_prefetch: int = 2
    tf_data_parallel_calls: int = 1
    graph_cache_size: int = 64
    memory_growth: bool = True
    mixed_precision: bool = True
    xla: bool = False
    batch_size: int = 16
    device: str = "gpu"

    def apply(self) -> None:
        if self.intra_op_threads > 0:
            tf.config.threading.set_intra_op_parallelism_threads(self.intra_op_threads)
        if self.inter_op_threads > 0:
            tf.config.threading.set_inter_op_parallelism_threads(self.inter_op_threads)
        tf.config.optimizer.set_jit(bool(self.xla))
        policy = "mixed_float16" if self.mixed_precision else "float32"
        tf.keras.mixed_precision.set_global_policy(policy)
        if self.memory_growth:
            for gpu in tf.config.list_physical_devices("GPU"):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass


@dataclass
class RuntimeTelemetry:
    started_at: float = field(default_factory=time.time)
    batch_construction_sec: list[float] = field(default_factory=list)
    device_transfer_sec: list[float] = field(default_factory=list)
    train_step_sec: list[float] = field(default_factory=list)
    validation_sec: list[float] = field(default_factory=list)
    graph_cache_hits: int = 0
    graph_cache_misses: int = 0
    peak_host_rss_bytes: int = 0
    peak_gpu_memory_bytes: int = 0

    def sample(self) -> None:
        self.peak_host_rss_bytes = max(
            self.peak_host_rss_bytes,
            psutil.Process(os.getpid()).memory_info().rss,
        )
        for device in tf.config.list_logical_devices("GPU"):
            try:
                info = tf.config.experimental.get_memory_info(device.name)
                self.peak_gpu_memory_bytes = max(
                    self.peak_gpu_memory_bytes, int(info.get("peak", 0)),
                )
            except (ValueError, RuntimeError):
                continue

    def to_dict(self) -> dict:
        data = asdict(self)
        total = self.graph_cache_hits + self.graph_cache_misses
        data["graph_cache_hit_rate"] = self.graph_cache_hits / total if total else 0.0
        data["cpu_percent"] = psutil.cpu_percent(interval=None)
        data["elapsed_sec"] = time.time() - self.started_at
        data["thread_settings"] = {
            "intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
            "inter_op": tf.config.threading.get_inter_op_parallelism_threads(),
        }
        return data


def environment_manifest() -> dict:
    build = tf.sysconfig.get_build_info()
    vm = psutil.virtual_memory()
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "tensorflow_build": build,
        "cpu_devices": [item.name for item in tf.config.list_physical_devices("CPU")],
        "gpu_devices": [item.name for item in tf.config.list_physical_devices("GPU")],
        "ram_total_bytes": int(vm.total),
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "tensorflow_environment": {
            key: os.environ.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES", "TF_CPP_MIN_LOG_LEVEL", "TF_ENABLE_ONEDNN_OPTS",
                "TF_DETERMINISTIC_OPS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            ]
        },
    }
