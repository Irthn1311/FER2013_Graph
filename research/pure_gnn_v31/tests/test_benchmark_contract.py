"""Runtime benchmark structure without executing the expensive benchmark."""

import inspect

from pure_gnn_v31.tools.benchmark_runtime import benchmark_batch_sizes


def test_runtime_benchmark_is_installed_and_contains_both_timed_paths():
    source = inspect.getsource(benchmark_batch_sizes)
    assert '"inference"' in source
    assert '"training_step"' in source
    assert 'get("B32"' in source
    assert "ResourceExhaustedError" in source
