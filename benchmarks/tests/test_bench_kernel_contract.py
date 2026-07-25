"""Contract tests for ``bench_kernel`` fallback and error propagation."""

import pytest
import torch

from benchmarks.benchmark_base import _bench_meta, bench_kernel

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_projection_failure_falls_back_to_cuda_events():
    """A callable launching no CUDA kernel projects no annotation windows;
    bench_kernel must fall back and mark the deviating timing method."""
    latency = bench_kernel(lambda: sum(range(64)), n_warmup=1, n_repeat=2, n_trials=1)
    assert latency >= 0.0
    assert _bench_meta.timing == "cuda-events"


def test_kernel_runtime_error_propagates():
    """Genuine RuntimeErrors must reach the caller, not the fallback path."""
    def boom():
        raise RuntimeError("kernel failure")

    with pytest.raises(RuntimeError, match="kernel failure"):
        bench_kernel(boom, n_warmup=0, n_repeat=1, n_trials=1)
