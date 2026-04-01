"""Pytest conftest plugin for kernel cache warmup mode.

Activated by environment variable TILEOPS_WARMUP_MODE=1 (set by
warmup_kernel_cache.py).  Applies patches in every process, including
pytest-xdist worker subprocesses.

Patches applied:
  1. ThreadPoolExecutor max_workers → capped to TILEOPS_WARMUP_MAX_WORKERS
  2. GPU memory released after each test to prevent OOM across workers

Note: profiling is NOT patched — real GPU measurements run so that
autotuner results can be cached for the benchmark job.
"""

import concurrent.futures
import gc
import os


def pytest_configure(config):
    """Called in every process (main + xdist workers) before collection."""
    if os.environ.get("TILEOPS_WARMUP_MODE") != "1":
        return

    # --- Patch 1: skip baseline profiling (only compile/tune tileops kernels) ---
    from benchmarks.benchmark import BenchmarkBase
    from tileops.ops.op import Op

    _orig_profile = BenchmarkBase.profile

    def _warmup_profile(self, functor, *inputs, **kwargs):
        if isinstance(functor, Op):
            return _orig_profile(self, functor, *inputs, **kwargs)
        # Baseline functor — return dummy result to skip profiling
        return {"latency_ms": 0.0}

    BenchmarkBase.profile = _warmup_profile
    config._warmup_orig_profile = _orig_profile

    # --- Patch 2: cap compilation parallelism ---
    max_workers = int(os.environ.get("TILEOPS_WARMUP_MAX_WORKERS", "64"))
    orig_pool = concurrent.futures.ThreadPoolExecutor
    config._warmup_orig_pool = orig_pool

    class _CappedPool(orig_pool):
        def __init__(self, max_workers=None, **kwargs):  # noqa: N803
            if max_workers is None or max_workers > _cap:
                max_workers = _cap
            super().__init__(max_workers=max_workers, **kwargs)

    _cap = max_workers
    concurrent.futures.ThreadPoolExecutor = _CappedPool


def pytest_runtest_teardown(item, nextitem):
    """Release GPU memory after each test to prevent OOM across xdist workers.

    Without this, each pytest-xdist worker accumulates GPU memory from compiled
    kernels and tensors over its lifetime.  Since workers are long-lived
    processes, idle workers hold onto GPU memory that active workers need,
    eventually exhausting the device and causing the last worker to hang.
    """
    if os.environ.get("TILEOPS_WARMUP_MODE") != "1":
        return
    try:
        import torch
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except (ImportError, AttributeError):
        pass


def pytest_unconfigure(config):
    """Cleanup all patches."""
    orig_profile = getattr(config, "_warmup_orig_profile", None)
    if orig_profile is not None:
        from benchmarks.benchmark import BenchmarkBase
        BenchmarkBase.profile = orig_profile

    orig_pool = getattr(config, "_warmup_orig_pool", None)
    if orig_pool is not None:
        concurrent.futures.ThreadPoolExecutor = orig_pool
