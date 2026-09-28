"""Pytest conftest plugin for kernel cache warmup and validation.

Two modes, selected by ``--tileops-warmup``:

--tileops-warmup=compile  (parallel warmup)
  - Skip baseline profiling (only compile/tune tileops kernels)
  - Cap ThreadPoolExecutor to --tileops-warmup-max-workers
  - Release GPU memory after each test

--tileops-warmup=validate  (serial validation)
  - Skip baseline profiling
  - Force autotuner cache miss so it re-tunes on a quiet GPU
  - Correct results overwrite the noisy parallel cache

The validation pass fixes a subtle issue: parallel warmup runs multiple
workers on one GPU, so autotuner latency measurements are inflated by
contention.  The cached "best config" may not be optimal under serial
execution.  Validation re-profiles all configs with exclusive GPU access
and overwrites any misselected entries.  Compilation is instant (`.so`
cache hit from warmup), so only profiling runs — typically a few seconds
per kernel.
"""

import concurrent.futures
import gc


def pytest_addoption(parser):
    """Register the warmup mode and the compile-thread cap."""
    parser.addoption(
        "--tileops-warmup",
        choices=("compile", "validate"),
        default=None,
        help="'compile' warms the kernel caches in parallel; 'validate' re-tunes serially.",
    )
    parser.addoption(
        "--tileops-warmup-max-workers",
        type=int,
        default=64,
        help="Max compile threads per autotune call in compile mode (default: 64).",
    )


def _mode(config):
    return config.getoption("--tileops-warmup")


def pytest_configure(config):
    """Called in every process (main + xdist workers) before collection."""
    if _mode(config) is None:
        return

    # --- Shared: skip baseline profiling ---
    from benchmarks.benchmark_base import BenchmarkBase
    from tileops.ops.op_base import Op

    _orig_profile = BenchmarkBase.profile

    def _warmup_profile(self, functor, *inputs, **kwargs):
        if isinstance(functor, Op):
            return _orig_profile(self, functor, *inputs, **kwargs)
        # Baseline functor — return dummy result to skip profiling
        return {"latency_ms": 0.0}

    BenchmarkBase.profile = _warmup_profile
    config._warmup_orig_profile = _orig_profile

    # --- Warmup-only: cap compilation parallelism ---
    if _mode(config) == "compile":
        max_workers = config.getoption("--tileops-warmup-max-workers")
        orig_pool = concurrent.futures.ThreadPoolExecutor
        config._warmup_orig_pool = orig_pool

        class _CappedPool(orig_pool):
            def __init__(self, max_workers=None, **kwargs):
                if max_workers is None or max_workers > _cap:
                    max_workers = _cap
                super().__init__(max_workers=max_workers, **kwargs)

        _cap = max_workers
        concurrent.futures.ThreadPoolExecutor = _CappedPool

    # --- Validate-only: force autotuner cache miss ---
    if _mode(config) == "validate":
        from tilelang.autotuner.tuner import AutoTuner

        config._validate_orig_load = AutoTuner._load_result_from_disk
        # Return None on disk lookup → forces re-tune with .so cache hit
        AutoTuner._load_result_from_disk = lambda self, key: None
        # Clear in-memory cache in case of prior hits in this process
        AutoTuner._memory_cache.clear()


def pytest_runtest_teardown(item, nextitem):
    """Release GPU memory after each test to prevent OOM across workers."""
    if _mode(item.config) is None:
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
        from benchmarks.benchmark_base import BenchmarkBase

        BenchmarkBase.profile = orig_profile

    orig_pool = getattr(config, "_warmup_orig_pool", None)
    if orig_pool is not None:
        concurrent.futures.ThreadPoolExecutor = orig_pool

    orig_load = getattr(config, "_validate_orig_load", None)
    if orig_load is not None:
        from tilelang.autotuner.tuner import AutoTuner

        AutoTuner._load_result_from_disk = orig_load
