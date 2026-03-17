"""Pytest conftest plugin for kernel cache warmup mode.

Activated by environment variable TILEOPS_WARMUP_MODE=1 (set by
warmup_kernel_cache.py).  Applies patches in every process, including
pytest-xdist worker subprocesses.

Patches applied:
  1. tilelang.profiler.do_bench → returns dummy latency (skip GPU profiling)
  2. ThreadPoolExecutor max_workers → capped to TILEOPS_WARMUP_MAX_WORKERS
"""

import concurrent.futures
import os
import unittest.mock


def pytest_configure(config):
    """Called in every process (main + xdist workers) before collection."""
    if os.environ.get("TILEOPS_WARMUP_MODE") != "1":
        return

    # --- Patch 1: skip GPU profiling ---
    patcher = unittest.mock.patch("tilelang.profiler.do_bench", return_value=1.0)
    patcher.start()
    # Store on config so we can stop it later if needed.
    config._warmup_patcher = patcher

    # --- Patch 2: cap compilation parallelism ---
    max_workers = int(os.environ.get("TILEOPS_WARMUP_MAX_WORKERS", "64"))
    _OrigPool = concurrent.futures.ThreadPoolExecutor

    class _CappedPool(_OrigPool):
        def __init__(self, max_workers=None, **kwargs):  # noqa: N803
            if max_workers is None or max_workers > _cap:
                max_workers = _cap
            super().__init__(max_workers=max_workers, **kwargs)

    _cap = max_workers
    concurrent.futures.ThreadPoolExecutor = _CappedPool


def pytest_unconfigure(config):
    """Cleanup patches."""
    patcher = getattr(config, "_warmup_patcher", None)
    if patcher is not None:
        patcher.stop()
