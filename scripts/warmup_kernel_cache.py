#!/usr/bin/env python3
"""Pre-compile all benchmark kernel variants to warm the tilelang cache.

Monkeypatches tilelang.profiler.do_bench to return a dummy value so that
kernel compilation happens normally (populating the disk cache) while GPU
profiling is effectively skipped.

The autotuner result cache is disabled (TILELANG_AUTO_TUNING_DISABLE_CACHE=1)
so that dummy profiling results are never persisted — the real benchmark run
will re-profile from compiled-kernel cache hits and store genuine best configs.

Usage:
    python scripts/warmup_kernel_cache.py
    python scripts/warmup_kernel_cache.py --shard 0 --total-shards 4
    python scripts/warmup_kernel_cache.py --max-workers 64
"""

import argparse
import concurrent.futures
import glob
import os
import sys
import unittest.mock


def _patch_compile_parallelism(max_workers):
    """Cap ThreadPoolExecutor max_workers used by the autotuner."""
    _OrigPool = concurrent.futures.ThreadPoolExecutor
    _cap = max_workers

    class _CappedPool(_OrigPool):
        def __init__(self, max_workers=None, **kwargs):
            if max_workers is None or max_workers > _cap:
                max_workers = _cap
            super().__init__(max_workers=max_workers, **kwargs)

    concurrent.futures.ThreadPoolExecutor = _CappedPool


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compile benchmark kernels to warm tilelang cache")
    parser.add_argument(
        "--shard", type=int, default=0,
        help="Shard index for splitting work across parallel jobs (0-based)")
    parser.add_argument(
        "--total-shards", type=int, default=1,
        help="Total number of shards")
    parser.add_argument(
        "--max-workers", type=int, default=64,
        help="Max parallel compilation threads (default: 64)")
    args = parser.parse_args()

    # Cap compilation parallelism.
    _patch_compile_parallelism(args.max_workers)
    print(f"Compilation parallelism capped at {args.max_workers} threads")

    # Prevent autotuner from caching dummy profiling results.
    os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "1"

    # Monkeypatch do_bench to skip real GPU profiling.
    # This covers:
    #   - BenchmarkBase.profile()          (benchmark harness)
    #   - _profile_manual() in bench_gla   (direct do_bench calls)
    #   - autotuner Phase 2                (config profiling inside tilelang)
    # Compilation (autotuner Phase 1) is unaffected — it does not use do_bench.
    _dummy_latency = unittest.mock.patch(
        "tilelang.profiler.do_bench", return_value=1.0)
    _dummy_latency.start()

    # Also patch the local import in benchmarks that grab do_bench directly.
    # bench_gla.py does: from tilelang.profiler import do_bench as _do_bench
    # The patch above covers the module-level attribute; re-imports before
    # our patch took effect are handled by also patching the profiler module.

    # Collect benchmark files and shard.
    bench_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "ops")
    all_files = sorted(glob.glob(os.path.join(bench_dir, "bench_*.py")))
    shard_files = all_files[args.shard::args.total_shards]

    if not shard_files:
        print(f"Shard {args.shard}/{args.total_shards}: no files to process")
        return

    print(f"Shard {args.shard}/{args.total_shards}: "
          f"{len(shard_files)}/{len(all_files)} benchmark files")
    for f in shard_files:
        print(f"  {os.path.basename(f)}")

    # Run benchmarks via pytest.
    # - Compilation happens during Op construction (autotune compiles all
    #   config variants in parallel via ThreadPoolExecutor).
    # - Profiling returns instantly (do_bench patched).
    # - continue-on-error: individual test failures don't block warmup.
    import pytest

    exit_code = pytest.main([
        *shard_files,
        "-v",
        "--tb=line",
        "-p", "no:cacheprovider",
        "--override-ini=continue_on_collection_errors=true",
    ])

    _dummy_latency.stop()

    # Exit 0 even if some tests "fail" — warmup success is measured by
    # cache population, not test pass/fail.
    print(f"\nWarmup complete (pytest exit code: {exit_code})")
    sys.exit(0)


if __name__ == "__main__":
    main()
