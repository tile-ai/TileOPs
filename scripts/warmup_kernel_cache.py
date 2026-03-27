#!/usr/bin/env python3
"""Pre-compile and autotune all benchmark kernel variants.

Runs every benchmark test to trigger kernel compilation and autotuning,
populating both:
  - tilelang kernel compilation cache  (compiled .so binaries)
  - tilelang autotuner result cache    (best config per kernel)

On subsequent runs, both caches are hit and warmup completes quickly.
The benchmark job then loads cached autotuner results directly instead
of re-profiling all configurations (~2ms vs minutes per kernel).

Profiling runs with real GPU measurements (not dummy values) so that
the cached best-config choices are meaningful.  Parallel pytest-xdist
workers introduce some measurement noise, but the relative config
ranking is preserved well enough for cache seeding.

Uses pytest-xdist to run benchmark test cases in parallel across multiple
workers, while the autotuner's ThreadPoolExecutor parallelizes config
compilation within each op.  Two levels of parallelism:

  Level 1: pytest-xdist workers  (-n flag)  — across ops/benchmarks
  Level 2: ThreadPoolExecutor    (--max-workers) — across autotune configs

Usage:
    python scripts/warmup_kernel_cache.py
    python scripts/warmup_kernel_cache.py --max-workers 64 -n 4
    python scripts/warmup_kernel_cache.py --shard 0 --total-shards 4
"""

import argparse
import glob
import os
import sys


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
        help="Max parallel compilation threads per autotune call (default: 64)")
    parser.add_argument(
        "-n", "--num-pytest-workers", type=int, default=16,
        help="Number of pytest-xdist workers for parallel test execution (default: 16)")
    args = parser.parse_args()

    # Communicate settings to worker processes via environment variables.
    # The conftest plugin (conftest_warmup) reads these in each worker.
    os.environ["TILEOPS_WARMUP_MODE"] = "1"
    os.environ["TILEOPS_WARMUP_MAX_WORKERS"] = str(args.max_workers)

    print(f"Compilation parallelism: {args.num_pytest_workers} pytest workers "
          f"x {args.max_workers} compile threads each")

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

    import pytest

    # Add scripts/ to PYTHONPATH so `-p conftest_warmup` resolves in
    # both the main process and xdist worker subprocesses.
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = scripts_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    pytest_args = [
        *shard_files,
        "-v",
        "--tb=line",
        "-p", "no:cacheprovider",
        "-p", "conftest_warmup",
        "--override-ini=continue_on_collection_errors=true",
    ]

    if args.num_pytest_workers > 1:
        pytest_args.extend(["-n", str(args.num_pytest_workers)])

    exit_code = pytest.main(pytest_args)

    # Distinguish test failures (exit 1) from infrastructure errors (exit 2+).
    # Test failures are expected during warmup (e.g., missing optional deps,
    # GPU OOM from parallel workers) — compilation still succeeds.
    # Infrastructure errors (bad args, internal error, no tests collected)
    # indicate the warmup didn't run at all and should be surfaced.
    print(f"\nWarmup complete (pytest exit code: {exit_code})")
    if exit_code in (0, 1):
        sys.exit(0)
    else:
        print(f"ERROR: warmup failed with infrastructure error (exit code {exit_code})", file=sys.stderr)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
