#!/usr/bin/env python3
"""Warm TileLang/Triton caches by running a selected pytest subset.

By default this warms all benchmark files under benchmarks/ops. The same
driver can also warm correctness suites by accepting explicit pytest
targets and a marker expression.

Warmup populates:
  - tilelang kernel compilation cache
  - tilelang autotuner result cache (when benchmark paths autotune)

Usage:
    python scripts/warmup_kernel_cache.py
    python scripts/warmup_kernel_cache.py --max-workers 64 -n 4
    python scripts/warmup_kernel_cache.py --shard 0 --total-shards 4
    python scripts/warmup_kernel_cache.py --pytest-targets tests -m smoke
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
    parser.add_argument(
        "--pytest-targets", nargs="*", default=None,
        help="Optional explicit pytest targets to warm instead of benchmarks/ops")
    parser.add_argument(
        "-m", "--marker", default=None,
        help="Optional pytest marker expression for selecting the warmup subset")
    args = parser.parse_args()

    # Communicate settings to worker processes via environment variables.
    # The conftest plugin (conftest_warmup) reads these in each worker.
    os.environ["TILEOPS_WARMUP_MODE"] = "1"
    os.environ["TILEOPS_WARMUP_MAX_WORKERS"] = str(args.max_workers)

    print(f"Compilation parallelism: {args.num_pytest_workers} pytest workers "
          f"x {args.max_workers} compile threads each")

    if args.pytest_targets:
        selected_targets = args.pytest_targets
        print(f"Using explicit pytest targets ({len(selected_targets)}):")
        for target in selected_targets:
            print(f"  {target}")
    else:
        bench_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "ops")
        all_files = sorted(glob.glob(os.path.join(bench_dir, "bench_*.py")))
        selected_targets = all_files[args.shard::args.total_shards]

        if not selected_targets:
            print(f"Shard {args.shard}/{args.total_shards}: no files to process")
            return

        print(f"Shard {args.shard}/{args.total_shards}: "
              f"{len(selected_targets)}/{len(all_files)} benchmark files")
        for f in selected_targets:
            print(f"  {os.path.basename(f)}")

    import pytest

    # Add scripts/ to PYTHONPATH so `-p conftest_warmup` resolves in
    # both the main process and xdist worker subprocesses.
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = scripts_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    pytest_args = [
        *selected_targets,
        "-v",
        "--tb=line",
        "-p", "no:cacheprovider",
        "-p", "conftest_warmup",
        "--override-ini=continue_on_collection_errors=true",
    ]

    if args.marker:
        pytest_args.extend(["-m", args.marker])

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
