"""Run all GPU hardware microbenchmarks with unified output."""

import os
import sys
import time

# Project root for hardware benchmarks
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    from utils import print_env_header

    # Print env header once
    print_env_header()
    print()

    total_start = time.time()

    # 1. Bandwidth first (sets measured peak)
    print("#" * 80)
    print("# 1/5 Bandwidth")
    print("#" * 80)
    t0 = time.time()
    from memory.hbm_bandwidth import benchmark_bandwidth
    benchmark_bandwidth()
    print(f"[bandwidth done in {time.time()-t0:.1f}s]\n")

    # 2. L2 cache
    print("#" * 80)
    print("# 2/5 L2 Cache")
    print("#" * 80)
    t0 = time.time()
    from memory.l2_bandwidth import benchmark_l2_cache
    benchmark_l2_cache()
    print(f"[l2_cache done in {time.time()-t0:.1f}s]\n")

    # 3. Latency
    print("#" * 80)
    print("# 3/5 Latency")
    print("#" * 80)
    t0 = time.time()
    from memory.latency import benchmark_latency
    benchmark_latency()
    print(f"[latency done in {time.time()-t0:.1f}s]\n")

    # 4. GEMM throughput
    print("#" * 80)
    print("# 4/5 GEMM Throughput")
    print("#" * 80)
    t0 = time.time()
    from compute.gemm_throughput import benchmark_gemm_throughput
    benchmark_gemm_throughput()
    print(f"[gemm_throughput done in {time.time()-t0:.1f}s]\n")

    # 5. Shared memory
    print("#" * 80)
    print("# 5/5 Shared Memory")
    print("#" * 80)
    t0 = time.time()
    from memory.shared_bandwidth import benchmark_shared_memory
    benchmark_shared_memory()
    print(f"[shared_memory done in {time.time()-t0:.1f}s]\n")

    total_elapsed = time.time() - total_start
    print("=" * 80)
    print(f"All benchmarks completed in {total_elapsed:.1f}s")
    print(f"CSV results in: {os.path.join(ROOT, 'results')}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
