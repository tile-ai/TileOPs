"""Hardware characterization microbenchmark suite.

Migrated from tilelang-microbench. Provides systematic measurement of:
- Memory subsystem: HBM bandwidth, L2 cache, shared memory, latency
- Compute: GEMM throughput (FP16, BF16) on Tensor Cores
- System: sync overhead, atomics, bank conflicts, async copy, warp specialization,
  occupancy, register spill, stream/event overhead
"""
