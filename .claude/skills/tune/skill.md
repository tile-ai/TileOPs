---
name: tune
description: GPU kernel profiling and performance tuning methodology for TileOPs — benchmark, nsys, bandwidth analysis
---

# GPU Kernel Profiling Methodology

> Principles for measuring and analysing kernel performance in TileOps. GPU time is the only metric that matters; Python-side overhead is irrelevant to kernel evaluation.

______________________________________________________________________

## 1. Primary Tool: Benchmark Script

**Always use `benchmarks/ops/bench_xxx.py` as the authoritative performance measurement.**

```bash
python -m pytest benchmarks/ops/bench_gemv.py -vvs
cat profile_run.log
```

`BenchmarkBase.profile` uses `tilelang.profiler.do_bench`:

- Measures **GPU execution time only** via CUDA events (or CUPTI hardware counters)
- Python-side call overhead (dispatch, closure creation) is not included — correctly so
- Runs `warmup` iterations to stabilise caches and GPU clocks before measuring
- Reports median latency → stable, reproducible number

**When to trust the benchmark**: always, for comparing kernel variants, configs, or implementations.

______________________________________________________________________

## 2. Deep Analysis: nsys + Fixed Config (No Autotune)

When the benchmark reveals a performance gap and you need to understand _why_, use `nsys profile`. However, you must first **disable autotune and fix the best config** to avoid polluting the trace with all the trial runs:

```python
# In test or benchmark script: pass config= to disable autotune
op = GemvOp(
    n, k, dtype=dtype, config={"block_n": 1, "reduce_threads": 32, "num_stages": 3}
)
```

Then profile:

```bash
nsys profile --trace=cuda --output=/tmp/gemv_fixed python -m pytest benchmarks/ops/bench_gemv.py::test_case -vvs
nsys stats /tmp/gemv_fixed.nsys-rep --report cuda_gpu_kern_sum
```

**Why disable autotune**: `nsys stats` aggregates ALL launches of a kernel name. With autotuning, 15 configs × 20 calls each = 300 slow trial runs are mixed with the 200 steady-state benchmark calls, inflating avg/stddev and masking the true kernel performance.

With a fixed config, nsys shows only the benchmark's steady-state launches — clean, comparable numbers.

______________________________________________________________________

## 3. Reading nsys Output

```
 Time (%)  ... Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Name
     28.9  ...  91 701    85 472    58 240   159 904    24 658      _gemv_main_kernel
```

| Field      | Meaning                                                                  |
| ---------- | ------------------------------------------------------------------------ |
| **Min**    | Best-case execution (optimal cache state, full pipeline, no stalls)      |
| **Med**    | Typical steady-state (more reliable than avg if outliers exist)          |
| **Avg**    | Inflated if slow warm-up or multi-config autotune runs are included      |
| **StdDev** | High → heterogeneous launches (autotune contamination or cache variance) |

**Rule**: compare **median** (not avg) when autotune is disabled. Use **min** to understand the kernel's theoretical best.

______________________________________________________________________

## 4. Effective Bandwidth Calculation

For GEMV `c = B @ a` with `n` rows, `k` columns, `dtype_bytes = 2` (fp16/bf16):

```
mem_bytes = (k + n*k + n) * dtype_bytes    # a + B + c
BW_GB_s   = mem_bytes / latency_s / 1e9
```

H200 peak: 4800 GB/s. Realistic achievable: 3000–4000 GB/s (60–80% of peak) for large GEMV.

When cuBLAS shows two kernels (`*_splitK_*` + `*_splitKreduce_*`), sum their latencies to get the true cuBLAS latency; their memory traffic is also higher (B is read multiple times in split-K mode).

______________________________________________________________________

## 5. Case Study: GEMV (n=7168, k=16384, fp16, H200)

### Without fixed config (autotune contaminated trace)

```
_gemv_main_kernel  Avg=91 701 ns  Med=85 472 ns  StdDev=24 658 ns
```

Uninterpretable — aggregates 15 configs × 20 trial runs + 200 steady-state calls.

### With fixed best config (bn=1, ns=2, 200 steady-state calls)

```
_gemv_main_kernel  Avg=64 261 ns  Med=66 208 ns  StdDev=3 733 ns
cuBLAS splitK      Avg=62 107 ns + 2 354 ns (reduce) ≈ 64 461 ns
```

- Our kernel: **3.55 TB/s** → 74% of H200 peak (4.8 TB/s)
- cuBLAS: **3.64 TB/s** (split-K, 2 passes)
- Both within 5% of each other; benchmark (CUPTI) shows tileops slightly ahead

### Why 74% and not higher

| Factor                      | Impact                                                                                   |
| --------------------------- | ---------------------------------------------------------------------------------------- |
| Warp occupancy: 54/64 = 84% | Fixed by problem size (n=7168 / 132 SMs)                                                 |
| HBM latency hiding          | Covered by warp switching (54 warps/SM); cp.async adds marginal benefit                  |
| 4-way shmem bank conflicts  | Shmem reads ~4× longer, but shmem is NOT the bottleneck (HBM loading dominates)          |
| Realistic HBM utilisation   | H200 achieves 60–80% of rated 4.8 TB/s in practice due to latency and row buffer effects |

**Conclusion**: 74% efficiency is near the practical ceiling for this problem size. Fixing bank conflicts or adding pipeline stages gives diminishing returns when warp switching already hides HBM latency.

```
1. Run benchmark → get GB/s and latency_ms for each shape
2. If result is good vs baseline → done; document in tune-multiplication/skill.md
3. If there's a gap → disable autotune, fix best config
4. nsys profile with fixed config → get clean per-kernel stats
5. Use ncu (Nsight Compute) for detailed per-metric analysis:
   - Memory throughput (L1, L2, HBM hit rates)
   - Warp occupancy and stall reasons
   - Shared memory bank conflict rate
   ⚠ ncu requires exclusive write access to /tmp/nsight-compute-lock.
     On shared systems this lock may be held by another user (sticky-bit /tmp
     prevents deletion). Fix: set TMPDIR to a user-owned directory before running:
       mkdir -p /home/$USER/ncu_tmp && export TMPDIR=/home/$USER/ncu_tmp
     Use --launch-skip N --launch-count M to skip warmup and capture only steady-state kernels:
       ncu --set full --launch-skip 51 --launch-count 3 -o /tmp/report python script.py
     If TMPDIR fix also fails, use nsys profile as the fallback — it covers timeline and kernel-level stats.
6. Identify bottleneck → implement fix → re-benchmark to confirm
7. Open PR — body must contain ONLY:
   - Performance tables (before/after/baseline BW, H200 utilization %)
   - Autotune best configs table
   - `Closes #NNN` for the relevant issue
   Omit implementation narrative, code diffs, and analysis prose — those belong in skill.md.
```
