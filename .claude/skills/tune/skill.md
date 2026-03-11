---
name: tune
description: GPU kernel profiling and performance tuning methodology for TileOPs — benchmark, nsys, ncu workflow
---

# GPU Kernel Profiling Methodology

> Process for measuring and analysing kernel performance in TileOps.
> GPU time is the only metric that matters; Python-side overhead is irrelevant to kernel evaluation.
> Hardware peak numbers and profiler API details live in the wiki KB — this skill references them.

______________________________________________________________________

## 1. Primary Tool: Benchmark Script

**Always use `benchmarks/ops/bench_xxx.py` as the authoritative performance measurement.**

```bash
python -m pytest benchmarks/ops/bench_gemv.py -vvs
cat profile_run.log
```

`BenchmarkBase.profile` uses `tilelang.profiler.do_bench` which measures GPU execution time only (CUDA events), excludes Python overhead, and reports median latency.

→ Full `do_bench` semantics: [Language Spec — Debug & Profiling](https://github.com/tile-ai/TileOPs/wiki/TileLang-Language-Spec#debug--profiling)

**When to trust the benchmark:** always, for comparing kernel variants, configs, or implementations.

______________________________________________________________________

## 2. Deep Analysis: nsys + Fixed Config (No Autotune)

When the benchmark reveals a performance gap and you need to understand *why*, use `nsys profile`. First **disable autotune and fix the best config** — otherwise all autotune trial runs contaminate the trace.

```python
# Pass config= to disable autotune
op = GemvOp(
    n, k, dtype=dtype, config={"block_n": 1, "reduce_threads": 32, "num_stages": 3}
)
```

Then profile:

```bash
nsys profile --trace=cuda --output=/tmp/gemv_fixed python -m pytest benchmarks/ops/bench_gemv.py::test_case -vvs
nsys stats /tmp/gemv_fixed.nsys-rep --report cuda_gpu_kern_sum
```

**Why disable autotune:** `nsys stats` aggregates all launches of a kernel name. With autotuning, many configs × trial runs are mixed with steady-state benchmark calls, inflating avg/stddev and masking true kernel performance.

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
| **Avg**    | Inflated if autotune trial runs are included                             |
| **StdDev** | High → heterogeneous launches (autotune contamination or cache variance) |

**Rule:** compare **median** when autotune is disabled. Use **min** to understand the kernel's theoretical best.

______________________________________________________________________

## 4. Effective Bandwidth

For GEMV `c = B @ a`:

```
mem_bytes = (k + n*k + n) * dtype_bytes    # a + B + c
BW_GB_s   = mem_bytes / latency_s / 1e9
```

→ H200 peak bandwidth, realistic achievable range, and formula derivation: [Hardware Constraints — Effective Bandwidth](https://github.com/tile-ai/TileOPs/wiki/TileLang-Hardware-Constraints#h200-effective-memory-bandwidth)

→ cuBLAS split-K issues two kernels; sum their latencies for comparison: [Hardware Constraints — cuBLAS split-K](https://github.com/tile-ai/TileOPs/wiki/TileLang-Hardware-Constraints#cublas-split-k-for-gemv)

______________________________________________________________________

## 5. Tuning Workflow (SOP)

```
1. Run benchmark → get GB/s and latency_ms for each shape
2. If result is good vs baseline → done; document in tune-multiplication/skill.md case studies
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
     Use --launch-skip N --launch-count M to skip warmup and capture steady-state kernels:
       ncu --set full --launch-skip 51 --launch-count 3 -o /tmp/report python script.py
     If TMPDIR fix also fails, use nsys profile as the fallback.
6. Identify bottleneck → implement fix → re-benchmark to confirm
7. Open PR — body must contain ONLY:
   - Performance tables (before/after/baseline BW, H200 utilization %)
   - Autotune best configs table
   - `Closes #NNN` for the relevant issue
   Omit implementation narrative, code diffs, and analysis prose — those belong in skill.md.
```

______________________________________________________________________

## 6. Case Studies

| Case | Summary                                                                         | Detail                                                                                                                            |
| ---- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| C    | nsys: autotune contamination makes Avg/StdDev uninterpretable; fix config first | [Case Studies C](https://github.com/tile-ai/TileOPs/wiki/TileLang-Case-Studies#case-c-nsys-profiling-with-autotune-contamination) |

______________________________________________________________________

## 7. References

- [Hardware Constraints — H200 Bandwidth](https://github.com/tile-ai/TileOPs/wiki/TileLang-Hardware-Constraints#h200-effective-memory-bandwidth)
- [Language Spec — Debug & Profiling](https://github.com/tile-ai/TileOPs/wiki/TileLang-Language-Spec#debug--profiling)
- [TileLang Case Studies](https://github.com/tile-ai/TileOPs/wiki/TileLang-Case-Studies)
