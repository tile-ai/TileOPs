---
name: tune-multiplication
description: Optimization patterns and lessons learned for matrix-related operators (GEMV, GEMM, Grouped GEMM) in TileOPs
---

# Matrix Multiplication Kernel Tuning Guide

> Optimization patterns and lessons learned for matrix-related operators (GEMV, GEMM, Grouped GEMM) in TileOps, written to guide future kernel work.

______________________________________________________________________

## 1. Core Principle: Profile the Bottleneck First

Before tuning any kernel, compute the **arithmetic intensity**:

```
Arithmetic Intensity = FLOPs / Bytes
                     = 2MNK / ((MK + KN + MN) × dtype.itemsize)
```

Compare against the GPU roofline:

| GPU      | Peak FP16    | Memory BW | Roofline crossover |
| -------- | ------------ | --------- | ------------------ |
| H100 SXM | ~1979 TFLOPS | 3.35 TB/s | ~591 FLOPs/Byte    |
| H200 SXM | ~1979 TFLOPS | 4.8 TB/s  | ~412 FLOPs/Byte    |
| A100 SXM | ~312 TFLOPS  | 2.0 TB/s  | ~156 FLOPs/Byte    |

- **Intensity < crossover** → Memory-bound: prioritize memory access patterns
- **Intensity > crossover** → Compute-bound: prioritize Tensor Core utilization

**Typical cases**:

- GEMV (M=1): intensity ≈ 1 FLOPs/Byte → **heavily memory-bound**
- GEMM (large M): high intensity → **compute-bound**, Tensor Core is critical

______________________________________________________________________

## 2. Memory-Bound Kernels (GEMV)

### 2.1 Coalescing Is the Top Priority

**Rule**: Threads within a warp (32 consecutive linear thread IDs) must access consecutive memory addresses.

In TileLang, thread layout determines the access pattern:

```python
# threads=(dim_x, dim_y): threadIdx.x is the fast-varying dimension (varies within a warp)
with T.Kernel(..., threads=(dim_x, dim_y)) as block_idx:
    tx = T.get_thread_binding(0)  # threadIdx.x — varies within a warp
    ty = T.get_thread_binding(1)  # threadIdx.y — varies across warps
```

**Lesson from GEMV (issue #232)**:

Original `threads=(block_n, reduce_threads)` made `tn = threadIdx.x`, so threads within a warp accessed different rows of B at the same column:

```
warp: B[row+0, col], B[row+1, col], ..., B[row+31, col]
stride = K × sizeof(dtype) ≈ 32 KB (for k=16384) → 32-way strided access
```

Fix: swap to `threads=(reduce_threads, block_n)` so `tk = threadIdx.x`; the warp now accesses consecutive columns of the same row:

```
warp: B[row, col+0:8], B[row, col+8:16], ..., B[row, col+248:256]
→ 512 bytes fully coalesced, approaching peak bandwidth
```

**Checklist**:

- [ ] Row-major matrix: fast-varying thread dim must correspond to the column index
- [ ] Stride between consecutive threads must be `sizeof(dtype)` (1–2 bytes)
- [ ] Use 128-bit vectorized loads: 8 fp16/bf16 elements per transaction

### 2.2 Shared Memory Reuse to Cut Global Traffic

When multiple outputs reuse the same input tile, cache it in shared memory:

```python
a_shared = T.alloc_shared((block_k,), dtype)
# Load once; all block_n rows share this tile
for _k in T.vectorized(tile_k):
    a_shared[tk * tile_k + _k] = a[bk * block_k + tk * tile_k + _k]
T.syncthreads()
# Use a_shared in FMA instead of global memory
```

**Savings** = `(block_n - 1) / block_n`; larger `block_n` gives more benefit.

**Bank conflict note**:

- Shared memory has 32 banks of 4 bytes each
- fp16: 2 elements/bank; `tile_k=8` spans 4 banks — no conflict
- If conflicts occur, add padding: `T.alloc_shared((block_k + 1,), dtype)`

### 2.3 Optimal Reduce Configuration

For memory-bound reductions:

- Use a **full warp (32 threads)** for reduction to leverage `__shfl_down_sync` hardware
- `reduce_threads < 32` causes cross-row access within a warp, breaking coalescing
- TileLang's `tvm_thread_allreduce` maps to the most efficient warp shuffle when `reduce_threads = 32`

### 2.4 Hopper / H200 Specific Features

| Feature                         | Use Case                        | TileLang API (reference)        |
| ------------------------------- | ------------------------------- | ------------------------------- |
| TMA (Tensor Memory Accelerator) | Async large-tile loads          | `T.use_tma` / pipeline          |
| WarpGroup GEMM (WGMMA)          | Compute-bound GEMM              | `T.gemm` with wgmma backend     |
| 50 MB L2 Cache                  | Small repeated vectors/matrices | Automatic — no special handling |
| cp.async pipeline               | Hide memory latency             | `T.pipeline()`                  |

______________________________________________________________________

## 3. Compute-Bound Kernels (GEMM)

### 3.1 Tensor Core Alignment

WGMMA on SM90 requires:

- M: multiple of 64 (warp group = 4 warps = 128 threads)
- N: multiple of 8
- K: multiple of 16 for fp16/bf16; multiple of 32 for fp8

Misaligned tile sizes cause Tensor Core utilization to collapse.

### 3.2 Tile Size Selection

For SM90 (H200):

```
block_M × block_N × block_K × 2 × dtype_bytes ≤ 192 KB shared memory
block_M = 64 or 128   (WGMMA alignment)
block_N = 128–256
block_K = 32–64       (minimum unit for latency hiding)
```

### 3.3 Double-Buffering Pipeline

```python
# Prefetch tile k+1 while computing tile k
# TileLang controls pipeline depth via pipeline stages parameter
```

For H200 (high BW + high compute), pipeline depth 2–4 is usually optimal.

______________________________________________________________________

## 4. Autotune Strategy

### 4.1 Search Space Design

**Principle**: Cover hardware-aligned configurations; avoid redundant candidates.

Recommended search space for GEMV (SM90):

```python
block_n_list = [1, 2, 4, 8, 16, 32]  # rows per block
reduce_threads_list = [32]  # full warp — guarantees coalescing
extra = [
    {"block_n": 64, "reduce_threads": 16},
    {"block_n": 128, "reduce_threads": 16},
    {"block_n": 256, "reduce_threads": 32},
]
```

Recommended search space for GEMM (SM90):

```python
block_M = [64, 128]
block_N = [64, 128, 256]
block_K = [32, 64]
# Constraint: block_M × block_N × block_K × 2 × dtype_bytes ≤ 192 KB
```

### 4.2 Autotune Best Practices

- Use at least `warmup=10, rep=10` (current codebase already does this)
- Different shapes may have different optimal configs — tune per critical shape
- Log results to `autotuner.log` to avoid re-tuning
- GPU must be in a stable, low-interference state during tuning

______________________________________________________________________

## 5. Tuning Workflow (SOP)

```
1. Compute arithmetic intensity → classify as Memory-bound or Compute-bound
2. Analyze memory access pattern → verify warp-level coalescing
3. Identify shared memory reuse opportunities
4. Determine thread block layout (dimension order and sizes)
5. Design autotune search space (respect hardware alignment constraints)
6. Run autotune; record best config
7. Validate correctness (unit tests)
8. Compare bandwidth / TFLOPS against cuBLAS / torch baseline
9. File an issue documenting findings; open a PR with the fix
```

______________________________________________________________________

## 6. Case Studies

### Case 1: GEMV Coalescing Fix (2026-02-27)

- **Issue**: [[PERF][GEMV] tile-ai/TileOPs#232](https://github.com/tile-ai/TileOPs/issues/232)

- **Root cause**: `threads=(block_n, reduce_threads)` → stride-K access to B; effective BW ~3%

- **Affected shapes**: `(7168, 16384)`, `(18432, 7168)` in fp16/bf16

- **Changes made**:

  - O1: Swapped thread dims → `threads=(reduce_threads, block_n)`, `tk = threadIdx.x`
  - O3: Default `reduce_threads=32` (full warp), `block_n=16` for SM90
  - O4/O5: Autotune search space expanded to `block_n=[1,2,4,8,16,32]` + `reduce_threads=32`

- **Autotune best configs** (from test run):

  - `(n=7168, k=16384)`: `block_n=8, reduce_threads=32`
  - `(n=18432, k=7168)`: `block_n=1, reduce_threads=32`

- **Benchmark results** vs `torch` baseline (H200, fp16, tuned):

  | Shape (n, k)  | tileops BW (GB/s) | baseline BW (GB/s) | speedup |
  | ------------- | ----------------- | ------------------ | ------- |
  | (7168, 16384) | **3.65**          | 3.34               | 1.09×   |
  | (18432, 7168) | **3.85**          | 3.34               | 1.15×   |
  | (1024, 1024)  | **0.76**          | 0.26               | 2.9×    |

- **H200 peak BW**: 4.8 TB/s → achieved ~80% utilization on large shapes

- **Lesson**: For small shapes (n=k=1024), the gap vs baseline is larger (2.9×) because
  torch's `b @ a` has higher dispatch overhead at small sizes; for large shapes both
  approach the roofline together (~3.3–3.9 TB/s effective BW)

- **Next step (O3)**: Pipeline B loads via `T.Pipelined + T.copy(disable_tma=True)` with
  `num_stages >= 1` to hide HBM3e latency using cp.async

### Case 1 addendum: O3 pipelined B loads (2026-02-27)

**Implementation**:

- Replaced `T.serial` loop + direct register loads with `T.Pipelined + T.copy + b_shared`
- `T.copy(b[bn * block_n, bk * block_k], b_shared, disable_tma=True)` — `disable_tma=True` is required on SM90 to use `cp.async` instead of TMA (TMA needs `mbarrier` that TileLang can't infer for manually-indexed shared memory in non-WGMMA kernels)
- `num_stages=1`: sequential through shared memory (no overlap)
- `num_stages=2`: double-buffer (one tile prefetching while current tile is consumed)
- `num_stages=3`: triple-buffer

**Design decision**: eliminated `num_stages=0` (register-only fallback) entirely. Using 0 as a sentinel for "disable pipeline" is semantically invalid — `T.Pipelined` requires `num_stages >= 1`. The Python `if num_stages > 0:` branch inside `@T.prim_func` caused JIT cache ambiguity and persistent correctness failures. See `.claude/skills/debug/skill.md` Case A for details.

**Autotune space after O3**:

```python
[
    {"block_n": bn, "reduce_threads": 32, "num_stages": ns}
    for bn in [1, 2, 4, 8, 16]
    for ns in [1, 2, 3]
]  # 15 configs total
```

**Autotune winners** (SM90 / H200):

- `(n=7168,  k=16384)`: `block_n=1, num_stages=3`
- `(n=18432, k=7168)`: `block_n=1, num_stages=3`
- `(n=18432, k=7168, bf16)`: `block_n=2, num_stages=3`

**Benchmark results vs torch baseline** (H200, tuned):

| Shape (n, k)  | dtype | tileops BW | baseline BW | speedup |
| ------------- | ----- | ---------- | ----------- | ------- |
| (7168, 16384) | fp16  | 3.47 TB/s  | 3.34 TB/s   | 1.04×   |
| (18432, 7168) | fp16  | 3.83 TB/s  | 3.34 TB/s   | 1.15×   |
| (7168, 16384) | bf16  | 3.62 TB/s  | 3.35 TB/s   | 1.08×   |
| (18432, 7168) | bf16  | 3.78 TB/s  | 3.35 TB/s   | 1.13×   |
| (1024, 1024)  | fp16  | 0.57 TB/s  | 0.26 TB/s   | 2.2×    |

**Important finding**: O3 pipelined shared-memory path is slightly slower than O1 register-only path for `(7168, 16384)` (3.47 vs 3.65 TB/s). Shared memory adds:

- Allocation and TLB overhead for `b_shared`
- Potential bank conflict (32 threads × 8 elements across 32 banks)
- `cp.async` setup cost, especially for small `block_n=1`

For shapes with large K (16384), the register path's latency is already partially hidden by the GPU's out-of-order execution; cp.async adds minimal incremental benefit. For shapes with smaller K (7168), O3 helps more (+14.7%) because HBM latency is less naturally hidden.

### Case 1 addendum: Large shape validation (2026-02-27)

Added LLM production-scale shapes to benchmark and test suite:

- `(n=28672, k=8192)` — Llama-3 70B MLP gate/up projection
- `(n=57344, k=7168)` — DeepSeek-V3 MoE aggregated output (8 experts × 7168)

**Benchmark results** (H200, tuned, `benchmarks/ops/bench_gemv.py`):

| Shape (n, k)  | dtype | tileops BW | baseline BW | speedup | H200 utilization |
| ------------- | ----- | ---------- | ----------- | ------- | ---------------- |
| (7168, 16384) | fp16  | 3.47 TB/s  | 3.34 TB/s   | 1.04×   | 72%              |
| (18432, 7168) | fp16  | 3.80 TB/s  | 3.34 TB/s   | 1.14×   | 79%              |
| (28672, 8192) | fp16  | 4.02 TB/s  | 3.80 TB/s   | 1.06×   | 84%              |
| (57344, 7168) | fp16  | 4.26 TB/s  | 3.92 TB/s   | 1.09×   | 89%              |
| (7168, 16384) | bf16  | 3.61 TB/s  | 3.35 TB/s   | 1.08×   | 75%              |
| (18432, 7168) | bf16  | 3.75 TB/s  | 3.34 TB/s   | 1.12×   | 78%              |
| (28672, 8192) | bf16  | 4.08 TB/s  | 3.81 TB/s   | 1.07×   | 85%              |
| (57344, 7168) | bf16  | 4.30 TB/s  | 3.93 TB/s   | 1.09×   | 90%              |

**Key trend**: larger shapes → higher utilization. `n=57344` gives ~4× more blocks than `n=7168`, placing ~13.5 blocks/SM vs ~3.4 blocks/SM. More warps per SM → better HBM3e latency hiding → 89–90% of peak 4.8 TB/s.

**Lesson**: for GEMV on Hopper, the practical utilization ceiling rises with problem size due to increased warp-level parallelism covering HBM latency. Small shapes (n=1024) remain far below peak due to low occupancy.

### Case 1 addendum: O2 shared memory for `a` (2026-02-27)

**Hypothesis**: caching `a` in shared memory (only `tn==0` warp writes, all rows read)
would reduce `a` global traffic by `(block_n-1)/block_n`, yielding measurable speedup.

**Implementation notes**:

- TileLang sync API is `T.sync_threads()` (NOT `T.syncthreads()` — easy mistake)
- Used Python compile-time flag `use_shmem_a = block_n > 1` to skip shmem + sync
  overhead for `block_n==1` (single warp; registers are faster than shared memory)
- Runtime guard `if tn == 0:` emits a CUDA `if (threadIdx.y == 0)` conditional
- Two `T.sync_threads()` per outer-loop iteration: one after write (consistency),
  one after FMA (prevent next-iteration write-after-read race)

**Actual results** (O1+O2 vs O1 only, H200, fp16):

| Shape (n, k)  | O1 BW     | O1+O2 BW      | delta | Autotune best config |
| ------------- | --------- | ------------- | ----- | -------------------- |
| (7168, 16384) | 3.65 GB/s | **3.67 GB/s** | +0.5% | `block_n=8, rt=32`   |
| (18432, 7168) | 3.85 GB/s | **3.85 GB/s** | 0%    | `block_n=1, rt=32`   |
| (1024, 1024)  | 0.76 GB/s | 0.71 GB/s     | -7%   | (default, no tune)   |

**Lesson (important)**:

- Explicit shared memory for `a` gave **negligible benefit** on H200 large shapes
- Root cause: `a` is tiny (14–32 KB) and fits entirely in L1 cache (256 KB per SM);
  the hardware L1 already deduplicates repeated loads across warps automatically
- For small shapes, shmem + syncthreads overhead caused a slight regression (-7%)
- **Rule**: explicit shared memory for a small repeated input only helps when the
  total working set exceeds L1 capacity; otherwise L1 handles it transparently
- The remaining ~20% gap to H200 peak (4.8 TB/s) is dominated by HBM3e latency,
  not by `a` traffic — prefetching/pipelining would be the right next step

### Case 1 addendum: `forward()` Python overhead fix (2026-02-27)

**Problem**: `GemvKernel.forward` was calling `_gemv_wrapped_kernel` (a `torch.library.custom_op`), which recreates a Python closure + JIT lookup on every forward pass. Wall-clock timing showed ~11ms per call even though the GPU kernel itself runs in ~70μs. The overhead scales with call frequency, not problem size.

**Root cause**: `_gemv_wrapped_kernel` calls `_gemv_kernel(n, k, dtype)(block_n, ...)` — creating a new closure object and triggering a JIT cache lookup each time.

**Fix**: Call `self.kernel(...)` directly in `forward()`. `self.kernel` is populated in `__init__` (after `init_config`/autotune) and hits the in-memory JIT cache:

```python
def forward(self, a, b):
    a = a.flatten().contiguous()
    return self.kernel(
        self.config["block_n"],
        self.config["reduce_threads"],
        self.config["num_stages"],
    )(a, b)
```

`_gemv_wrapped_kernel` is kept for `torch.compile` compatibility.

______________________________________________________________________

## 7. Problems & Solutions Summary (GEMV Optimization)

A consolidated reference table for all issues encountered during the GEMV tuning process:

| #   | Problem                               | Symptom                                       | Root Cause                                                                                                    | Fix                                                                                                                                    |
| --- | ------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| P1  | Uncoalesced B access                  | ~3% of peak BW                                | `threads=(block_n, reduce_threads)` → `tn=threadIdx.x` → stride-K access                                      | Swap to `threads=(reduce_threads, block_n)` so `tk=threadIdx.x`; consecutive threads access consecutive columns                        |
| P2  | `num_stages=0` sentinel               | max_err=0.125 on ALL configs                  | `T.Pipelined` requires `num_stages>=1`; `if ns>0` inside `@T.prim_func` created JIT trace ambiguity           | Remove `else` branch entirely; autotune space: `ns in [1,2,3]` only                                                                    |
| P3  | `a` shared memory overhead            | -7% on small shapes                           | `a` (14–32 KB) fits in L1 (256 KB/SM); hardware already deduplicates; shmem+syncthreads overhead not worth it | Remove O2; rely on L1 cache for `a`                                                                                                    |
| P4  | TMA not usable for manual shmem       | Compilation error (mbarrier)                  | `T.copy` defaults to TMA on SM90; TMA needs mbarrier layout inference not available for non-WGMMA kernels     | `T.copy(..., disable_tma=True)` → `cp.async`                                                                                           |
| P5  | Python closure overhead in forward    | ~11ms wall-clock vs 70μs GPU                  | `_gemv_wrapped_kernel` recreates closure+JIT lookup each call                                                 | Call `self.kernel(...)` directly in `forward()`; keep wrapper for `torch.compile`                                                      |
| P6  | nsys with autotune contaminates trace | Avg=91μs, StdDev=24μs (uninterpretable)       | 15 autotune configs × 20 trial runs aggregated with 200 steady-state launches                                 | Fix config via `GemvOp(config=...)` before nsys; only steady-state launches appear                                                     |
| P7  | ncu requires exclusive lock           | InterprocessLockFailed                        | `/tmp/nsight-compute-lock` held by another user; sticky-bit prevents deletion                                 | Set `TMPDIR` to a user-owned path: `mkdir -p ~/ncu_tmp && export TMPDIR=~/ncu_tmp`; if that also fails, use `nsys profile` as fallback |
| P8  | `T.syncthreads()` not found           | AttributeError                                | Wrong API name                                                                                                | Use `T.sync_threads()` (with underscore)                                                                                               |
| P9  | JIT cache staleness                   | Correctness failures after structural changes | `@tilelang.jit` caches compiled binary; old binary reused if cache key misses new params                      | Clear `~/.tilelang/cache/` after structural kernel changes                                                                             |

**Key lessons**:

- For memory-bound kernels: coalescing (P1) is the single highest-impact fix; everything else is secondary
- Design-level flaws (P2) produce identical errors across all configs; config-level bugs show selective failures
- L1 cache handles small repeated vectors automatically (P3); explicit shmem only helps above L1 capacity
- Always profile with fixed config (P6); autotune contamination makes traces uninterpretable
- GPU time (CUDA events) is the only valid metric for kernel evaluation; wall-clock (P5) includes Python overhead

______________________________________________________________________

## 8. References

- [CUDA C Programming Guide — Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [TileLang Documentation](https://github.com/tile-ai/tilelang)
- [CUTLASS GEMM API Design](https://github.com/NVIDIA/cutlass/blob/main/media/docs/gemm_api.md)
- [Roofline Model for GPU Performance Analysis](https://crd.lbl.gov/assets/pubs_presos/parlab08-roofline-talk.pdf)
