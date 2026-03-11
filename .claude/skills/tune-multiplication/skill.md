---
name: tune-multiplication
description: Optimization SOP for matrix-related operators (GEMV, GEMM, Grouped GEMM) in TileOPs
---

# Matrix Multiplication Kernel Tuning Guide

> Optimization process and lessons learned for GEMV, GEMM, and Grouped GEMM kernels in TileOps.
> Hardware numbers (roofline, peak BW, alignment constraints) live in the wiki KB — this skill references them rather than embedding them.

______________________________________________________________________

## 1. Core Principle: Profile the Bottleneck First

Before tuning any kernel, compute the **arithmetic intensity** to classify the bottleneck:

```
Arithmetic Intensity = FLOPs / Bytes
                     = 2MNK / ((MK + KN + MN) × dtype.itemsize)
```

Then compare against the GPU roofline crossover point to determine if the kernel is **memory-bound** or **compute-bound**.

→ GPU peak numbers, crossover points, and roofline table: [Hardware Constraints — GPU Roofline Model](https://github.com/tile-ai/TileOPs/wiki/TileLang-Hardware-Constraints#gpu-roofline-model)

**Typical cases:**

- GEMV (M=1): intensity ≈ 1 FLOPs/Byte → **heavily memory-bound**
- GEMM (large M): high intensity → **compute-bound**, Tensor Core utilization is critical

______________________________________________________________________

## 2. Memory-Bound Kernels (GEMV)

### 2.1 Coalescing Is the Top Priority

Threads within a warp (32 consecutive `threadIdx.x` values) must access consecutive memory addresses.

In TileLang, `threads=(dim_x, dim_y)` maps `threadIdx.x` to the first (fast-varying) dimension. For row-major matrices the fast-varying thread dimension must correspond to the **column index**.

→ Thread layout semantics: [Language Spec — Kernel Launch](https://github.com/tile-ai/TileOPs/wiki/TileLang-Language-Spec#kernel-launch)

**Checklist:**

- [ ] Fast-varying thread dim maps to column index of the row-major matrix
- [ ] Stride between consecutive threads is `sizeof(dtype)` (1–2 bytes)
- [ ] 128-bit vectorized loads: 8 fp16/bf16 elements per transaction

### 2.2 Shared Memory Reuse to Cut Global Traffic

When multiple outputs reuse the same input tile, cache it in shared memory:

```python
a_shared = T.alloc_shared((block_k,), dtype)
for _k in T.vectorized(tile_k):
    a_shared[tk * tile_k + _k] = a[bk * block_k + tk * tile_k + _k]
T.sync_threads()
# use a_shared in FMA instead of global memory
```

**Savings** = `(block_n - 1) / block_n`; larger `block_n` gives more benefit.

→ Bank conflict padding pattern: [Anti-Patterns #27](https://github.com/tile-ai/TileOPs/wiki/TileLang-Anti-Patterns#27-missing-shared-memory-padding-for-bank-conflicts)

> **Note:** On H200, vectors that fit in L1 cache (< 128 KB) are already deduplicated by hardware — explicit shmem for small repeated inputs yields negligible benefit. See [Case Studies B.2](https://github.com/tile-ai/TileOPs/wiki/TileLang-Case-Studies#case-b2-o2-shared-memory-for-a).

### 2.3 Optimal Reduce Configuration

- Use a **full warp (32 threads)** for reduction to leverage `__shfl_down_sync`
- `reduce_threads < 32` causes cross-row access within a warp, breaking coalescing
- TileLang's `tvm_thread_allreduce` maps to efficient warp shuffle when `reduce_threads=32`

### 2.4 Hopper / H200 Specific Features

| Feature                         | Use Case               | TileLang API                         |
| ------------------------------- | ---------------------- | ------------------------------------ |
| TMA (Tensor Memory Accelerator) | Async large-tile loads | `T.use_tma` / pipeline               |
| WarpGroup GEMM (WGMMA)          | Compute-bound GEMM     | `T.gemm` with wgmma backend          |
| 50 MB L2 Cache                  | Small repeated vectors | Automatic — no special handling      |
| cp.async pipeline               | Hide memory latency    | `T.Pipelined()` + `disable_tma=True` |

→ WGMMA alignment requirements, SM90 shared memory limits: [Hardware Constraints](https://github.com/tile-ai/TileOPs/wiki/TileLang-Hardware-Constraints#wgmma-tile-size-constraints)

______________________________________________________________________

## 3. Compute-Bound Kernels (GEMM)

### 3.1 Tensor Core Alignment

WGMMA on SM90 requires specific tile size multiples. Misaligned tile sizes cause Tensor Core utilization to collapse.

→ Exact alignment requirements (M, N, K per dtype): [Hardware Constraints — WGMMA Tile Size Constraints](https://github.com/tile-ai/TileOPs/wiki/TileLang-Hardware-Constraints#wgmma-tile-size-constraints)

### 3.2 Tile Size Selection

For SM90 (H200), the total shared memory usage must stay within the device limit per SM.

→ SM90 shared memory capacity and tile sizing guidance: [Hardware Constraints](https://github.com/tile-ai/TileOPs/wiki/TileLang-Hardware-Constraints#shared-memory-smem)

### 3.3 Double-Buffering Pipeline

For H200 (high BW + high compute), pipeline depth 2–4 is usually optimal. Always use `num_stages >= 1` — `num_stages=0` is semantically invalid.

→ Pipeline pitfall: [Anti-Patterns #25](https://github.com/tile-ai/TileOPs/wiki/TileLang-Anti-Patterns#25-num_stages0-as-sentinel-for-t-pipelined)

______________________________________________________________________

## 4. Autotune Strategy

### 4.1 Search Space Design

**Principle:** Cover hardware-aligned configurations; avoid redundant candidates.

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
# Constraint: total shmem usage ≤ device limit per SM
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

| Case | Summary                                                                                  | Detail                                                                                                                    |
| ---- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| B    | GEMV: `threads=(block_n, reduce_threads)` → 3% peak BW; swap dims → 80%                  | [Case Studies B](https://github.com/tile-ai/TileOPs/wiki/TileLang-Case-Studies#case-b-gemv-uncoalesced-b-access)          |
| B.1  | GEMV O3: pipelined shmem loads via cp.async — helps for small K, marginal for large K    | [Case Studies B.1](https://github.com/tile-ai/TileOPs/wiki/TileLang-Case-Studies#case-b1-o3-pipelined-b-loads)            |
| B.2  | GEMV O2: explicit shmem for `a` — negligible benefit (L1 handles it)                     | [Case Studies B.2](https://github.com/tile-ai/TileOPs/wiki/TileLang-Case-Studies#case-b2-o2-shared-memory-for-a)          |
| B.3  | GEMV large shapes (Llama-3 70B, DeepSeek-V3): 89–90% H200 utilization at n=57344         | [Case Studies B.3](https://github.com/tile-ai/TileOPs/wiki/TileLang-Case-Studies#case-b3-large-shape-validation)          |
| B.4  | GEMV `forward()`: Python closure overhead 11 ms vs 70 µs GPU; cache kernel in `__init__` | [Case Studies B.4](https://github.com/tile-ai/TileOPs/wiki/TileLang-Case-Studies#case-b4-forward-python-closure-overhead) |

______________________________________________________________________

## 7. References

- [Hardware Constraints](https://github.com/tile-ai/TileOPs/wiki/TileLang-Hardware-Constraints)
- [TileLang Anti-Patterns](https://github.com/tile-ai/TileOPs/wiki/TileLang-Anti-Patterns)
- [TileLang Language Spec](https://github.com/tile-ai/TileOPs/wiki/TileLang-Language-Spec)
- [TileLang Case Studies](https://github.com/tile-ai/TileOPs/wiki/TileLang-Case-Studies)
- [CUDA C Programming Guide — Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
