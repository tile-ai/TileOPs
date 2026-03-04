# FFT C2C Kernel Optimization Notes

## Algorithm

Radix-2 Cooley-Tukey decimation-in-time (DIT) FFT on 1D complex arrays.

- **Bit-reversal** permutation: loads input in bit-reversed order into shared memory (SMEM)
- **Butterfly stages**: `log2(N)` stages, each with `N/2` butterflies

Arithmetic intensity (complex64):

```
FLOPs  = 5 * N * log2(N)       # (N/2 butterflies) × log2(N) stages × 10 FLOPs each
Bytes  = 2 * N * 8             # read N complex64 + write N complex64
AI     = 5*log2(N) / 16        # ~2.5 FLOP/B at N=1024 → memory-bound on H200
```

The kernel is memory-bandwidth-bound for all practical sizes. cuFFT stays fast by
minimising global memory round-trips; closing this gap requires fusing all stages.

______________________________________________________________________

## Optimizations Applied (`fft_c2c_lut`)

| Optimization                                                    | Kernel      | Impact                                                     |
| --------------------------------------------------------------- | ----------- | ---------------------------------------------------------- |
| Twiddle LUT (CPU-precomputed, flat tensor size `n-1`)           | LUT stages  | Eliminates `T.cos`/`T.sin` in large-stride stages          |
| Fused bit-reversal + SMEM stages (single kernel launch)         | SMEM kernel | Fuses bit-reversed load + all stages that fit in SMEM      |
| Float32 accumulation for complex64 (`accum_dtype = real_dtype`) | Both        | ~1.2–1.5× speedup; FP32 throughput ~30× FP64 on H200 sm90a |
| Extended autotune `threads ∈ {32,64,128,256,512,1024}`          | SMEM kernel | Full SMEM fusion for N ≤ 1024 (0 LUT stages)               |

**SMEM fusion threshold**: `smem_stages = log2(min(N, 2*threads))`.
When `threads ≥ N/2`, the entire FFT fits in one SMEM kernel (0 LUT stage launches).

Full grid-level fusion (a single kernel for any N) requires CUDA cooperative groups
(grid sync) and is not yet implemented.

______________________________________________________________________

## Benchmark Results (complex64, H200 sm90a, 2026-03-04)

| N      | cuFFT    | `fft_c2c` (base) | `fft_c2c_lut` (optimised) | vs base      | vs cuFFT     |
| ------ | -------- | ---------------- | ------------------------- | ------------ | ------------ |
| 64     | 9.06 µs  | 30.23 µs         | 12.18 µs                  | 2.48× faster | 1.34× slower |
| 128    | 9.13 µs  | 31.76 µs         | 12.16 µs                  | 2.61× faster | 1.33× slower |
| 256    | 8.96 µs  | 34.17 µs         | 12.49 µs                  | 2.74× faster | 1.39× slower |
| 512    | 9.14 µs  | 39.56 µs         | 12.35 µs                  | 3.20× faster | 1.35× slower |
| 1024   | 9.09 µs  | 39.04 µs         | 12.12 µs                  | 3.22× faster | 1.33× slower |
| 4096   | 8.79 µs  | 42.05 µs         | 17.46 µs                  | 2.41× faster | 1.99× slower |
| 16384  | 14.94 µs | 48.84 µs         | 22.99 µs                  | 2.12× faster | 1.54× slower |
| 65536  | 15.06 µs | 63.50 µs         | 30.33 µs                  | 2.09× faster | 2.01× slower |
| 262144 | 15.35 µs | 101.90 µs        | 35.92 µs                  | 2.84× faster | 2.34× slower |

N ≤ 1024: full SMEM fusion (0 LUT stages), ~12 µs = 1.33–1.39× cuFFT.
Large N: O(log N) separate LUT kernel launches dominate; cuFFT gap widens.

______________________________________________________________________

## Remaining Gap

cuFFT advantages not yet replicated:

| Factor                | cuFFT approach                     | Our approach                  |
| --------------------- | ---------------------------------- | ----------------------------- |
| Larger radix          | Radix-4/8 (fewer stages)           | Radix-2 only                  |
| Grid sync             | Fuses all stages in one launch     | Separate LUT kernel per stage |
| Memory coalescing     | Hardware-optimised access patterns | Sequential per-thread access  |
| Warp-level primitives | Shuffle-based butterfly            | Shared memory only            |

Closing the gap for large N requires at minimum: CUDA cooperative groups for grid
synchronisation, or splitting N into tiles that each fit in SMEM.
