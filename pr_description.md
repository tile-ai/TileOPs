## Summary

- Replace `exp(dA_l[l] - dA_s[s])` with `exp_dA_l[l] * exp(-dA_s[s])` in the intra-chunk causal path, reducing MUFU calls from `block_l×block_s` (4096) to `block_l+block_s` (128) per s-block — a 32× reduction in transcendental ops
- Fuse `exp(-dA_s[ss]) * dt[ss]` into a single `decay_dt_s` shared buffer, saving one shared-memory buffer and one multiply per `(ll, ss)` element in the `lcb_cast` loop; use a local variable for `dA_val` instead of a second shared buffer
- Split the s-block loop into a predicate-free full-lower path and a guarded diagonal path, eliminating branch divergence on lower-triangle blocks; tighten loop bound to skip upper-triangle blocks entirely
- Fix default `block_n` from 32 to `min(128, d_state)` — the autotuner consistently picks 128 for all real Mamba-2 shapes (d_state=128), eliminating 4× unnecessary N-loop iterations
- Replace the 48-config Cartesian autotune search with a focused 10-config set anchored at the known-good default, based on NCU profiling evidence

## Test plan

- [x] pre-commit passed
- [x] All 23 `test_ssd_chunk_scan_fwd_bench` shapes pass correctness (max_err ≤ 1.5e-4)

## Benchmark

**Environment**: NVIDIA H200 (GPU 1), CUDA 12.9, PyTorch 2.11.0.dev20260107. Baseline = upstream `main` at branch point (`f4c360e`). Both runs back-to-back on the same GPU with `tune=True`; autotune picked `block_l=64, block_p=64, block_n=128, block_s=64, threads=128` for all real-model shapes in both runs.

### SSDChunkScanFwdOp (fp16, d_state=128, chunk_len=256)

| shape              | batch | baseline (ms) | this PR (ms) | speedup   | TFLOPS |
| ------------------ | ----- | ------------- | ------------ | --------- | ------ |
| latency-130m-4k    | 1     | 0.0693        | 0.0382       | **1.81×** | 84.4   |
| serving-130m-4k    | 8     | 0.7478        | 0.5059       | **1.48×** | 51.0   |
| longctx-130m-32k   | 4     | 3.0834        | 2.1501       | **1.43×** | 48.0   |
| latency-370m-4k    | 1     | 0.1017        | 0.0546       | **1.86×** | 78.9   |
| serving-370m-4k    | 8     | 0.9695        | 0.6628       | **1.46×** | 51.9   |
| longctx-370m-32k   | 4     | 4.0592        | 2.8199       | **1.44×** | 48.8   |
| throughput-370m-2k | 32    | 1.9743        | 1.3712       | **1.44×** | 50.2   |
| latency-780m-4k    | 1     | 0.1730        | 0.1059       | **1.63×** | 60.9   |
| serving-780m-4k    | 8     | 1.5053        | 1.0258       | **1.47×** | 50.3   |
| longctx-780m-32k   | 4     | 6.2232        | 4.1925       | **1.48×** | 49.3   |
| throughput-780m-2k | 16    | 1.4999        | 1.0204       | **1.47×** | 50.6   |
| latency-1p3b-4k    | 1     | 0.2296        | 0.1513       | **1.52×** | 56.9   |
| serving-1p3b-4k    | 8     | 1.9711        | 1.3469       | **1.46×** | 51.1   |
| longctx-1p3b-32k   | 2     | 4.0638        | 2.7089       | **1.50×** | 50.8   |
| throughput-1p3b-2k | 8     | 0.9766        | 0.6632       | **1.47×** | 51.9   |
| latency-2p7b-4k    | 1     | 0.3042        | 0.1966       | **1.55×** | 54.7   |
| serving-2p7b-4k    | 4     | 1.2643        | 0.8417       | **1.50×** | 51.1   |
| longctx-2p7b-32k   | 2     | 5.1565        | 3.3533       | **1.54×** | 51.3   |
| throughput-2p7b-2k | 4     | 0.6263        | 0.4167       | **1.50×** | 51.6   |

**Takeaways**: **1.43–1.86× speedup across all 19 shapes** (130M–2.7B, all serving regimes). No regressions. Latency shapes gain the most (1.52–1.86×) because the exp-factoring eliminates the dominant MUFU cost at low occupancy. Serving/throughput/longctx shapes converge to 49–52 TFLOPS, consistent with the memory-bandwidth ceiling for these batch sizes.

## MLA / DSA Decode Kernel Benchmark

**Environment**: NVIDIA H200, CUDA 12.9, PyTorch 2.11.0.dev20260107+cu129. `TILELANG_CLEANUP_TEMP_FILES=1` set as a workaround for a shared `/tmp/tvm-debug-mode-tempdirs` permission issue on this machine.

### Before (pre-fix code, official TileLang 0.1.8 `pypi:tilelang==0.1.8`)

All 4 benchmark shapes **fail** with `tvm.error.InternalError: no available layout found` during `LayoutInference`. This is the bug this PR fixes — the unguarded 384-thread `T.copy` on the same shared buffer as the 128-thread `T.wgmma_gemm` triggers the layout conflict.

```
FAILED bench_deepseek_mla_decode.py::test_mla_decode_bench[mainstream-fp16]  # AutoTune: all configs fail LayoutInference
FAILED bench_deepseek_mla_decode.py::test_mla_decode_bench[mid-cache-fp16]   # AutoTune: all configs fail LayoutInference
FAILED bench_deepseek_dsa_decode.py::test_dsa_decode_bench[single-batch-mainstream]  # no available layout found
FAILED bench_deepseek_dsa_decode.py::test_dsa_decode_bench[longer-kv-lower-topk]     # no available layout found
```

### After (this PR, TileLang 0.1.9 `flashmlaenv`)

All 4 benchmark shapes pass. Results below (4/4 passed).

### MultiHeadLatentAttentionDecodeWithKVCacheFwdOp (fp16, tune=True)

| shape           | batch | heads | seq_len_kv | dim | dim_pe | tileops (ms) | torch-ref (ms) | speedup   | tileops TFLOPS | best config                                                    |
| --------------- | ----- | ----- | ---------- | --- | ------ | ------------ | -------------- | --------- | -------------- | -------------------------------------------------------------- |
| mainstream-fp16 | 32    | 128   | 8192       | 512 | 64     | 0.3027       | 0.8910         | **2.94×** | 241.2          | block_H=64, block_N=64, num_split=2, num_stages=1, threads=384 |
| mid-cache-fp16  | 16    | 128   | 4096       | 512 | 64     | 0.0650       | 0.1498         | **2.30×** | 280.9          | block_H=64, block_N=64, num_split=4, num_stages=3, threads=384 |

### DeepSeekSparseAttentionDecodeWithKVCacheFwdOp (fp16, tune=False)

| shape                   | batch | heads | seq_len_q | seq_len_kv | dim | dim_tail | topk | tileops (ms) | torch-ref (ms) | speedup    | tileops TFLOPS | best config             |
| ----------------------- | ----- | ----- | --------- | ---------- | --- | -------- | ---- | ------------ | -------------- | ---------- | -------------- | ----------------------- |
| single-batch-mainstream | 1     | 128   | 1024      | 2048       | 512 | 64       | 2048 | 2.8308       | 31.2540        | **11.04×** | 206.3          | block_i=64, threads=384 |
| longer-kv-lower-topk    | 1     | 128   | 512       | 4096       | 512 | 64       | 1024 | 0.7502       | 31.6637        | **42.21×** | 194.7          | block_i=64, threads=384 |

**Takeaways**: Both ops compile and run cleanly under TileLang 0.1.9 after the fix. MLA achieves 2.30–2.94× over torch-ref; DSA achieves 11–42× over torch-ref (unoptimized baseline). The `tune=True` path for MLA completed successfully for both shapes.
