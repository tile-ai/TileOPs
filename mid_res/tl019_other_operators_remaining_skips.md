# TileLang 0.1.9 Other Operators Remaining Skips

## Context

Issue: #1041

Other operators skip audit was run with:

```bash
TMPDIR=/home/lyc/Project/TileOPs/.tmp/tvm_tmp
```

The goal was to determine which temporary skips can be restored under
TileLang 0.1.9 and which still represent real failures.

## Released Skips

These groups passed after releasing their temporary skips:

```text
tests/ops/test_mhc_pre.py tests/ops/test_mhc_post.py
6 passed in 36.46s

tests/ops/test_fft.py
9 passed, 1 warning in 63.35s

tests/ops/test_topk_selector.py
4 passed in 50.48s
```

The final combined validation with the remaining local skips restored passed:

```text
89 passed, 15 skipped, 1 warning in 39.46s
```

## RoPE Remaining Skip

The following RoPE tests still fail when their local skips are released:

```text
tests/ops/test_rope.py::test_rope_non_neox_2d[2-128-8-64-dtype0]
tests/ops/test_rope.py::test_rope_non_neox_2d[2-128-8-64-dtype1]
tests/ops/test_rope.py::test_rope_non_neox_2d[2-128-8-64-dtype2]
tests/ops/test_rope.py::test_rope_non_neox_2d[1-256-4-128-dtype3]
tests/ops/test_rope.py::test_rope_non_neox_edge[1-1-1-16-dtype0]
tests/ops/test_rope.py::test_rope_non_neox_edge[1-1-1-16-dtype1]
tests/ops/test_rope.py::test_rope_non_neox_edge[2-512-8-64-dtype2]
```

Observed failure:

```text
tvm.error.InternalError: Check failed: undefined.size() == 0 (1 vs. 0) :
In PrimFunc main variables [i_j_fused] are used, but are not passed in as API arguments
```

This is a TileLang compile-time failure in `RopeNonNeoxKernel`, not a pytest or
runtime synchronization issue. Keep the local skips in:

```text
tests/ops/test_rope.py::test_rope_non_neox_2d
tests/ops/test_rope.py::test_rope_non_neox_edge
```

## Convolution Remaining Skip

The following convolution tests still fail when their local skips are released:

```text
tests/ops/test_convolution.py::test_conv2d[smoke-fp16-3x3]
tests/ops/test_convolution.py::test_conv2d[smoke-bf16-3x3]
tests/ops/test_convolution.py::test_conv2d[full-small-5x5-s1-fp16]
tests/ops/test_convolution.py::test_conv2d_accepts_zero_bias
tests/ops/test_convolution.py::test_conv3d[smoke-3d-unet-k3-s1-fp16]
tests/ops/test_convolution.py::test_conv3d[smoke-3d-unet-k3-s1-bf16]
tests/ops/test_convolution.py::test_conv3d[full-unet-encoder-k3-s1-bf16]
tests/ops/test_convolution.py::test_conv3d_accepts_zero_bias
```

Observed failure:

```text
tvm.error.InternalError: Check failed: (IsValidCPAsyncTransferBytes(total_bytes)) is false:
tl::ptx_cp_async requires a final PTX byte width in {4, 8, 16}, but got 2
```

The failing Conv2d and Conv3d cases use the TileLang convolution kernels with
`num_stages=3`, which triggers an invalid final `cp.async` transfer width during
CUDA codegen. Several other Conv2d/Conv3d cases pass, so the remaining skips
should stay scoped to `SKIPPED_CONV2D_CASES`, `SKIPPED_CONV3D_CASES`, and the
zero-bias smoke tests in `tests/ops/test_convolution.py`.
