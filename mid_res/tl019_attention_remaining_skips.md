# TileLang 0.1.9 Attention Remaining Skips

## Context

Issue: #1041

Attention skip audit was run with:

```bash
TMPDIR=/home/lyc/Project/TileOPs/.tmp/tvm_tmp
```

The goal was to determine which temporary skips can be restored under
TileLang 0.1.9 and which still represent real failures.

## Released Skips

These attention tests passed after releasing their temporary skips:

```text
tests/ops/attention/test_deepseek_nsa.py
3 passed, 3 warnings in 111.32s

tests/ops/attention/test_deepseek_nsa_topk.py
2 passed, 2 warnings in 48.93s

tests/ops/attention/test_deepseek_nsa_cmp.py
1 passed, 1 warning in 16.45s

tests/ops/attention/test_gqa.py::test_gqa_fwd
4 passed

tests/ops/attention/test_gqa_decode.py
3 passed in 32.52s

tests/ops/attention/test_gqa_decode_paged.py
7 passed in 46.45s

tests/ops/attention/test_gqa_sliding_window.py
21 passed in 93.77s
```

`tests/ops/attention/test_gqa_sliding_window_varlen.py` was reduced from a
full-file skip to two exact failing nodeids. The final combined validation
passed:

```text
54 passed, 14 skipped, 6 warnings in 28.95s
```

## Remaining Skips

### GQA Backward

`tests/ops/attention/test_gqa.py::test_gqa_bwd` still fails for all four
parameter sets.

Observed failure:

```text
TypeError: gemm() got an unexpected keyword argument 'wg_wait'
```

This occurs in `tileops/kernels/attention/gqa_bwd.py` while building
`_gqa_bwd_wgmma_pipelined_main`.

### MHA Decode Long Context

The short `s_kv=5` case passes, but the two `s_kv=8192` cases still fail.

Observed failure:

```text
tvm.error.InternalError: Fragment buffer access with non-zero index [2] is not supported.
Only fragment[0] access is allowed within T.Parallel loop.
```

Keep the local `s_kv == 8192` skip in
`tests/ops/attention/test_mha_decode.py`.

### Paged MHA Decode

All four paged MHA decode cases still fail with the same fragment access
layout inference error:

```text
tvm.error.InternalError: Fragment buffer access with non-zero index [2] is not supported.
Only fragment[0] access is allowed within T.Parallel loop.
```

Keep `tests/ops/attention/test_mha_decode_paged.py` skipped.

### DeepSeek MLA / DSA Decode

Both DeepSeek decode tests still fail at TileLang TIR construction:

```text
TypeError: gemm() got an unexpected keyword argument 'wg_wait'
```

Keep local skips in:

```text
tests/ops/attention/test_deepseek_mla_decode.py
tests/ops/attention/test_deepseek_dsa_decode.py
```

### GQA Sliding Window Varlen

Two varlen cases compile and run but fail numerical comparison:

```text
tests/ops/attention/test_gqa_sliding_window_varlen.py::test_gqa_sliding_window_varlen_fwd_op[2-seqlens_q8-seqlens_k8-8-2-64-False-64-64-dtype8-False]
tests/ops/attention/test_gqa_sliding_window_varlen.py::test_gqa_sliding_window_varlen_fwd_op[2-seqlens_q13-seqlens_k13-8-2-64-True-0--1-dtype13-False]
```

Observed mismatches:

```text
case dtype8: 97090 / 393216 mismatched elements, max abs diff 0.474609375
case dtype13: 38836 / 196608 mismatched elements, max abs diff 0.6015625
```

Keep these two exact nodeid skips while restoring the rest of the varlen file.
