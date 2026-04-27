# TileLang 0.1.9 Linear Attention / Recurrent Remaining Skips

## Context

Issue: #1041

Linear attention and recurrent operator skip audit was run with:

```bash
TMPDIR=/home/lyc/Project/TileOPs/.tmp/tvm_tmp
```

The goal was to determine which temporary skips can be restored under
TileLang 0.1.9 and which still represent real failures.

## Released Skips

These tests passed after releasing their temporary skips:

```text
tests/ops/test_deltanet_fwd.py
8 passed in 176.39s

tests/ops/test_deltanet_chunkwise_bwd.py
6 passed in 238.30s

tests/ops/test_gated_deltanet_fwd.py
8 passed in 193.82s

tests/ops/test_gated_deltanet_chunkwise_bwd.py
6 passed in 280.96s

tests/ops/test_gla_chunkwise_fwd.py
7 passed, 21 warnings in 145.68s

tests/ops/test_gla_chunkwise_bwd.py
6 passed, 42 warnings in 231.62s

tests/ops/test_mamba.py
23 passed in 165.43s
```

The recurrence decode tests were narrowed: fp32 decode and multi-step decode
cases pass, while fp16/bf16 cases remain skipped.

The final combined validation passed:

```text
82 passed, 31 skipped, 45 warnings in 30.59s
```

## Remaining Skips

### DeltaNet Decode Low Precision

The fp32 DeltaNet decode cases pass. The fp16/bf16 decode and multi-step decode
cases still fail:

```text
tests/ops/test_deltanet_recurrence.py::test_deltanet_decode[1-4-64-64-dtype1-False]
tests/ops/test_deltanet_recurrence.py::test_deltanet_decode[1-4-64-64-dtype2-False]
tests/ops/test_deltanet_recurrence.py::test_deltanet_decode[2-8-64-64-dtype5-False]
tests/ops/test_deltanet_recurrence.py::test_deltanet_decode[2-8-64-64-dtype6-False]
tests/ops/test_deltanet_recurrence.py::test_deltanet_decode_multi_step[1-4-64-64-dtype1-False]
tests/ops/test_deltanet_recurrence.py::test_deltanet_decode_multi_step[1-4-64-64-dtype2-False]
tests/ops/test_deltanet_recurrence.py::test_deltanet_decode_multi_step[2-8-64-64-dtype5-False]
tests/ops/test_deltanet_recurrence.py::test_deltanet_decode_multi_step[2-8-64-64-dtype6-False]
```

Observed failure:

```text
tvm.error.InternalError: Check failed:
(az->CanProveEqual(input_shape_product * rescale_num, shape_product * rescale_den)) is false:
InputShape() = [2, 16, 16] shape = [16, 16], rescale_num = 16, rescale_den = 16
```

These remain exact nodeid skips in `tests/conftest.py`.

### Gated DeltaNet Decode Low Precision

The fp32 gated DeltaNet decode cases pass. The fp16/bf16 decode and multi-step
decode cases still fail with the same TileLang layout reshape error:

```text
InputShape() = [2, 16, 16] shape = [16, 16], rescale_num = 16, rescale_den = 16
```

The low-precision parameter skips remain in
`tests/ops/test_gated_deltanet_recurrence.py`.

### GLA Decode Low Precision

The fp32 GLA decode cases pass after removing the function-level skip. The
fp16/bf16 decode and multi-step decode cases still fail with the same TileLang
layout reshape error:

```text
InputShape() = [2, 16, 16] shape = [16, 16], rescale_num = 16, rescale_den = 16
```

The low-precision parameter skips remain in
`tests/ops/test_gla_recurrence.py`.
