# TileLang 0.1.9 BatchNorm Persistent Path Hang

## Context

Issue: #1041

Normalization skip audit was run with:

```bash
TMPDIR=/home/lyc/Project/TileOPs/.tmp/tvm_tmp
```

The goal was to determine which normalization skips can be restored under
TileLang 0.1.9 and which still represent real failures.

## Results

These normalization tests passed after releasing their temporary skips:

```text
tests/ops/test_instance_norm.py
10 passed in 67.32s

tests/ops/test_fused_add_layer_norm.py
17 passed in 69.37s
```

BatchNorm still has a real hang:

```bash
TMPDIR=/home/lyc/Project/TileOPs/.tmp/tvm_tmp \
timeout 180s python -m pytest \
  'tests/ops/test_batch_norm.py::test_batch_norm_fwd[32-64-spatial0-dtype0-True]' \
  -v -s --tb=short
```

Observed result:

```text
collected 1 item
tests/ops/test_batch_norm.py::test_batch_norm_fwd[32-64-spatial0-dtype0-True]
timeout after 180s
```

`tests/ops/test_norm_ops.py` was also checked:

```text
TestBatchNormFwdValidation::* passed
TestBatchNormCustomOp::test_fwd_torch_compile_smoke passed
TestBatchNormCustomOp::test_bwd_torch_compile_smoke timed out after 180s
```

## Minimal Kernel Triage

The smallest forward training case hangs after kernel launch, during CUDA
synchronization:

```python
from tileops.ops.norm.batch_norm import BatchNormFwdOp

op = BatchNormFwdOp(32, 64, dtype=torch.float16)
x_cl, _ = op._prepare(x)
y_cl, mean, rstd = op.train_kernel(x_cl, weight.float(), bias.float(), rm, rv)
torch.cuda.synchronize()  # hangs
```

For this shape the default config is:

```text
BatchNormFwdTrainKernel: {'block_l': 32, 'num_stages': 1, 'threads': 32}
BatchNormFwdInferKernel: {'block_l': 32, 'num_stages': 0, 'threads': 32}
```

Forcing the same forward training kernel to `num_stages=0` completes:

```python
BatchNormFwdTrainKernel(
    C=64,
    L=32,
    dtype=torch.float16,
    config={"block_l": 32, "num_stages": 0, "threads": 32},
)
```

Observed result:

```text
launched
synced torch.Size([64, 32]) torch.Size([64])
```

The backward persistent path shows the same behavior. With:

```python
BatchNormBwdKernel(
    C=8,
    L=64,
    dtype=torch.float16,
    config={"block_l": 64, "num_stages": 1, "threads": 64},
)
```

the kernel launches and then hangs at `torch.cuda.synchronize()`. Forcing:

```python
config = {"block_l": 64, "num_stages": 0, "threads": 64}
```

completes successfully.

## Current Assessment

The remaining BatchNorm issue is not caused by pytest, `TMPDIR`, Python
wrapping, or torch.compile itself. It is a TileLang 0.1.9 kernel runtime hang
in the BatchNorm persistent path when `num_stages=1` is used with `T.Pipelined`
and `T.copy`.

The likely fix is to disable pipelining for the persistent BatchNorm forward
training and backward paths by changing persistent default/autotune configs
from `num_stages=1` to `num_stages=0`, then rerun:

```bash
TMPDIR=/home/lyc/Project/TileOPs/.tmp/tvm_tmp python -m pytest \
  tests/ops/test_batch_norm.py tests/ops/test_norm_ops.py -v --tb=short
```

Until that fix lands, keep these skips:

```text
tests/ops/test_batch_norm.py
tests/ops/test_norm_ops.py::TestBatchNormCustomOp::test_fwd_torch_compile_smoke
tests/ops/test_norm_ops.py::TestBatchNormCustomOp::test_bwd_torch_compile_smoke
```
