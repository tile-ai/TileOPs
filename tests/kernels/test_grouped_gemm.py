"""Grouped GEMM autotune: what `tune=True` measures."""

import pytest
import torch

from tileops.kernels.gemm.grouped import GroupedGemmKernel


class _FakeKernelParam:
    """The part of TileLang's ``KernelParam`` its tensor supplier reads."""

    def __init__(self, dtype: str, shape: list[int]) -> None:
        self.dtype = dtype
        self.shape = shape

    def torch_dtype(self):
        return getattr(torch, self.dtype)

    def __getattr__(self, name):  # is_unsigned / is_float8 / is_float4 / is_boolean
        return lambda: False


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_supply_prog_keeps_every_row_in_the_k_loop():
    """Random int32 metadata drops the NT/NN guard sum to ~0 and every tile skips the K-loop."""
    batch_sum, batch_count, n, k = 64, 8, 32, 32
    kernel = GroupedGemmKernel(batch_sum, batch_count, n, k, torch.float16)
    # TileLang supplies inputs only, so the ``out_idx=[2]`` output is absent.
    params = [
        _FakeKernelParam("float16", [batch_sum, k]),
        _FakeKernelParam("float16", [batch_count, n, k]),
        *(_FakeKernelParam("int32", [batch_count]) for _ in range(2)),
    ]
    supplied = kernel.autotune_supply_prog(params)

    assert [list(t.shape) for t in supplied] == [p.shape for p in params]
    sizes, offsets = supplied[2:]
    assert int(sizes.sum()) == batch_sum
    assert int(offsets[0]) == 0 and int(offsets[-1]) == batch_sum - int(sizes[-1])

    # A third such parameter must fail rather than silently receive the offsets.
    with pytest.raises(RuntimeError, match="expects 2 int32"):
        kernel.autotune_supply_prog(params + [_FakeKernelParam("int32", [batch_count])])
