"""Tests for the decode subset of GroupedQueryAttentionPagedFwdOp."""

import pytest
import torch

from tileops.ops import GroupedQueryAttentionPagedFwdOp
from workloads.attention.gqa import GroupedQueryAttentionPagedDecodeWorkload


def _check(op, workload, inputs) -> None:
    torch.testing.assert_close(op(*inputs), workload.ref_program(*inputs), atol=1e-3, rtol=1e-2)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype,sm_scale,softcap",
    [
        pytest.param(torch.float16, None, None, id="fp16"),
        pytest.param(torch.bfloat16, None, None, id="bf16"),
        pytest.param(torch.float16, 0.25, None, id="custom-scale"),
        pytest.param(torch.float16, None, 2.0, id="softcap"),
    ],
)
def test_gqa_paged_decode_subset(
    dtype: torch.dtype, sm_scale: float | None, softcap: float | None
) -> None:
    test = GroupedQueryAttentionPagedDecodeWorkload(
        2, 16, 4, 1024, 128, 128, dtype, sm_scale=sm_scale, softcap=softcap
    )
    inputs = test.gen_inputs()
    inputs[4].copy_(torch.tensor([513, 1000], device="cuda", dtype=torch.int32))
    op = GroupedQueryAttentionPagedFwdOp(sm_scale=sm_scale, softcap=softcap)
    _check(op, test, inputs)


@pytest.mark.smoke
def test_gqa_paged_decode_selects_bs1_kernel() -> None:
    test = GroupedQueryAttentionPagedDecodeWorkload(1, 32, 4, 8192, 128, 256, torch.float16)
    inputs = test.gen_inputs()
    op = GroupedQueryAttentionPagedFwdOp()
    op(*inputs)
    kernels = list(op.built_kernels("gqa_paged").values())
    assert len(kernels) == 1
    assert kernels[0].__class__.__name__ == "GQADecodePagedBs1Kernel"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_kwargs,input_change,error",
    [
        pytest.param(
            {"window_size_left": 128},
            None,
            "does not yet support sliding windows",
            id="window",
        ),
        pytest.param(
            {},
            "multi-token",
            "requires one query token per request",
            id="multi-token",
        ),
    ],
)
def test_gqa_paged_builtin_capability_boundary(
    op_kwargs: dict, input_change: str | None, error: str
) -> None:
    test = GroupedQueryAttentionPagedDecodeWorkload(2, 8, 2, 512, 64, 64, torch.float16)
    inputs = list(test.gen_inputs())
    if input_change == "multi-token":
        inputs[0] = torch.randn(4, 8, 64, device="cuda", dtype=torch.float16)
        inputs[5] = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    op = GroupedQueryAttentionPagedFwdOp(**op_kwargs)
    with pytest.raises(ValueError, match=error):
        op(*inputs)


@pytest.mark.smoke
def test_gqa_paged_decode_rejects_noncanonical_query_offsets() -> None:
    test = GroupedQueryAttentionPagedDecodeWorkload(2, 8, 2, 512, 64, 64, torch.float16)
    inputs = list(test.gen_inputs())
    inputs[5] = torch.tensor([0, 0, 2], device="cuda", dtype=torch.int32)
    error = "requires one query token per request"
    with pytest.raises(ValueError, match=error):
        GroupedQueryAttentionPagedFwdOp()(*inputs)
    with pytest.raises(ValueError, match=error):
        test.ref_program(*inputs)


@pytest.mark.smoke
def test_gqa_paged_physical_pool_is_independent_of_table_width() -> None:
    test = GroupedQueryAttentionPagedDecodeWorkload(2, 8, 2, 512, 64, 64, torch.float16)
    inputs = list(test.gen_inputs())
    inputs[3] = inputs[3][:, :4].contiguous()
    inputs[4].fill_(4 * test.page_size)
    op = GroupedQueryAttentionPagedFwdOp()
    _check(op, test, inputs)
