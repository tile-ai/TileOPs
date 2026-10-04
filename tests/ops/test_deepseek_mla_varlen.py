"""Packed MLA prefill consumes the shared output and log-sum-exp contract."""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.attention import (
    MLAVarlenPrefillFwdKernel,
    MLAVarlenPrefillWSFwdKernel,
)
from tileops.ops import MLAVarlenFwdOp
from workloads.attention.mla import MLAVarlenWorkload, mla_varlen_inputs
from workloads.numerics import compare_outputs


@pytest.mark.smoke
def test_mla_varlen_default_config_is_owned_by_each_kernel() -> None:
    """Changing one kernel's config must not change another kernel or future defaults."""
    budget, shape = 101376, (128, 64, 128, torch.float16.itemsize)
    first = MLAVarlenPrefillFwdKernel._default_config_for(budget, 1, *shape)
    second = MLAVarlenPrefillFwdKernel._default_config_for(budget, 2, *shape)
    expected = second.copy()

    first["block_n"] = 32

    assert second == expected
    assert MLAVarlenPrefillFwdKernel._default_config_for(budget, 1, *shape) == expected


class MLAVarlenFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "seq_lens, heads, dim_nope, dim_pe, dim_v, is_causal, sm_scale, dtype",
                [
                    pytest.param(
                        [256, 384],
                        4,
                        128,
                        64,
                        128,
                        True,
                        None,
                        torch.bfloat16,
                        marks=pytest.mark.smoke,
                    ),
                    pytest.param(
                        [512, 512],
                        8,
                        128,
                        64,
                        128,
                        True,
                        None,
                        torch.float16,
                        marks=pytest.mark.smoke,
                    ),
                    # request lengths that are not multiples of any tile
                    pytest.param(
                        [200, 333, 91],
                        4,
                        128,
                        64,
                        128,
                        True,
                        None,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                    # one request shorter than a single query tile
                    pytest.param(
                        [17, 1024],
                        4,
                        128,
                        64,
                        128,
                        True,
                        None,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                    pytest.param(
                        [256, 256],
                        4,
                        128,
                        64,
                        128,
                        False,
                        None,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                    pytest.param(
                        [256, 256],
                        4,
                        128,
                        64,
                        128,
                        True,
                        0.05,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                    pytest.param(
                        [1024],
                        16,
                        128,
                        64,
                        128,
                        True,
                        None,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                ],
            ),
        ]


@MLAVarlenFwdFixture
def test_mla_varlen_fwd_op(
    seq_lens, heads, dim_nope, dim_pe, dim_v, is_causal, sm_scale, dtype
) -> None:
    workload = MLAVarlenWorkload(
        seq_lens, heads, dim_nope, dim_pe, dim_v, dtype, is_causal, sm_scale
    )
    op = MLAVarlenFwdOp(is_causal=is_causal, sm_scale=sm_scale)
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
def test_mla_varlen_lse_merges_a_split_context() -> None:
    """``lse`` is the natural-base log-sum-exp its own outputs were divided by.

    A caller merging chunked-context partials weights each chunk by
    ``exp(lse_chunk - lse_all)``; that identity is what makes the returned
    ``lse`` usable, and it holds only if the scale is the one the manifest
    states.
    """
    torch.manual_seed(0)
    q, k_nope, k_pe, v, cu_seqlens = mla_varlen_inputs([512], 4, 128, 64, 128, torch.float16)
    op = MLAVarlenFwdOp(is_causal=True)
    out, lse = op(q, k_nope, k_pe, v, cu_seqlens)

    workload = MLAVarlenWorkload([512], 4, 128, 64, 128, torch.float16)
    inputs = (q, k_nope, k_pe, v, cu_seqlens)
    compare_outputs((out, lse), workload.ref_program(*inputs), workload.verification(*inputs))
    assert out.dtype == q.dtype
    assert lse.dtype == torch.float32


@pytest.mark.parametrize(
    "kernel_cls",
    [
        pytest.param(MLAVarlenPrefillFwdKernel, id="general"),
        pytest.param(MLAVarlenPrefillWSFwdKernel, id="ws", marks=[pytest.mark.sm90]),
    ],
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([256, 384], id="ragged", marks=pytest.mark.smoke),
        pytest.param([17, 1024], id="below-one-tile", marks=pytest.mark.full),
    ],
)
def test_each_implementation_matches_the_reference(kernel_cls, seq_lens) -> None:
    """Both implementations answer the same thing.

    The op dispatches to whichever one the device admits, so a test that only
    calls the op leaves the other unexercised on any given machine.
    """
    inputs = mla_varlen_inputs(seq_lens, 4, 128, 64, 128, torch.bfloat16)
    kernel = kernel_cls(
        batch=len(seq_lens),
        heads=4,
        dim_nope=128,
        dim_pe=64,
        dim_v=128,
        is_causal=True,
        dtype=torch.bfloat16,
    )
    workload = MLAVarlenWorkload(seq_lens, 4, 128, 64, 128, torch.bfloat16)
    TestBase.check(workload, MLAVarlenFwdOp(is_causal=True), *inputs, runs=kernel.forward)
