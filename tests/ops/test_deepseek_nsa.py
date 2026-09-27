"""Test NativeSparseAttention operation."""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import NSAVarlenFwdOp
from workloads.attention.deepseek import NsaFwdWorkload


class NsaFwdTest(NsaFwdWorkload, TestBase):
    pass


class NsaFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, c_seq_len, dim, is_causal, scale, block_size, "
            "groups, selected_blocks, dtype, tune",
            [
                pytest.param(
                    1,
                    16,
                    1024,
                    64,
                    True,
                    0.1,
                    32,
                    16,
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    4,
                    16,
                    8192,
                    64,
                    True,
                    0.1,
                    32,
                    16,
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2,
                    16,
                    8192,
                    64,
                    True,
                    0.1,
                    32,
                    16,
                    4,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@NsaFwdFixture
def test_nsa_varlen_op(
    batch: int,
    heads: int,
    c_seq_len: int,
    dim: int,
    is_causal: bool,
    scale: float,
    block_size: int,
    groups: int,
    selected_blocks: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    assert groups % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NsaFwdTest(
        batch,
        heads,
        c_seq_len,
        dim,
        is_causal,
        scale,
        block_size,
        groups,
        selected_blocks,
        dtype,
    )
    op = NSAVarlenFwdOp(
        is_causal=is_causal,
        scale=scale,
        block_size=block_size,
        tune=tune,
    )
    test.check(op, *test.gen_inputs(), atol=5e-4, rtol=1e-5)
