"""Tests for multi-head attention paged decode."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops import MHADecodePagedWithKVCacheFwdOp
from workloads.attention.mha import MHADecodePagedWorkload
from workloads.numerics import compare_outputs


class MHADecodePagedTest(MHADecodePagedWorkload, TestBase):
    pass


class MHADecodePagedFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune",
            [
                pytest.param(
                    1,
                    16,
                    1,
                    512,
                    128,
                    128,
                    False,
                    torch.float16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="attention")],
                ),
                # bfloat16 dispatch: the same signature admits it and the paged
                # decode kernels are selected on dtype.
                pytest.param(
                    1,
                    16,
                    1,
                    1024,
                    128,
                    128,
                    False,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1,
                    8,
                    1,
                    1024,
                    64,
                    256,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2,
                    8,
                    1,
                    1024,
                    64,
                    256,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1,
                    8,
                    1,
                    512,
                    64,
                    256,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@MHADecodePagedFixture
def test_mha_decode_paged_op(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MHADecodePagedTest(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
    op = MHADecodePagedWithKVCacheFwdOp(page_size=page_size, is_causal=is_causal)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "seqlen_q, is_causal, real_lengths",
    [
        pytest.param(1, False, [1], marks=pytest.mark.smoke),
        # Several queries; a cache shorter than the query block leaves the first ones no key.
        pytest.param(2, True, [65], marks=pytest.mark.smoke),
        pytest.param(4, True, [700, 3], marks=pytest.mark.smoke),
        pytest.param(4, False, [700, 3], marks=pytest.mark.smoke),
        pytest.param(1, False, [37], marks=pytest.mark.full),
        pytest.param(1, False, [700, 1024, 1], marks=pytest.mark.full),
    ],
)
def test_mha_decode_paged_cache_shorter_than_bound(
    seqlen_q: int, is_causal: bool, real_lengths: list
) -> None:
    """Splits, warps and query rows that see no key keep the output finite and exact.

    A cache far shorter than the static bound, or shorter than the causal queries,
    leaves them with no live score. Pool rows no request reads hold NaN.
    """
    _check_partly_filled_cache(seqlen_q, is_causal, real_lengths)


@pytest.mark.sm89
@pytest.mark.smoke
def test_mha_decode_paged_without_wgmma_drops_unread_rows() -> None:
    """SM89 has no TMA or WGMMA, so its paged decode stages the key tile the cache end cuts
    without them; the NaN rows past the cache end must not reach the output."""
    _check_partly_filled_cache(1, False, [700])


def _check_partly_filled_cache(seqlen_q: int, is_causal: bool, real_lengths: list) -> None:
    batch, heads, seqlen_kv, dim, page_size = len(real_lengths), 8, 1024, 64, 256
    test = MHADecodePagedTest(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, torch.float16
    )
    q, k, v, _full, block_table = test.gen_inputs()
    real_seqlen_kv = torch.tensor(real_lengths, dtype=torch.int32, device=q.device)
    pos = torch.arange(block_table.shape[1] * page_size, device=q.device)
    rows = block_table[:, pos // page_size] * page_size + pos % page_size
    read = torch.zeros(seqlen_kv, dtype=torch.bool, device=q.device)
    read[rows[pos < real_seqlen_kv[:, None]]] = True
    k[~read] = float("nan")
    v[~read] = float("nan")

    op = MHADecodePagedWithKVCacheFwdOp(page_size=page_size, is_causal=is_causal)
    output = op(q, k, v, real_seqlen_kv, block_table)

    assert torch.isfinite(output).all(), "output is not finite for a partly filled cache"
    compare_outputs(
        output, test.ref_program(q, k, v, real_seqlen_kv, block_table), test.verification(q)
    )


@pytest.mark.smoke
def test_mha_decode_paged_table_width_is_independent_of_pool() -> None:
    test = MHADecodePagedTest(2, 8, 1, 1024, 64, 256, False, torch.float16)
    q, k, v, lengths, table = test.gen_inputs()
    lengths.fill_(512)
    op = MHADecodePagedWithKVCacheFwdOp(page_size=256)
    for width in (2, 4):
        block_table = table[:, :width].contiguous()
        output = op(q, k, v, lengths, block_table)
        compare_outputs(
            output, test.ref_program(q, k, v, lengths, block_table), test.verification(q)
        )
