"""Tests for multi-head attention backward and paged decode."""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MultiHeadAttentionBwdOp, MultiHeadAttentionDecodePagedWithKVCacheFwdOp
from workloads.device import run_device
from workloads.mha import (
    MhaBwdWorkload,
    MhaDecodePagedWorkload,
)


class MhaBwdTest(MhaBwdWorkload, TestBase):
    pass


class MhaBwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim, causal, dtype, tune",
            [
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-bwd-fp16",
                ),
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-bwd-bf16",
                ),
                pytest.param(
                    1,
                    256,
                    4,
                    128,
                    True,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-bwd-ws-causal",
                ),
                pytest.param(
                    16,
                    2048,
                    16,
                    128,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-bwd-fp16-large",
                ),
                pytest.param(
                    4,
                    4096,
                    16,
                    128,
                    False,
                    torch.bfloat16,
                    True,
                    marks=pytest.mark.full,
                    id="full-bwd-bf16-tuned",
                ),
            ],
        ),
    ]


@MhaBwdFixture
def test_mha_bwd(
    batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype, tune: bool
) -> None:
    test = MhaBwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionBwdOp(causal, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


class MhaDecodePagedTest(MhaDecodePagedWorkload, TestBase):
    #: bfloat16 carries 8 explicit mantissa bits against float16's 10, so one
    #: rounding of a unit-scale output is four times coarser.
    ATOL = {torch.float16: 0.001, torch.bfloat16: 0.005}

    def _maxdiff_cosine_compare(self, output: torch.Tensor, output_ref: torch.Tensor) -> None:
        """Compare using max-diff and cosine similarity."""
        atol = self.ATOL[self.dtype]
        if isinstance(output, (tuple, list)):
            output = output[0]
        max_diff = (output - output_ref).abs().max().item()
        assert max_diff < atol, f"max diff {max_diff} too large (atol={atol})"
        cos_sim = F.cosine_similarity(
            output.reshape(self.batch, -1), output_ref.reshape(self.batch, -1), dim=-1, eps=1e-8
        )
        assert cos_sim.min() > 0.99, f"cosine similarity {cos_sim.min().item()} too low"


class MhaDecodePagedFixture(FixtureBase):
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
                    marks=pytest.mark.smoke,
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


@MhaDecodePagedFixture
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
    test = MhaDecodePagedTest(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
    op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(
        page_size=page_size, is_causal=is_causal, tune=tune
    )
    test.check(op, *test.gen_inputs(), compare=test._maxdiff_cosine_compare)


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
    leaves them with no live score.
    """
    batch, heads, seqlen_kv, dim, page_size = len(real_lengths), 8, 1024, 64, 256
    test = MhaDecodePagedTest(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, torch.float16
    )
    q, k, v, _full, block_table = test.gen_inputs()
    real_seqlen_kv = torch.tensor(real_lengths, dtype=torch.int32, device=q.device)

    op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(page_size=page_size, is_causal=is_causal)
    output = op(q, k, v, real_seqlen_kv, block_table)

    assert torch.isfinite(output).all(), "output is not finite for a partly filled cache"
    test._maxdiff_cosine_compare(output, test.ref_program(q, k, v, real_seqlen_kv, block_table))


@pytest.mark.smoke
def test_mha_decode_paged_dispatch_declines_multi_token_query() -> None:
    """A query longer than one token belongs to the general kernel.

    The warp-specialized kernel exists because ``seqlen_q`` is 1; selection has
    to hand a longer query back rather than serve it.
    """
    op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(page_size=256, is_causal=False)
    q = torch.empty(1, 4, 8, 64, dtype=torch.float16, device=run_device())
    k = torch.empty(1024, 8, 64, dtype=torch.float16, device=run_device())
    chosen = op.select_kernel(op._attention_call(q, k))
    assert chosen.__name__ == "MHADecodePagedKernel"
