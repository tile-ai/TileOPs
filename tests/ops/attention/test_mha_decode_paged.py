"""Test MultiHeadAttentionDecodePagedWithKVCacheFwdOp (paged MHA decode with dynamic KV cache)."""



import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MultiHeadAttentionDecodePagedWithKVCacheFwdOp
from tileops.ops.attention.selection import MHA_PAGED_DECODE_KEYS
from workloads.attention.mha import (
    MhaDecodePagedWorkload,
)


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
        assert max_diff < atol, (
            f"max diff {max_diff} too large (atol={atol})")
        cos_sim = F.cosine_similarity(
            output.reshape(self.batch, -1), output_ref.reshape(self.batch, -1), dim=-1, eps=1e-8)
        assert cos_sim.min() > 0.99, f"cosine similarity {cos_sim.min().item()} too low"


class MhaDecodePagedFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune", [
            pytest.param(
                1, 16, 1, 512, 128, 128, False, torch.float16, False,
                marks=pytest.mark.smoke,
            ),
            # bfloat16 dispatch: the same signature admits it and the paged
            # decode kernels are selected on dtype.
            pytest.param(
                1, 16, 1, 1024, 128, 128, False, torch.bfloat16, False,
                marks=pytest.mark.smoke,
            ),
            pytest.param(
                1, 8, 1, 1024, 64, 256, False, torch.float16, False,
                marks=pytest.mark.full,
            ),
            pytest.param(
                2, 8, 1, 1024, 64, 256, False, torch.float16, False,
                marks=pytest.mark.full,
            ),
            pytest.param(
                1, 8, 1, 512, 64, 256, False, torch.float16, False,
                marks=pytest.mark.full,
            ),
        ]),
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
        batch=batch,
        heads=heads,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        dim=dim,
        page_size=page_size,
        is_causal=is_causal,
        tune=tune,
    )
    test.check(op, *test.gen_inputs(), compare=test._maxdiff_cosine_compare)


@pytest.mark.parametrize("real_lengths", [
    pytest.param([1], marks=pytest.mark.smoke),
    pytest.param([37], marks=pytest.mark.full),
    pytest.param([700, 1024, 1], marks=pytest.mark.full),
])
def test_mha_decode_paged_cache_shorter_than_bound(real_lengths: list) -> None:
    """A cache far shorter than the static bound leaves splits with no rows.

    Regression: a split past the end of the cache, and a consumer warp whose
    rows are all masked, reach the epilogue having seen no live score. With the
    running max initialised to -inf that epilogue evaluates exp2(-inf - -inf),
    and the NaN propagates through the cross-split merge into every element of
    the output.
    """
    batch, heads, seqlen_kv, dim, page_size = len(real_lengths), 8, 1024, 64, 256
    test = MhaDecodePagedTest(batch, heads, 1, seqlen_kv, dim, page_size, False, torch.float16)
    q, k, v, _full, block_table = test.gen_inputs()
    real_seqlen_kv = torch.tensor(real_lengths, dtype=torch.int32, device=q.device)

    op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(
        batch=batch, heads=heads, seqlen_q=1, seqlen_kv=seqlen_kv, dim=dim,
        page_size=page_size, is_causal=False)
    output = op(q, k, v, real_seqlen_kv, block_table)

    assert torch.isfinite(output).all(), "output is not finite for a partly filled cache"
    test._maxdiff_cosine_compare(
        output, test.ref_program(q, k, v, real_seqlen_kv, block_table))


@pytest.mark.smoke
def test_mha_decode_paged_dispatch_declines_multi_token_query() -> None:
    """A query longer than one token belongs to the general kernel.

    The warp-specialized kernel exists because ``seqlen_q`` is 1; selection has
    to hand a longer query back rather than serve it.
    """
    op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(
        batch=1, heads=8, seqlen_q=4, seqlen_kv=1024, dim=64, page_size=256,
        is_causal=False)
    key = op.select_kernel_key(MHA_PAGED_DECODE_KEYS, op._attention_call(torch.float16))
    assert key == "mha_decode_paged_kernel"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
