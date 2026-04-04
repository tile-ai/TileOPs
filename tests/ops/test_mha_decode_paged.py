"""Test MultiHeadAttentionDecodePagedWithKVCacheOp (paged MHA decode with dynamic KV cache)."""


import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MultiHeadAttentionDecodePagedWithKVCacheOp
from workloads.ops.mha_decode_paged import MhaDecodePagedTest as _MhaDecodePagedTestWorkload


class MhaDecodePagedTest(_MhaDecodePagedTestWorkload, TestBase):

    def _maxdiff_cosine_compare(self, output: torch.Tensor, output_ref: torch.Tensor, atol: float = 0.001) -> None:
        """Compare using max-diff and cosine similarity."""
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
    op = MultiHeadAttentionDecodePagedWithKVCacheOp(
        batch=batch,
        heads=heads,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        dim=dim,
        page_size=page_size,
        is_causal=is_causal,
        dtype=dtype,
        tune=tune,
    )
    test.check(op, *test.gen_inputs(), compare=test._maxdiff_cosine_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
