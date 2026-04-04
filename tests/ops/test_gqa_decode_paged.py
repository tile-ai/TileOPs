"""Test GroupQueryAttentionDecodePagedWithKVCacheOp (paged GQA decode with dynamic KV cache)."""


import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GroupQueryAttentionDecodePagedWithKVCacheOp
from workloads.ops.gqa_decode_paged import GqaDecodePagedTest as _GqaDecodePagedTestWorkload


class GqaDecodePagedTest(_GqaDecodePagedTestWorkload, TestBase):

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


class GqaDecodePagedFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune", [
            pytest.param(1, 16, 8, 512, 128, 128, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(2, 8, 4, 1024, 64, 256, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 32, 8, 256, 128, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 8, 4, 1024, 64, 256, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 16, 8, 512, 128, 128, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 16, 4, 2048, 128, 512, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 32, 16, 512, 64, 128, torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


@GqaDecodePagedFixture
def test_gqa_decode_paged_op(
    batch: int,
    heads: int,
    heads_kv: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GqaDecodePagedTest(batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype)
    op = GroupQueryAttentionDecodePagedWithKVCacheOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seqlen_kv=seqlen_kv,
        dim=dim,
        page_size=page_size,
        dtype=dtype,
        tune=tune,
    )
    test.check(op, *test.gen_inputs(), compare=test._maxdiff_cosine_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
