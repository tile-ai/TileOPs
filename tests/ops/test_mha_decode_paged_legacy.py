"""Legacy-style test for MultiHeadAttentionDecodePagedWithKVCacheOp (argparse + check + profile)."""

import pytest
import torch

from benchmarks.flash_decode import MultiHeadAttentionDecodePagedBenchmark
from top.ops import MultiHeadAttentionDecodePagedWithKVCacheOp


@pytest.mark.parametrize("batch,heads,seqlen_q,seqlen_kv,dim,page_size,is_causal,dtype", [
    (1, 16, 1, 512, 128, 128, False, torch.float16),
])
def test_mha_decode_paged(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: torch.dtype,
    tune: bool = False,
) -> None:
    op = MultiHeadAttentionDecodePagedWithKVCacheOp(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune=tune)
    benchmark = MultiHeadAttentionDecodePagedBenchmark(batch, heads, seqlen_q, seqlen_kv, dim,
                                                       page_size, is_causal, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs, atol=2e-3, rtol=1e-5)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
