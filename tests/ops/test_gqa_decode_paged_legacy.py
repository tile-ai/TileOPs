"""Legacy-style test for GroupQueryAttentionDecodePagedWithKVCacheOp (argparse + check + profile)."""

import sys

import pytest
import torch

from benchmarks.flash_decode import GroupQueryAttentionDecodePagedBenchmark
from top.ops import GroupQueryAttentionDecodePagedWithKVCacheOp


@pytest.mark.parametrize("batch,heads,groups,seqlen_kv,dim,page_size,dtype", [
    (1, 16, 8, 512, 128, 128, torch.float16),
])
def test_gqa_decode_paged(
    batch: int,
    heads: int,
    groups: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    dtype: torch.dtype,
    tune: bool = False,
) -> None:
    torch.manual_seed(123)  # 替代 fixture 中的随机种子设置
    op = GroupQueryAttentionDecodePagedWithKVCacheOp(
        batch, heads, groups, seqlen_kv, dim, page_size, dtype, tune=tune)
    benchmark = GroupQueryAttentionDecodePagedBenchmark(batch, heads, groups, seqlen_kv, dim,
                                                        page_size, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs, atol=1e-2, rtol=1e-2)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
