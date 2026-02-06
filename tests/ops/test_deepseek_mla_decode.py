import sys

import pytest
import torch

from benchmarks import MultiHeadLatentAttentionDecodeBenchmark
from top.ops import MultiHeadLatentAttentionDecodeWithKVCacheOp


@pytest.mark.parametrize(
    "batch, heads, head_num_kv, seq_len_kv, dim, dim_pe, dtype, tune",
    [
        (32, 128, 1, 8192, 512, 64, torch.float16, False),
    ],
)
def test_mla_decode(batch: int, heads: int, head_num_kv: int, seq_len_kv: int, dim: int,
                    dim_pe: int, dtype: torch.dtype, tune: bool):
    op = MultiHeadLatentAttentionDecodeWithKVCacheOp(
        batch, heads, head_num_kv, seq_len_kv, dim, dim_pe, dtype, tune=tune)
    benchmark = MultiHeadLatentAttentionDecodeBenchmark(batch, heads, head_num_kv, seq_len_kv, dim,
                                                        dim_pe, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs, atol=3e-4, rtol=1e-5)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
