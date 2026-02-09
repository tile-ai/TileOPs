import sys
import pytest
import torch

from benchmarks import GroupQueryAttentionDecodeBenchmark
from top.layers import GroupQueryAttentionDecodeLayer


@pytest.mark.parametrize(
    "batch, heads, seq_len_kv, dim, groups, dtype",
    [
        (1, 32, 8192, 128, 1, torch.float16),
    ],
)
def test_gqa_decode_layer(batch: int, heads: int, seq_len_kv: int, dim: int, groups: int,
                          dtype: torch.dtype):
    fn = GroupQueryAttentionDecodeLayer(batch, heads, groups, seq_len_kv, dim, dtype)
    benchmark = GroupQueryAttentionDecodeBenchmark(batch, heads, groups, seq_len_kv, dim, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check_fn(fn, *inputs, grad=False, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
