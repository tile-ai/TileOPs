import pytest
import torch

from benchmarks import MultiHeadAttentionBenchmark
from top.layers import MultiHeadAttentionLayer


@pytest.mark.parametrize(
    "batch, seq_len, heads, dim, causal, dtype",
    [
        (8, 1024, 32, 128, False, torch.float16),
    ],
)
def test_mha_layer(batch: int, seq_len: int, heads: int, dim: int, causal: bool,
                   dtype: torch.dtype) -> None:

    mha = MultiHeadAttentionLayer(batch, heads, seq_len, dim, causal, dtype)
    benchmark = MultiHeadAttentionBenchmark(batch, heads, seq_len, dim, causal, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check_fn(mha, *inputs, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    import sys

    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
