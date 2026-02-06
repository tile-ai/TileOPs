import pytest
import torch

from benchmarks import MultiHeadAttentionDecodeBenchmark
from top.layers import MultiHeadAttentionDecodeLayer


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    "batch, seq_len_q, seq_len_kv, heads, dim, dtype",
    [
        (1, 128, 8192, 32, 128, torch.float16),
    ],
)
def test_mha_decode_layer(batch: int, seq_len_q: int, seq_len_kv: int, heads: int, dim: int,
                          dtype: torch.dtype):
    fn = MultiHeadAttentionDecodeLayer(batch, heads, seq_len_q, seq_len_kv, dim, dtype)
    benchmark = MultiHeadAttentionDecodeBenchmark(batch, heads, seq_len_q, seq_len_kv, dim, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check_fn(fn, *inputs, grad=False, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    import sys

    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
