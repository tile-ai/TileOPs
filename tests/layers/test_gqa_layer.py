import pytest
import torch

from benchmarks import GroupQueryAttentionBenchmark
from top.layers import GroupQueryAttentionLayer


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    "batch, seq_len, heads, heads_kv, dim, causal, dtype",
    [
        (8, 1024, 32, 32, 128, False, torch.float16),
    ],
)
def test_gqa_layer(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                   dtype: torch.dtype) -> None:

    gqa = GroupQueryAttentionLayer(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    benchmark = GroupQueryAttentionBenchmark(batch, heads, heads_kv, seq_len, dim, causal, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check_fn(gqa, *inputs, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    import sys

    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
