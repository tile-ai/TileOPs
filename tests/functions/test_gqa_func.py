import pytest
import torch

from benchmarks import GroupQueryAttentionBenchmark
from top.functions import GroupQueryAttentionFunc, gqa


@pytest.mark.parametrize(
    "batch, seq_len, heads, heads_kv, dim, causal, dtype",
    [
        (8, 1024, 32, 8, 128, False, torch.float16),
    ],
)
def test_gqa_fn(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                dtype: torch.dtype) -> None:
    benchmark = GroupQueryAttentionBenchmark(batch, heads, heads_kv, seq_len, dim, causal, dtype)

    inputs = benchmark.gen_inputs()

    print("=========Testing gqa function inference=========")
    benchmark.check_fn(gqa, *inputs, atol=3e-4, rtol=1e-5)

    print("=========Testing gqa function class=========")
    fn = GroupQueryAttentionFunc(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    benchmark.check_fn(fn, *inputs, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    import sys

    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
