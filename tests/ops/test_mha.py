import sys
import pytest
import torch

from benchmarks import MultiHeadAttentionBwdBenchmark, MultiHeadAttentionFwdBenchmark
from top.ops import MultiHeadAttentionBwdOp, MultiHeadAttentionFwdOp


@pytest.mark.parametrize("batch, seq_len, heads, dim, causal, dtype, tune", [
    (1, 1024, 8, 64, False, torch.float16, False),
    (16, 2048, 16, 128, False, torch.float16, False),
    (4, 4096, 16, 128, False, torch.bfloat16, True),
])
def test_mha_fwd(batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype,
                 tune: bool) -> None:
    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    benchmark = MultiHeadAttentionFwdBenchmark(batch, heads, seq_len, dim, causal, dtype)

    inputs = benchmark.gen_inputs()
    print("Forward Results:")
    benchmark.check(op, *inputs, atol=5e-3, rtol=1e-5)
    benchmark.profile(op, *inputs)


@pytest.mark.parametrize("batch, seq_len, heads, dim, causal, dtype, tune", [
    (1, 1024, 8, 64, False, torch.float16, False),
    (16, 2048, 16, 128, False, torch.float16, False),
    (4, 4096, 16, 128, False, torch.bfloat16, True),
])
def test_mha_bwd(batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype,
                 tune: bool) -> None:
    op = MultiHeadAttentionBwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    benchmark = MultiHeadAttentionBwdBenchmark(batch, heads, seq_len, dim, causal, dtype)

    inputs = benchmark.gen_inputs()
    print("Backward Results:")
    benchmark.check(op, *inputs, atol=5e-3, rtol=1e-5)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
