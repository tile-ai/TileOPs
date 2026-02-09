import sys
import pytest
import torch

from benchmarks import GroupQueryAttentionBwdBenchmark, GroupQueryAttentionFwdBenchmark
from top.ops import GroupQueryAttentionBwdOp, GroupQueryAttentionFwdOp


@pytest.mark.parametrize("batch, seq_len, heads, heads_kv, dim, causal, dtype, tune", [
    (1, 1024, 8, 4, 64, False, torch.float16, False),
    (4, 2048, 64, 4, 128, False, torch.float16, False),
    (4, 2048, 64, 4, 128, False, torch.bfloat16, False),
])
def test_gqa_fwd(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                 dtype: torch.dtype, tune: bool) -> None:
    op = GroupQueryAttentionFwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    benchmark = GroupQueryAttentionFwdBenchmark(batch, heads, heads_kv, seq_len, dim, causal, dtype)

    inputs = benchmark.gen_inputs()
    print("Forward Results:")
    benchmark.check(op, *inputs, atol=5e-3, rtol=1e-5)
    benchmark.profile(op, *inputs)


@pytest.mark.parametrize("batch, seq_len, heads, heads_kv, dim, causal, dtype, tune", [
    (1, 1024, 8, 4, 64, False, torch.float16, False),
    (4, 2048, 64, 4, 128, False, torch.float16, False),
    (4, 2048, 64, 4, 128, False, torch.bfloat16, False),
])
def test_gqa_bwd(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                 dtype: torch.dtype, tune: bool) -> None:
    op = GroupQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    benchmark = GroupQueryAttentionBwdBenchmark(batch, heads, heads_kv, seq_len, dim, causal, dtype)

    inputs = benchmark.gen_inputs()
    print("Backward Results:")
    benchmark.check(op, *inputs, atol=5e-3, rtol=1e-5)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
