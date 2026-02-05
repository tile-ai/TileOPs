import argparse

import pytest
import torch

from benchmarks import GroupQueryAttentionBwdBenchmark, GroupQueryAttentionFwdBenchmark
from top.ops import GroupQueryAttentionBwdOp, GroupQueryAttentionFwdOp
from top.utils import str2dtype


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(123)


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
    benchmark.check(op, *inputs, atol=5e-4, rtol=1e-5)
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
    benchmark.check(op, *inputs, atol=5e-4, rtol=1e-5)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--heads_kv', type=int, default=8, help='num heads kv')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--causal', action='store_true', default=False, help='causal attention')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    parser.add_argument(
        '--disable_bwd', action='store_false', default=True, help='when test fwd profile')
    args = parser.parse_args()

    test_gqa_fwd(args.batch, args.seq_len, args.heads, args.heads_kv, args.dim, args.causal,
                 str2dtype[args.dtype], args.tune)

    if args.disable_bwd:
        test_gqa_bwd(args.batch, args.seq_len, args.heads, args.heads_kv, args.dim, args.causal,
                     str2dtype[args.dtype], args.tune)
