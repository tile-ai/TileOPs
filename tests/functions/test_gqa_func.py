import argparse

import torch

from benchmarks import GroupQueryAttentionBenchmark
from top.functions import GroupQueryAttentionFunc, gqa
from top.utils import str2dtype


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--heads_kv', type=int, default=8, help='num heads for key/value')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--causal', action='store_true', default=False, help='causal attention')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    args = parser.parse_args()

    test_gqa_fn(args.batch, args.seq_len, args.heads, args.heads_kv, args.dim, args.causal,
                str2dtype[args.dtype])
