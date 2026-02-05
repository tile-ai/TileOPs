import argparse

import torch

from benchmarks import GroupQueryAttentionDecodeBenchmark
from top.ops import GroupQueryAttentionDecodeWithKVCacheOp
from top.utils import str2dtype


def test_gqa_decode(b: int,
                    h: int,
                    g: int,
                    s_kv: int,
                    d: int,
                    dtype: torch.dtype,
                    tune: bool = False) -> None:
    op = GroupQueryAttentionDecodeWithKVCacheOp(b, h, g, s_kv, d, dtype, tune=tune)
    benchmark = GroupQueryAttentionDecodeBenchmark(b, h, g, s_kv, d, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs, atol=1e-3, rtol=1e-5)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--groups', type=int, default=8, help='number of groups')
    parser.add_argument('--seq_len_kv', type=int, default=8192, help='key/value sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_gqa_decode(args.batch, args.heads, args.groups, args.seq_len_kv, args.dim,
                    str2dtype['float16'], args.tune)
    test_gqa_decode(args.batch, args.heads, args.groups, args.seq_len_kv, args.dim,
                    str2dtype['bfloat16'], args.tune)
    test_gqa_decode(args.batch, args.heads, args.groups, 10, args.dim, str2dtype[args.dtype],
                    args.tune)
