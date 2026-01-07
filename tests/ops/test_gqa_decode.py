import argparse

from benchmarks import GroupQueryAttentionDecodeBenchmark
from top.ops import GroupQueryAttentionDecodeWithKVCacheOp
from top.utils import str2dtype


def test_gqa_decode(batch, heads, groups, seq_len_kv, dim, dtype, tune=False):
    op = GroupQueryAttentionDecodeWithKVCacheOp(
        batch, heads, groups, seq_len_kv, dim, dtype, tune=tune)
    benchmark = GroupQueryAttentionDecodeBenchmark(batch, heads, groups, seq_len_kv, dim, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
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
                    str2dtype[args.dtype], args.tune)
