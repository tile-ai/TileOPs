import argparse

from benchmarks import MultiHeadLatentAttentionDecodeBenchmark
from top.ops import MultiHeadLatentAttentionDecodeWithKVCacheOp
from top.utils import str2dtype


def test_mla_decode(batch, heads, head_num_kv, seq_len_kv, dim, dim_pe, dtype, tune=False):
    op = MultiHeadLatentAttentionDecodeWithKVCacheOp(
        batch, heads, head_num_kv, seq_len_kv, dim, dim_pe, dtype, tune=tune)
    benchmark = MultiHeadLatentAttentionDecodeBenchmark(batch, heads, head_num_kv, seq_len_kv, dim,
                                                        dim_pe, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--head_num_kv', type=int, default=1, help='number of key/value heads')
    parser.add_argument('--seq_len_kv', type=int, default=8192, help='key/value sequence length')
    parser.add_argument('--heads', type=int, default=128, help='num heads')
    parser.add_argument('--dim', type=int, default=512, help='head dim')
    parser.add_argument('--dim_pe', type=int, default=64, help='positional encoding dim')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_mla_decode(args.batch, args.heads, args.head_num_kv, args.seq_len_kv, args.dim,
                    args.dim_pe, str2dtype[args.dtype], args.tune)
