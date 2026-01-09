import argparse

from benchmarks import GroupQueryAttentionDecodeBenchmark
from top.layers import GroupQueryAttentionDecodeLayer
from top.utils import str2dtype


def test_gqa_decode_layer(batch, heads, seq_len_kv, dim, groups, dtype):
    fn = GroupQueryAttentionDecodeLayer(batch, heads, groups, seq_len_kv, dim, dtype)
    benchmark = GroupQueryAttentionDecodeBenchmark(batch, heads, groups, seq_len_kv, dim, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check_fn(fn, *inputs, grad=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--groups', type=int, default=1, help='num groups')
    parser.add_argument('--seq_len_kv', type=int, default=8192, help='key/value sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_gqa_decode_layer(args.batch, args.heads, args.seq_len_kv, args.dim, args.groups,
                          str2dtype[args.dtype])
