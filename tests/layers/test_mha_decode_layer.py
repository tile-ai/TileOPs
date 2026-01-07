import argparse

from benchmarks import MultiHeadAttentionDecodeBenchmark
from top.layers import MultiHeadAttentionDecodeLayer
from top.utils import str2dtype


def test_mha_decode_layer(batch, seq_len_q, seq_len_kv, heads, dim, dtype):
    fn = MultiHeadAttentionDecodeLayer(batch, heads, seq_len_q, seq_len_kv, dim, dtype)
    benchmark = MultiHeadAttentionDecodeBenchmark(batch, heads, seq_len_q, seq_len_kv, dim, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check_fn(fn, *inputs, grad=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--seq_len_q', type=int, default=128, help='query sequence length')
    parser.add_argument('--seq_len_kv', type=int, default=8192, help='key/value sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_mha_decode_layer(args.batch, args.seq_len_q, args.seq_len_kv, args.heads, args.dim,
                          str2dtype[args.dtype])
