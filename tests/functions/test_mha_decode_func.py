import argparse
from top.functions import mha_decode_with_kvcache
from top.utils import str2dtype
from benchmarks import MultiHeadAttentionDecodeBenchmark


def test_mha_decode_fn(B, S_q, S_kv, H, D, dtype):
    benchmark = MultiHeadAttentionDecodeBenchmark(B, H, S_q, S_kv, D, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check_fn(mha_decode_with_kvcache, *inputs, grad=False)


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

    test_mha_decode_fn(args.batch, args.seq_len_q, args.seq_len_kv, args.heads, args.dim,
                       str2dtype[args.dtype])
