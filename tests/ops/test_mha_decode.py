import argparse
from top.ops import MultiHeadAttentionDecodeOp
from top.utils import str2dtype
from benchmarks import MultiHeadAttentionDecodeBenchmark as mha_decode_benchmark


def test_mha_decode(B, H, S_q, S_kv, D, dtype, tune=False):
    op = MultiHeadAttentionDecodeOp(B, H, S_q, S_kv, D, dtype, tune=tune)
    benchmark = mha_decode_benchmark(B, H, S_q, S_kv, D, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


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

    test_mha_decode(args.batch, args.heads, args.seq_len_q, args.seq_len_kv, args.dim,
                    str2dtype[args.dtype], args.tune)
