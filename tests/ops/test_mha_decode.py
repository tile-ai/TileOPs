import argparse
from top import mha_decode
from top.utils import str2dtype
from benchmarks import mha_decode_benchmark


def test_mha_decode(B, H, S_q, S_kv, D, causal, dtype):
    op = mha_decode(B, H, S_q, S_kv, D, causal, dtype)
    benchmark = mha_decode_benchmark(B, H, S_q, S_kv, D, causal, dtype)

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
    parser.add_argument('--causal', action='store_true', default=False, help='causal attention')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    args = parser.parse_args()

    test_mha_decode(args.batch, args.heads, args.seq_len_q, args.seq_len_kv, args.dim, args.causal, str2dtype[args.dtype])
    