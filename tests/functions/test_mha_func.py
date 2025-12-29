import argparse
from top.functions import mha, MultiHeadAttentionFunc
from top.utils import str2dtype
from benchmarks import MultiHeadAttentionBenchmark


def test_mha_fn(B, S, H, D, causal, dtype):
    benchmark = MultiHeadAttentionBenchmark(B, H, S, D, causal, dtype)

    inputs = benchmark.gen_inputs()

    print("=========Testing mha function inference=========")
    benchmark.check_fn(mha, *inputs)

    print("=========Testing mha function class=========")
    fn = MultiHeadAttentionFunc(B, H, S, D, causal, dtype)
    benchmark.check_fn(fn, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--causal', action='store_true', default=False, help='causal attention')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    args = parser.parse_args()

    test_mha_fn(args.batch, args.seq_len, args.heads, args.dim, args.causal, str2dtype[args.dtype])
