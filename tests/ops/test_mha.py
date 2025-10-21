import argparse
from top import mha_fwd, mha_fwd_benchmark, mha_bwd, mha_bwd_benchmark
from top.utils import str2dtype


def test_mha_fwd(B, S, H, D, causal, dtype):
    op = mha_fwd(B, H, S, D, causal, dtype)
    benchmark = mha_fwd_benchmark(B, H, S, D, causal, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


def test_mha_bwd(B, S, H, D, causal, dtype):
    op = mha_bwd(B, H, S, D, causal, dtype)
    benchmark = mha_bwd_benchmark(B, H, S, D, causal, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


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

    test_mha_fwd(args.batch, args.seq_len, args.heads, args.dim, args.causal, str2dtype[args.dtype])
    test_mha_bwd(args.batch, args.seq_len, args.heads, args.dim, args.causal, str2dtype[args.dtype])