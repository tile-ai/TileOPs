import argparse
from top.ops import gqa_fwd, gqa_bwd
from top.utils import str2dtype
from benchmarks import gqa_fwd_benchmark, gqa_bwd_benchmark


def test_gqa_fwd(B, S, H, H_KV, D, causal, dtype, tune=False):
    op = gqa_fwd(B, H, H_KV, S, D, causal, dtype, tune=tune)
    benchmark = gqa_fwd_benchmark(B, H, H_KV, S, D, causal, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)
    benchmark.baseline_profile(*inputs)


def test_gqa_bwd(B, S, H, H_KV, D, causal, dtype, tune=False):
    op = gqa_bwd(B, H, H_KV, S, D, causal, dtype, tune=tune)
    benchmark = gqa_bwd_benchmark(B, H, H_KV, S, D, causal, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)
    benchmark.baseline_profile(*inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--heads_kv', type=int, default=8, help='num heads kv')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--causal', action='store_true', default=False, help='causal attention')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    parser.add_argument(
        '--disable_bwd', action='store_false', default=True, help='when test fwd profile')
    args = parser.parse_args()

    test_gqa_fwd(args.batch, args.seq_len, args.heads, args.heads_kv, args.dim, args.causal,
                 str2dtype[args.dtype], args.tune)

    if args.disable_bwd:
        test_gqa_bwd(args.batch, args.seq_len, args.heads, args.heads_kv, args.dim, args.causal,
                     str2dtype[args.dtype], args.tune)
