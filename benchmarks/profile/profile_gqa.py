import argparse
from top.ops import GroupQueryAttentionFwdOp, GroupQueryAttentionBwdOp
from top.utils import str2dtype
from benchmarks import GroupQueryAttentionFwdBenchmark, GroupQueryAttentionBwdBenchmark


def test_gqa_fwd(B, S, H, H_kv, D, causal, dtype, tune=False):
    op = GroupQueryAttentionFwdOp(B, H, H_kv, S, D, causal, dtype, tune=tune)
    benchmark = GroupQueryAttentionFwdBenchmark(B, H, H_kv, S, D, causal, dtype)

    inputs = benchmark.gen_inputs()
    print("Forward Results:")
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)
    benchmark.baseline_profile(*inputs)


def test_gqa_bwd(B, S, H, H_kv, D, causal, dtype, tune=False):
    op = GroupQueryAttentionBwdOp(B, H, H_kv, S, D, causal, dtype, tune=tune)
    benchmark = GroupQueryAttentionBwdBenchmark(B, H, H_kv, S, D, causal, dtype)

    inputs = benchmark.gen_inputs()
    print("Backward Results:")
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
