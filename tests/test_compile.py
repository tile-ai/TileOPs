# This test validates the compatibility of TileOps operators with torch.compile().
# Check: https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html

import argparse
from top.ops import MultiHeadAttentionFwdOp
from top.utils import str2dtype
from benchmarks import MultiHeadAttentionFwdBenchmark as mha_fwd_benchmark
import torch


def test_mha_kernel_compile(B, S, H, D, causal, dtype):
    op = MultiHeadAttentionFwdOp(B, H, S, D, causal, dtype)
    benchmark = mha_fwd_benchmark(B, H, S, D, causal, dtype)

    compiled_op = torch.compile(op, fullgraph=True)
    inputs = benchmark.gen_inputs()
    benchmark.check(compiled_op, *inputs)  # will throw an error if not compatible
    benchmark.profile(compiled_op, *inputs)

    print('Successfully validate the compatibility with torch.compile().✅')


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

    test_mha_kernel_compile(args.batch, args.seq_len, args.heads, args.dim, args.causal,
                            str2dtype[args.dtype])
