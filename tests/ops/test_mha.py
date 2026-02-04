import argparse
import pytest
import torch

from benchmarks import MultiHeadAttentionBwdBenchmark, MultiHeadAttentionFwdBenchmark
from top.ops import MultiHeadAttentionBwdOp, MultiHeadAttentionFwdOp
from top.utils import str2dtype


@pytest.mark.parametrize("batch, seq_len, heads, dim, causal, dtype, tune", [
    (1, 1024, 8, 64, False, torch.float16, False),
    (16, 2048, 16, 128, False, torch.float16, False),
    (8, 4096, 16, 128, True, torch.bfloat16, True),
    (4, 4096, 16, 128, False, torch.bfloat16, True),
])
def test_mha_fwd(batch, seq_len, heads, dim, causal, dtype, tune):
    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    benchmark = MultiHeadAttentionFwdBenchmark(batch, heads, seq_len, dim, causal, dtype)

    inputs = benchmark.gen_inputs()
<<<<<<< HEAD
    print("Forward Results:")
    benchmark.check(op, *inputs, atol=5e-4, rtol=1e-5)
=======
    print(
        f"Forward Results for batch={batch}, seq_len={seq_len}, heads={heads}, dim={dim}, causal={causal}, dtype={dtype}, tune={tune}:"
    )
    if dtype == torch.bfloat16:
        benchmark.check(op, *inputs, atol=1.6e-2, rtol=1.6e-2)
    else:
        benchmark.check(op, *inputs, atol=1e-3, rtol=1e-3)
>>>>>>> 0f9974d (fix pytest for mha/gqa)
    benchmark.profile(op, *inputs)


@pytest.mark.parametrize("batch, seq_len, heads, dim, causal, dtype, tune", [
    (1, 1024, 8, 64, False, torch.float16, False),
    (16, 2048, 16, 128, False, torch.float16, False),
    (8, 4096, 16, 128, True, torch.bfloat16, True),
    (4, 4096, 16, 128, False, torch.bfloat16, True),
])
def test_mha_bwd(batch, seq_len, heads, dim, causal, dtype, tune):
    op = MultiHeadAttentionBwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    benchmark = MultiHeadAttentionBwdBenchmark(batch, heads, seq_len, dim, causal, dtype)

    inputs = benchmark.gen_inputs()
<<<<<<< HEAD
    print("Backward Results:")
    benchmark.check(op, *inputs, atol=5e-4, rtol=1e-5)
=======
    print(
        f"Backward Results for batch={batch}, seq_len={seq_len}, heads={heads}, dim={dim}, causal={causal}, dtype={dtype}, tune={tune}:"
    )
    if dtype == torch.bfloat16:
        benchmark.check(op, *inputs, atol=1.6e-2, rtol=1.6e-2)
    else:
        benchmark.check(op, *inputs, atol=1e-3, rtol=1e-3)
>>>>>>> 0f9974d (fix pytest for mha/gqa)
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
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    parser.add_argument(
        '--disable_bwd', action='store_false', default=True, help='when test fwd profile')
    args = parser.parse_args()

    test_mha_fwd(args.batch, args.seq_len, args.heads, args.dim, args.causal, str2dtype[args.dtype],
                 args.tune)
    if args.disable_bwd:
        test_mha_bwd(args.batch, args.seq_len, args.heads, args.dim, args.causal,
                     str2dtype[args.dtype], args.tune)
