# This test validates the compatibility of TileOps operators with torch.compile().
# Check: https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html

import argparse
import pytest
import torch

from benchmarks import MultiHeadAttentionFwdBenchmark
from top.ops import MultiHeadAttentionFwdOp
from top.utils import str2dtype


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    "B, S, H, D, causal, dtype",
    [
        (8, 1024, 32, 128, False, torch.float16),
        (4, 512, 16, 64, True, torch.bfloat16),
        (2, 2048, 64, 128, False, torch.float16),
    ],
)
def test_mha_kernel_compile(B: int, S: int, H: int, D: int, causal: bool, dtype: torch.dtype):
    op = MultiHeadAttentionFwdOp(B, H, S, D, causal, dtype)
    benchmark = MultiHeadAttentionFwdBenchmark(B, H, S, D, causal, dtype)

    compiled_op = torch.compile(op, fullgraph=True)
    inputs = benchmark.gen_inputs()
    benchmark.check(
        compiled_op, *inputs, atol=3e-4, rtol=1e-5)  # will throw an error if not compatible
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

    # Convert string dtype to torch.dtype
    dtype = str2dtype[args.dtype]

    # Run the test with command line arguments
    test_mha_kernel_compile(args.batch, args.seq_len, args.heads, args.dim, args.causal, dtype)
