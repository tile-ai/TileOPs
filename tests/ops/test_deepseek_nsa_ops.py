"""Test NativeSparseAttention operation."""
import argparse
import pytest
import torch

from top.ops import NativeSparseAttentionForwardOp
from benchmarks.deepseek_nsa.deepseek_nsa import NativeSparseAttentionForwardBenchmark


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    "batch, heads, seq_len, dim, is_causal, scale, block_size, groups, selected_blocks, tune",
    [
        # default configuration
        (1, 64, 8192, 128, True, 0.1, 32, 16, 16, True),
        (1, 64, 8192*2, 128, True, 0.1, 32, 16, 16, True),
        (1, 64, 8192*4, 128, True, 0.1, 32, 16, 16, True),
        (1, 64, 8192*8, 128, True, 0.1, 32, 16, 16, True),
        (16, 64, 8192, 128, True, 0.1, 32, 16, 16, True),
        # (16, 64, 8192*2, 128, True, 0.1, 32, 16, 16, True),
        # (16, 64, 8192*4, 128, True, 0.1, 32, 16, 16, True),
        # (16, 64, 8192*8, 128, True, 0.1, 32, 16, 16, True),
        # small batch size configuration
        (1, 16, 1024, 128, True, 0.1, 32, 16, 16, True),
        (4, 32, 2048, 128, True, 0.1, 32, 16, 16, True),
        # different sequence length
        (8, 32, 4096, 128, True, 0.1, 32, 16, 16, True),
        (8, 32, 16384, 128, True, 0.1, 32, 16, 16, True),
        # different block_size
        (8, 32, 4096, 128, True, 0.1, 64, 16, 16, True),
        (8, 32, 4096, 128, True, 0.1, 16, 16, 16, True),
        # different groups
        (8, 32, 4096, 128, True, 0.1, 32, 32, 16, True),
        # different selected_blocks
        (8, 32, 4096, 128, True, 0.1, 32, 16, 8, True),
        (8, 32, 4096, 128, True, 0.1, 32, 16, 32, True),
        # different scale
        (8, 32, 4096, 128, True, 0.05, 32, 16, 16, True),
        (8, 32, 4096, 128, True, 0.2, 32, 16, 16, True),
    ],
)
def test_nsa_op(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    scale,
    block_size,
    groups,
    selected_blocks,
    tune,
):
    """Test NativeSparseAttention forward operation with various configurations."""
    op = NativeSparseAttentionForwardOp(
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        scale,
        block_size,
        groups,
        selected_blocks,
        tune=tune)
    benchmark = NativeSparseAttentionForwardBenchmark(
        batch, heads, seq_len, dim, is_causal, scale,
        block_size, groups, selected_blocks)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    # benchmark.profile(op, *inputs)
    # benchmark.baseline_profile(*inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--heads', type=int, default=16 * 4, help='number of heads')
    parser.add_argument('--seq_len', type=int, default=8192 * 1, help='sequence length')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument(
        '--is_causal', action='store_true', default=True, help='enable causal attention')
    parser.add_argument('--scale', type=float, default=0.1, help='scale')
    parser.add_argument('--block_size', type=int, default=32, help='block size')
    parser.add_argument('--groups', type=int, default=16, help='number of groups')
    parser.add_argument('--selected_blocks', type=int, default=16, help='number of selected blocks')
    parser.add_argument('--tune', action='store_true', default=True, help='enable autotune')
    args = parser.parse_args()

    test_nsa_op(
        args.batch,
        args.heads,
        args.seq_len,
        args.dim,
        args.is_causal,
        args.scale,
        args.block_size,
        args.groups,
        args.selected_blocks,
        args.tune,
    )