import argparse
from top.layers import NativeSparseAttentionLayer
from benchmarks.deepseek_nsa.deepseek_nsa import NativeSparseAttentionForwardBenchmark


def test_nsa_op(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    scale=None,
    block_size=64,
    groups=1,
    selected_blocks=16,
    # dtype='float16',
    tune=False,
):
    layer = NativeSparseAttentionLayer(
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
    benchmark = NativeSparseAttentionForwardBenchmark(batch, heads, seq_len, dim, is_causal, scale,
                                                      block_size, groups, selected_blocks)

    inputs = benchmark.gen_inputs()
    benchmark.check(layer, *inputs)
    benchmark.profile(layer, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=2, help='batch size')
    parser.add_argument('--heads', type=int, default=16 * 4, help='number of heads')
    parser.add_argument('--seq_len', type=int, default=8192 * 3, help='sequence length')
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
