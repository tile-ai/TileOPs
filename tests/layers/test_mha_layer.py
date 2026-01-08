import argparse

from benchmarks import MultiHeadAttentionBenchmark
from top.layers import MultiHeadAttentionLayer
from top.utils import str2dtype


def test_mha_layer(batch, seq_len, heads, dim, causal, dtype):

    mha = MultiHeadAttentionLayer(batch, heads, seq_len, dim, causal, dtype)
    benchmark = MultiHeadAttentionBenchmark(batch, heads, seq_len, dim, causal, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check_fn(mha, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--causal', action='store_true', default=False, help='causal attention')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'bfloat16'],
                        help='data type')
    args = parser.parse_args()

    test_mha_layer(args.batch, args.seq_len, args.heads, args.dim, args.causal,
                   str2dtype[args.dtype])
