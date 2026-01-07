import argparse

from benchmarks import GroupQueryAttentionBenchmark
from top.layers import GroupQueryAttentionLayer
from top.utils import str2dtype


def test_gqa_layer(batch, seq_len, heads, heads_kv, dim, causal, dtype):

    gqa = GroupQueryAttentionLayer(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    benchmark = GroupQueryAttentionBenchmark(batch, heads, heads_kv, seq_len, dim, causal, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check_fn(gqa, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--heads_kv', type=int, default=32, help='num heads for key/value')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--causal', action='store_true', default=False, help='causal attention')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    args = parser.parse_args()

    test_gqa_layer(args.batch, args.seq_len, args.heads, args.heads_kv, args.dim, args.causal,
                   str2dtype[args.dtype])
