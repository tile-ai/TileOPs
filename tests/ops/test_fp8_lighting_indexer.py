import argparse
from typing import Optional

from benchmarks import Fp8LightingIndexerBenchmark
from top.ops import Fp8LightingIndexerOp


def test_indexer(seq_len: int,
                 heads: int,
                 index_dim: int,
                 seq_len_kv: int,
                 clean_logits: bool,
                 config: Optional[dict],
                 tune: bool = False) -> None:
    op = Fp8LightingIndexerOp(
        seq_len, heads, index_dim, seq_len_kv, clean_logits, config, tune=tune)
    benchmark = Fp8LightingIndexerBenchmark(seq_len, heads, index_dim, seq_len_kv, clean_logits,
                                            config)

    inputs = benchmark.gen_inputs()

    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=4096, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='number of heads')
    parser.add_argument('--index_dim', type=int, default=64, help='index dim')
    parser.add_argument('--seq_len_kv', type=int, default=8192, help='key/value sequence length')
    parser.add_argument(
        '--clean_logits',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='whether to clean logits outside the valid range')
    parser.add_argument('--config', type=str, default=None, help='positional encoding dim')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_indexer(args.seq_len, args.heads, args.index_dim, args.seq_len_kv, args.clean_logits,
                 args.config, args.tune)
