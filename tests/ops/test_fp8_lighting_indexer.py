import argparse

from benchmarks import Fp8LightingIndexerBenchmark
from top.ops import Fp8LightingIndexerOp


def test_indexer(batch: int,
                 seq_len: int,
                 heads: int,
                 index_dim: int,
                 seq_len_kv: int,
                 kv_group: int,
                 clean_logits: bool,
                 tune: bool = False) -> None:
    op = Fp8LightingIndexerOp(
        batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits, tune=tune)
    benchmark = Fp8LightingIndexerBenchmark(batch, seq_len, heads, index_dim, seq_len_kv, kv_group,
                                            clean_logits)

    inputs = benchmark.gen_inputs()

    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--seq_len', type=int, default=512, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='number of heads')
    parser.add_argument('--index_dim', type=int, default=64, help='index dim')
    parser.add_argument('--seq_len_kv', type=int, default=1024, help='key/value sequence length')
    parser.add_argument('--kv_group', type=int, default=1, help='kv group')
    parser.add_argument(
        '--clean_logits',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='whether to clean logits outside the valid range')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_indexer(args.batch, args.seq_len, args.heads, args.index_dim, args.seq_len_kv,
                 args.kv_group, args.clean_logits, args.tune)
