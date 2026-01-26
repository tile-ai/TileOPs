import argparse

import torch

from benchmarks import TopkSelectorBenchmark
from top.ops import TopkSelectorOp
from top.utils import str2dtype


def test_topk_selector(batch: int,
                       seq_len: int,
                       topk: int,
                       in_dtype: torch.dtype,
                       out_dtype: torch.dtype,
                       tune: bool = False) -> None:
    op = TopkSelectorOp(batch, seq_len, topk, in_dtype, out_dtype, tune=tune)
    benchmark = TopkSelectorBenchmark(
        batch,
        seq_len,
        topk,
        in_dtype,
        out_dtype,
    )

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--seq_len', type=int, default=32 * 1024, help='sequence length')
    parser.add_argument('--topk', type=int, default=2048, help='topk')
    parser.add_argument('--in_dtype', type=str, default="float32", help='input type')
    parser.add_argument('--out_dtype', type=str, default="int32", help='output type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_topk_selector(args.batch, args.seq_len, args.topk, str2dtype[args.in_dtype],
                       str2dtype[args.out_dtype], args.tune)
