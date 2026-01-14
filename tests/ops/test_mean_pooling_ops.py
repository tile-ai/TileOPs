import argparse

import pytest
import torch

from benchmarks.deepseek_nsa.deepseek_nsa import MeanPoolingForwardBenchmark
from top.ops import MeanPoolingForwardOp


@pytest.mark.parametrize(
    "batch_size, seq_len, heads, dim, chunk_size, chunks_per_bacth, seq_num, \
        use_offsets, dtype, accum_dtype, tune",
    [
        (1, 8192, 64, 128, 64, 128, 1, 0, torch.float16, torch.float32, False),
        (1, 8192, 64, 128, 64, 128, 1, 0, torch.float16, torch.float32, True),
        (1, 8192, 64, 128, 64, 128, 1, 1, torch.float16, torch.float32, False),
        (1, 8192, 64, 128, 64, 128, 1, 1, torch.float16, torch.float32, True),
    ],
)
def test_mean_pooling_op(
    batch_size: int,
    seq_len: int,
    heads: int,
    dim: int,
    chunk_size: int,
    chunks_per_bacth: int,
    seq_num: int,
    use_offsets: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
) -> None:
    op = MeanPoolingForwardOp(
        batch_size=batch_size,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
        chunk_size=chunk_size,
        chunks_per_bacth=chunks_per_bacth,
        seq_num=seq_num,
        use_offsets=use_offsets,
        dtype=dtype,
        accum_dtype=accum_dtype,
        tune=tune)

    benchmark = MeanPoolingForwardBenchmark(
        batch_size=batch_size,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
        chunk_size=chunk_size,
        chunks_per_bacth=chunks_per_bacth,
        seq_num=seq_num,
        use_offsets=use_offsets,
        dtype=dtype,
        accum_dtype=accum_dtype,
        tune=tune)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)
    benchmark.baseline_profile(*inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='logical batch size')
    parser.add_argument('--seq_len', type=int, default=8192, help='sequence length')
    parser.add_argument('--heads', type=int, default=64, help='num heads')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--chunk_size', type=int, default=64, help='chunk size')
    parser.add_argument('--chunks_per_bacth', type=int, default=128, help='chunks per batch')
    parser.add_argument('--seq_num', type=int, default=1, help='number of sequences')
    parser.add_argument(
        '--use_offsets', type=int, default=0, help='not use offsets: 0, use offsets: 1')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument(
        '--accum_dtype',
        type=str,
        default='float16',
        choices=['float32', 'bfloat32'],
        help='accumulation data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()
    test_mean_pooling_op(
        args.batch_size,
        args.seq_len,
        args.heads,
        args.dim,
        args.chunk_size,
        args.chunks_per_bacth,
        args.seq_num,
        args.use_offsets,
        torch.float16 if args.dtype == 'float16' else torch.bfloat16,
        torch.float32 if args.accum_dtype == 'float16' else torch.bfloat32,
        args.tune,
    )
