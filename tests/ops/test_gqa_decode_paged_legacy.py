"""Legacy-style test for GroupQueryAttentionDecodePagedWithKVCacheOp (argparse + check + profile)."""

import argparse

import torch

from benchmarks.flash_decode import GroupQueryAttentionDecodePagedBenchmark
from top.ops import GroupQueryAttentionDecodePagedWithKVCacheOp
from top.utils import str2dtype


def test_gqa_decode_paged(
    batch: int,
    heads: int,
    groups: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    dtype: torch.dtype,
    tune: bool = False,
) -> None:
    op = GroupQueryAttentionDecodePagedWithKVCacheOp(
        batch, heads, groups, seqlen_kv, dim, page_size, dtype, tune=tune)
    benchmark = GroupQueryAttentionDecodePagedBenchmark(batch, heads, groups, seqlen_kv, dim,
                                                        page_size, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=16, help="num heads")
    parser.add_argument("--groups", type=int, default=8, help="num kv groups")
    parser.add_argument("--seqlen_kv", type=int, default=512, help="key/value sequence length")
    parser.add_argument("--dim", type=int, default=128, help="head dim")
    parser.add_argument("--page_size", type=int, default=128, help="page size")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="data type",
    )
    parser.add_argument("--tune", action="store_true", default=False, help="enable autotune")
    args = parser.parse_args()

    dtype = str2dtype[args.dtype]
    test_gqa_decode_paged(
        args.batch,
        args.heads,
        args.groups,
        args.seqlen_kv,
        args.dim,
        args.page_size,
        dtype,
        args.tune,
    )
