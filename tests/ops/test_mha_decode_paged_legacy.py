"""Legacy-style test for MultiHeadAttentionDecodePagedWithKVCacheOp (argparse + check + profile)."""

import argparse

import torch

from benchmarks.flash_decode import MultiHeadAttentionDecodePagedBenchmark
from top.ops import MultiHeadAttentionDecodePagedWithKVCacheOp
from top.utils import str2dtype


def test_mha_decode_paged(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: torch.dtype,
    tune: bool = False,
) -> None:
    op = MultiHeadAttentionDecodePagedWithKVCacheOp(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune=tune)
    benchmark = MultiHeadAttentionDecodePagedBenchmark(batch, heads, seqlen_q, seqlen_kv, dim,
                                                       page_size, is_causal, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=16, help="num heads")
    parser.add_argument("--seqlen_q", type=int, default=1, help="query sequence length")
    parser.add_argument("--seqlen_kv", type=int, default=512, help="key/value sequence length")
    parser.add_argument("--dim", type=int, default=128, help="head dim")
    parser.add_argument("--page_size", type=int, default=128, help="page size")
    parser.add_argument("--is_causal", action="store_true", default=False, help="causal mask")
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
    test_mha_decode_paged(
        args.batch,
        args.heads,
        args.seqlen_q,
        args.seqlen_kv,
        args.dim,
        args.page_size,
        args.is_causal,
        dtype,
        args.tune,
    )
