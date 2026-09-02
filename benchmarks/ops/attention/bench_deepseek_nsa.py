"""Benchmarks for the three Native Sparse Attention (NSA) varlen passes.

Each pass is compared against the torch reference in ``workloads.attention.deepseek``;
the top-k pass returns block ids, so it is timed but not compared for closeness.
"""

from typing import List

import pytest
import torch

from benchmarks.benchmark_base import ManifestBenchmark, fields, workload_params
from tileops.attention import NSACmpFwdVarlenOp, NSAFwdVarlenOp, NSATopkVarlenOp
from tileops.manifest import load_workloads
from workloads.attention.deepseek import (
    NsaCmpFwdWorkload,
    NsaFwdWorkload,
    NsaTopkWorkload,
)

# One config wide, so there is nothing for a search to find.
_TUNE = False


@pytest.mark.parametrize(
    "seq_num, c_seq_len, heads, head_kv, dim_k, dim_v, scale, bc, bs, accum_dtype, seq_lens, dtype",
    workload_params(
        load_workloads(NSACmpFwdVarlenOp),
        fields(
            "seq_num",
            "c_seq_len",
            "heads",
            "head_kv",
            "dim_k",
            "dim_v",
            "scale",
            "bc",
            "bs",
            "accum_dtype",
            "seq_lens",
            dtype_last=True,
        ),
    ),
)
def test_nsa_cmp_fwd_varlen_bench(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    head_kv: int,
    dim_k: int,
    dim_v: int,
    scale: float,
    bc: int,
    bs: int,
    accum_dtype: torch.dtype,
    seq_lens: List[int],
    dtype: torch.dtype,
) -> None:
    test = NsaCmpFwdWorkload(
        seq_num,
        c_seq_len,
        heads,
        dim_k,
        dim_v,
        heads // head_kv,
        scale,
        bc,
        bs,
        dtype,
        accum_dtype,
        seq_lens=seq_lens,
    )
    inputs = test.gen_inputs()
    op = NSACmpFwdVarlenOp(scale=scale, bc=bc, bs=bs, accum_dtype=accum_dtype, tune=_TUNE)

    bm = ManifestBenchmark(op, test)
    bm.compare({"tileops": op, "torch-ref": test.ref_program}, *inputs)


@pytest.mark.parametrize(
    "seq_num, c_seq_len, heads, head_kv, dim, selected_block_num, scale, bc, bs, "
    "accum_dtype, seq_lens, dtype",
    workload_params(
        load_workloads(NSATopkVarlenOp),
        fields(
            "seq_num",
            "c_seq_len",
            "heads",
            "head_kv",
            "dim",
            "selected_block_num",
            "scale",
            "bc",
            "bs",
            "accum_dtype",
            "seq_lens",
            dtype_last=True,
        ),
    ),
)
def test_nsa_topk_varlen_bench(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    head_kv: int,
    dim: int,
    selected_block_num: int,
    scale: float,
    bc: int,
    bs: int,
    accum_dtype: torch.dtype,
    seq_lens: List[int],
    dtype: torch.dtype,
) -> None:
    test = NsaTopkWorkload(
        seq_num,
        c_seq_len,
        heads,
        dim,
        heads // head_kv,
        scale,
        selected_block_num,
        bc,
        bs,
        dtype,
        accum_dtype,
        seq_lens=seq_lens,
    )
    inputs = test.gen_inputs()
    op = NSATopkVarlenOp(
        scale=scale,
        selected_block_num=selected_block_num,
        bc=bc,
        bs=bs,
        accum_dtype=accum_dtype,
        tune=_TUNE,
    )

    bm = ManifestBenchmark(op, test)
    bm.compare({"tileops": op, "torch-ref": test.ref_program}, *inputs)


@pytest.mark.parametrize(
    "batch, c_seq_len, heads, head_kv, dim, selected_blocks, is_causal, scale, block_size, "
    "accum_dtype, seq_lens, dtype",
    workload_params(
        load_workloads(NSAFwdVarlenOp),
        fields(
            "batch",
            "c_seq_len",
            "heads",
            "head_kv",
            "dim",
            "selected_blocks",
            "is_causal",
            "scale",
            "block_size",
            "accum_dtype",
            "seq_lens",
            dtype_last=True,
        ),
    ),
)
def test_nsa_fwd_varlen_bench(
    batch: int,
    c_seq_len: int,
    heads: int,
    head_kv: int,
    dim: int,
    selected_blocks: int,
    is_causal: bool,
    scale: float,
    block_size: int,
    accum_dtype: torch.dtype,
    seq_lens: List[int],
    dtype: torch.dtype,
) -> None:
    test = NsaFwdWorkload(
        batch,
        heads,
        c_seq_len,
        dim,
        is_causal,
        scale,
        block_size,
        heads // head_kv,
        selected_blocks,
        dtype,
        accum_dtype,
        seq_lens=seq_lens,
    )
    inputs = test.gen_inputs()
    op = NSAFwdVarlenOp(
        is_causal=is_causal,
        scale=scale,
        block_size=block_size,
        accum_dtype=accum_dtype,
        tune=_TUNE,
    )

    bm = ManifestBenchmark(op, test)
    bm.compare({"tileops": op, "torch-ref": test.ref_program}, *inputs)
