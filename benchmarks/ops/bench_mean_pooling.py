"""Benchmarks for MeanPoolingFwdOp, the chunked sequence mean."""

from typing import List, Optional

import pytest
import torch

from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
)
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    fields,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.pool import MeanPoolingFwdOp
from workloads.pool import MeanPoolingWorkload

# Autotuning is a bench-run policy; manifest workloads do not carry it.
_TUNE = True


_MEAN_POOLING_PARAMS = workload_params(
    load_workloads(MeanPoolingFwdOp),
    fields(
        "batch",
        "seq_len",
        "heads",
        "dim",
        "chunk_size",
        "accum_dtype",
        "seq_lens",
        dtype_last=True,
    ),
    smoke_first=True,
)


def _torch_view_mean(test: MeanPoolingWorkload):
    """The same mean over a reshaped view, or None where the chunks are ragged.

    ``ref_program`` averages one slice per chunk, a launch each. Where every chunk is full
    the chunk axis is a reshape away and the pooling is a single reduction.
    """
    if test.seq_lens is not None or test.seq_len % test.chunk_size:
        return None

    chunks = test.seq_len // test.chunk_size

    def fn(x, *_):
        b, _, h, d = x.shape
        return x.view(b, chunks, test.chunk_size, h, d).mean(dim=2)

    return fn


@pytest.mark.parametrize(
    "batch, seq_len, heads, dim, chunk_size, accum_dtype, seq_lens, dtype",
    _MEAN_POOLING_PARAMS,
)
def test_mean_pooling_bench(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    chunk_size: int,
    accum_dtype: torch.dtype,
    seq_lens: Optional[List[int]],
    dtype: torch.dtype,
) -> None:
    test = MeanPoolingWorkload(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
        chunk_size=chunk_size,
        dtype=dtype,
        accum_dtype=accum_dtype,
        seq_lens=seq_lens,
    )
    op = MeanPoolingFwdOp(chunk_size=chunk_size, accum_dtype=accum_dtype, tune=_TUNE)

    inputs = test.gen_inputs()
    bm = ManifestBenchmark(op, test)

    functors = {
        "tileops": op,
        "torch-ref": test.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
    }
    view_mean = _torch_view_mean(test)
    if view_mean is not None:
        assert_matches_reference(view_mean, test.ref_program, *inputs)
        functors["torch-view-mean"] = view_mean

    bm.compare(functors, *inputs)
