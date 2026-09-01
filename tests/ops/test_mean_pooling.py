"""Tests for the chunked sequence mean."""

from typing import List, Optional

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.pool import MeanPoolingFwdOp
from workloads.pool import MeanPoolingWorkload, mean_pooling_chunk_index


class MeanPoolingTest(MeanPoolingWorkload, TestBase):
    pass


def _cosine_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Compare by cosine similarity: a long chunk sum accumulates in fp32 and stores in the
    input dtype, so an elementwise tolerance is the wrong instrument."""
    cos_sim = F.cosine_similarity(output_ref, output, dim=-1, eps=1e-8)
    assert cos_sim.min() > 0.99, f"cosine similarity too low: {cos_sim.min().item()}"


class MeanPoolingFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim, chunk_size, dtype, tune, seq_lens",
            [
                pytest.param(
                    1, 8192, 64, 128, 64, torch.float16, False, None, marks=pytest.mark.smoke
                ),
                pytest.param(
                    1, 8192, 64, 128, 64, torch.float16, True, None, marks=pytest.mark.full
                ),
                pytest.param(
                    2, 2048, 64, 128, 64, torch.float16, False, None, marks=pytest.mark.full
                ),
                # Ragged, every sequence ending mid-chunk: the case that divides by a short
                # count.
                pytest.param(
                    1,
                    1000,
                    64,
                    128,
                    32,
                    torch.float16,
                    False,
                    [100, 200, 300, 400],
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2,
                    2048,
                    64,
                    128,
                    64,
                    torch.float16,
                    False,
                    [1024, 1024],
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@MeanPoolingFixture
def test_mean_pooling_op(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
    seq_lens: Optional[List[int]],
) -> None:
    test = MeanPoolingTest(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
        chunk_size=chunk_size,
        dtype=dtype,
        accum_dtype=torch.float32,
        seq_lens=seq_lens,
    )
    op = MeanPoolingFwdOp(chunk_size=chunk_size, accum_dtype=torch.float32, tune=tune)
    test.check(op, *test.gen_inputs(), compare=_cosine_compare)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", [100, 256])
def test_mean_pooling_dim_not_one_full_tile(dim: int) -> None:
    """`dim` values the manifest allows but no workload row carries.

    The kernel tiles `dim` by a width of at most 128: 100 is one tile with a short tail, 256
    is two whole tiles. Between them TileLang finds no layout, which is why the manifest
    rules out that range rather than the op checking for it.
    """
    test = MeanPoolingTest(
        batch=1,
        seq_len=64,
        heads=2,
        dim=dim,
        chunk_size=32,
        dtype=torch.float16,
        accum_dtype=torch.float32,
    )
    op = MeanPoolingFwdOp(chunk_size=32, accum_dtype=torch.float32)
    test.check(op, *test.gen_inputs(), atol=1e-3, rtol=1e-5)


def _op() -> MeanPoolingFwdOp:
    return MeanPoolingFwdOp(chunk_size=32, accum_dtype=torch.float32)


def _x() -> torch.Tensor:
    return torch.randn(1, 64, 2, 64, device="cuda", dtype=torch.float16)


@pytest.mark.smoke
def test_mean_pooling_rejects_a_wrong_offsets_dtype() -> None:
    """`forward` once dispatched a kernel without running the generated validator, so an
    `int64` offsets tensor reached TileLang instead of being rejected here."""
    offsets, indices = mean_pooling_chunk_index([64], 32)
    with pytest.raises(ValueError, match="offsets"):
        _op()(_x(), offsets.to(torch.int64), indices)


@pytest.mark.smoke
def test_mean_pooling_rejects_one_ragged_tensor_without_the_other() -> None:
    """`offsets` and `indices` describe one split, so half of it is a caller error rather
    than a uniform call."""
    offsets, indices = mean_pooling_chunk_index([64], 32)
    with pytest.raises(ValueError, match="either both are passed"):
        _op()(_x(), offsets, None)
    with pytest.raises(ValueError, match="either both are passed"):
        _op()(_x(), None, indices)


@pytest.mark.smoke
def test_mean_pooling_rejects_indices_that_disagree_with_offsets() -> None:
    """The output's chunk axis comes from `indices`, because a shape is all the compile fake
    is handed, so an `indices` that does not match `offsets` is caught rather than believed.
    """
    offsets, indices = mean_pooling_chunk_index([32, 32], 32)
    with pytest.raises(ValueError, match="offsets imply"):
        _op()(_x(), offsets, indices[:-1])
    indices[1, 0], indices[1, 1] = 0, 1
    with pytest.raises(ValueError, match="does not have"):
        _op()(_x(), offsets, indices)
    indices[1, 0], indices[1, 1] = 0, 0
    with pytest.raises(ValueError, match="exactly once"):
        _op()(_x(), offsets, indices)


@pytest.mark.smoke
def test_mean_pooling_rejects_offsets_that_leave_tokens_out() -> None:
    """`offsets` partitions the sequence axis. Bounds that stop short would drop rows from
    the mean without saying so."""
    offsets, indices = mean_pooling_chunk_index([32], 32)
    with pytest.raises(ValueError, match="offsets must run 0 to"):
        _op()(_x(), offsets, indices)


@pytest.mark.smoke
def test_mean_pooling_reads_its_shapes_off_the_call() -> None:
    """One op, two shapes, two chunk counts — what taking shapes from the call buys."""
    op = _op()
    assert tuple(op(_x()).shape) == (1, 2, 2, 64)
    tall = torch.randn(3, 128, 2, 64, device="cuda", dtype=torch.float16)
    assert tuple(op(tall).shape) == (3, 4, 2, 64)
