import math

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops._kernels.indexing import EmbeddingCall
from tileops.backend import OpNotAvailableError
from tileops.indexing import EmbeddingFwdOp
from workloads.device import run_device
from workloads.indexing import EmbeddingWorkload
from workloads.workload_base import manifest_call


class EmbeddingTest(EmbeddingWorkload, TestBase):
    pass


def _test(S, V, D, index_dtype, dtype, **params) -> EmbeddingTest:
    return EmbeddingTest(
        manifest_call("EmbeddingFwdOp", {"I": index_dtype, "T": dtype}, S=S, V=V, D=D, **params)
    )


_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
# One case per dtype and per tiling path. Every case reads the table's last row.
_CASES = (
    # A row of more vectors than a CTA has threads: two passes.
    pytest.param([4, 33], 1000, 2304, "int64", "bfloat16", {}, marks=_SMOKE, id="bf16-two-passes"),
    pytest.param([129], 300, 1152, "int64", "float32", {}, marks=_SMOKE, id="fp32"),
    # Fewer ids than a tile of narrow rows holds: a partial row tile. The gradient-only
    # params ride on it.
    pytest.param(
        [3, 5],
        700,
        64,
        "int32",
        "float16",
        {"padding_idx": -1, "sparse": True},
        marks=_SMOKE,
        id="fp16-row-tail",
    ),  # fmt: skip
    pytest.param([500], 129, 6, "int64", "int64", {}, marks=_SMOKE, id="int64-table"),
    # 12-byte rows: 4-byte vectors.
    pytest.param([2, 9], 40, 3, "int32", "int32", {}, marks=_SMOKE, id="int32-table"),
    # 24-byte rows: 8-byte vectors.
    pytest.param([5, 7], 90, 3, "int32", "int64", {}, marks=_FULL, id="eight-byte-vectors"),
    # An odd row of 2-byte elements: 2-byte vectors, which the evict-last load does not move.
    pytest.param([7], 50, 1001, "int64", "bfloat16", {}, marks=_FULL, id="two-byte-vectors"),
    # 1021 vectors, a prime: no thread count divides them, so the last pass is guarded.
    pytest.param([2], 1, 8168, "int64", "bfloat16", {}, marks=_FULL, id="guarded-pass"),
    # 1025-vector rows: two column tiles, the second holding one vector; more than 2**20
    # tiles, so the grid's y axis holds the rest.
    pytest.param([2**19 + 1], 4, 1025, "int64", "float16", {}, marks=_FULL, id="column-tiles"),
    pytest.param([], 10, 128, "int64", "bfloat16", {}, marks=_FULL, id="rank-0"),
    pytest.param([0, 4], 10, 128, "int64", "bfloat16", {}, marks=_FULL, id="empty"),
    pytest.param([3], 10, 0, "int64", "bfloat16", {}, marks=_FULL, id="zero-width"),
    # A table of more rows than an int32 extent holds.
    pytest.param([3], 2**31 + 1, 1, "int64", "bfloat16", {}, marks=_FULL, id="rows-past-int32"),
    # Column offsets past 2**31 elements in one row, on 16-byte vectors; the last column
    # tile holds one vector.
    pytest.param([2], 1, 2**31 + 8, "int64", "float16", {}, marks=_FULL, id="widest-row"),
)


def _column_tiles(cfg: dict, S: list, D: int) -> bool:
    """Two column tiles, the second one vector, and more tiles than grid x's 2**20."""
    vectors = D // cfg["vec"]
    col_tiles = -(-vectors // cfg["block_vectors"])
    tiles = -(-S[0] // cfg["block_rows"]) * col_tiles
    return col_tiles == 2 and vectors % cfg["block_vectors"] == 1 and tiles > 2**20


# The launch tile each path-named case relies on, so a retune fails the case instead of
# moving it off its path.
_ON_PATH = {
    "guarded-pass": lambda cfg, S, D: cfg["block_rows"] * cfg["block_vectors"] % cfg["threads"]
    != 0,
    "column-tiles": _column_tiles,
}


@pytest.mark.parametrize("S, V, D, index_dtype, dtype, params", _CASES)
def test_embedding(S, V, D, index_dtype, dtype, params, request) -> None:
    test = _test(S, V, D, index_dtype, dtype, **params)
    op = EmbeddingFwdOp(**test.arguments())
    on_path = _ON_PATH.get(request.node.callspec.id)
    if on_path is not None:
        call = EmbeddingCall(
            dim=D,
            dtype=getattr(torch, dtype),
            index_dtype=getattr(torch, index_dtype),
            device=torch.device(run_device()),
        )
        kernel = op.kernel_for("embedding", lambda: call)
        cfg = kernel.tile(kernel.config, kernel.vectors_per_thread(math.prod(S)))
        assert on_path(cfg, S, D), cfg
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_embedding_strided_inputs() -> None:
    """Every second element of a wider id tensor and of a wider table."""
    test = _test([4, 33], 1000, 2304, "int64", "bfloat16")
    input, weight = test.gen_inputs()
    wide_input = torch.zeros(4, 66, dtype=input.dtype, device=run_device())
    wide_input[:, ::2] = input
    wide_weight = torch.zeros(1000, 4608, dtype=weight.dtype, device=run_device())
    wide_weight[:, ::2] = weight
    test.check(EmbeddingFwdOp(), wide_input[:, ::2], wide_weight[:, ::2])


@pytest.mark.smoke
@pytest.mark.parametrize("bad", [-1, 1000], ids=["negative", "past-end"])
def test_embedding_validate_inputs_rejects_out_of_range_ids(bad: int) -> None:
    input, weight = _test([4, 33], 1000, 2304, "int64", "bfloat16").gen_inputs()
    input[1, 3] = bad
    with pytest.raises(ValueError, match="input"):
        EmbeddingFwdOp(validate_inputs=True)(input, weight)


@pytest.mark.smoke
def test_embedding_rejects_float_ids() -> None:
    input, weight = _test([4, 33], 1000, 2304, "int64", "bfloat16").gen_inputs()
    with pytest.raises(ValueError, match="dtype is outside"):
        EmbeddingFwdOp()(input.float(), weight)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_embedding_refuses_max_norm() -> None:
    """No in-tree kernel renormalizes the table in place; the op refuses before building."""
    input, weight = _test([129], 300, 1152, "int64", "float32").gen_inputs()
    before = weight.clone()
    op = EmbeddingFwdOp(max_norm=1.0)
    with pytest.raises(OpNotAvailableError, match="max_norm"):
        op(input, weight)
    assert not op._entries_built("embedding")
    assert torch.equal(weight, before)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "dim, waves, most",
    [
        pytest.param(64, 4, 4, marks=_SMOKE, id="four-waves"),
        pytest.param(64, 2, 2, marks=_FULL, id="two-waves"),
        pytest.param(64, 5, 1, marks=_FULL, id="five-waves"),
        # Rows of 288 vectors: a tile of two rows, so a thread's vectors span both.
        pytest.param(2304, 4, 4, marks=_FULL, id="four-waves-wide-rows"),
    ],
)
def test_embedding_wave_tiles(dim, waves, most) -> None:
    """A call of *waves* waves of resident threads at one vector a thread.

    bf16 rows of 16-byte vectors: up to four waves take more than one vector a thread and no
    more than *most*, on row tiles whose last one is partial; past four, one. The call is
    checked against the reference on the tile it launches.
    """
    call = EmbeddingCall(
        dim=dim, dtype=torch.bfloat16, index_dtype=torch.int32, device=torch.device(run_device())
    )
    ids = waves * call.sm_count * call.threads_per_sm // (dim // 8) - 3
    test = _test([ids], 1000, dim, "int32", "bfloat16")
    op = EmbeddingFwdOp(**test.arguments())
    kernel = op.kernel_for("embedding", lambda: call)
    cfg = kernel.tile(kernel.config, kernel.vectors_per_thread(ids))
    vectors = cfg["block_rows"] * cfg["block_vectors"] / cfg["threads"]
    assert 1 <= vectors <= most and (vectors > 1) == (most > 1), cfg
    assert ids % cfg["block_rows"] != 0, cfg
    test.check(op, *test.gen_inputs())
