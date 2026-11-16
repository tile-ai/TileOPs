import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.backend import OpNotAvailableError
from tileops.indexing import EmbeddingFwdOp
from workloads.device import run_device
from workloads.indexing import EmbeddingWorkload
from workloads.workload_base import manifest_call


class EmbeddingTest(EmbeddingWorkload, TestBase):
    pass


_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
# One case per dtype and per tiling: column tiles of a wide row with a tail, several narrow
# rows a CTA, a row width off the 16-byte vector. The gradient-only params ride on one case.
_CASES = (
    pytest.param([4, 33], 1000, 2304, "int64", "bfloat16", {}, marks=_SMOKE, id="bf16-col-tiles"),
    pytest.param([129], 300, 1152, "int64", "float32", {}, marks=_SMOKE, id="fp32-col-tail"),
    pytest.param(
        [3, 5],
        700,
        64,
        "int32",
        "float16",
        {"padding_idx": -1, "sparse": True},
        marks=_SMOKE,
        id="fp16-rows",
    ),  # fmt: skip
    pytest.param([500], 129, 6, "int64", "int64", {}, marks=_SMOKE, id="int64-table"),
    pytest.param([2, 9], 40, 3, "int32", "int32", {}, marks=_SMOKE, id="int32-odd-dim"),
    pytest.param([], 10, 128, "int64", "bfloat16", {}, marks=_FULL, id="rank-0"),
    pytest.param([0, 4], 10, 128, "int64", "bfloat16", {}, marks=_FULL, id="empty"),
    pytest.param([3], 10, 0, "int64", "bfloat16", {}, marks=_FULL, id="zero-width"),
    # Scalar copies past 65535 column tiles of 128, the grid's y limit, and past 2**31 columns.
    pytest.param([2], 1, 2**31 + 1, "int64", "float16", {}, marks=_FULL, id="widest-row"),
)


@pytest.mark.parametrize("S, V, D, index_dtype, dtype, params", _CASES)
def test_embedding(S, V, D, index_dtype, dtype, params) -> None:
    call = manifest_call("EmbeddingFwdOp", {"I": index_dtype, "T": dtype}, S=S, V=V, D=D, **params)
    test = EmbeddingTest(call)
    test.check(EmbeddingFwdOp(**call.arguments()), *test.gen_inputs())


@pytest.mark.smoke
def test_embedding_row_offset_past_int32() -> None:
    """Rows whose element offset is past 2**31 read the right elements."""
    dim = 8
    rows = 2**31 // dim + 2
    weight = torch.empty(rows, dim, device=run_device(), dtype=torch.bfloat16)
    index = torch.tensor([rows - 1, 0, rows - 2], device=run_device())
    weight[index] = torch.randn(len(index), dim, device=run_device(), dtype=weight.dtype)
    call = manifest_call(
        "EmbeddingFwdOp", {"I": "int64", "T": "bfloat16"}, S=[len(index)], V=rows, D=dim
    )

    EmbeddingTest(call).check(EmbeddingFwdOp(), index, weight)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_embedding_max_norm_is_refused() -> None:
    """No kernel renormalizes rows, so the call refuses before writing the table."""
    weight = torch.randn(10, 16, device=run_device())
    index = torch.tensor([1, 2], device=run_device())

    before = weight.clone()

    with pytest.raises(OpNotAvailableError, match="max_norm"):
        EmbeddingFwdOp(max_norm=1.0)(index, weight)
    assert torch.equal(weight, before)


@pytest.mark.smoke
@pytest.mark.parametrize("bad", [10, -1], ids=["past-end", "negative"])
def test_embedding_validate_inputs_rejects_an_index_outside_the_table(bad: int) -> None:
    weight = torch.randn(10, 16, device=run_device())
    index = torch.tensor([0, bad], device=run_device())

    with pytest.raises(ValueError, match=r"^input must be in \[0, 10\)"):
        EmbeddingFwdOp(validate_inputs=True)(index, weight)
