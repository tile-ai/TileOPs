"""EmbeddingFwdOp: row lookup by id against advanced indexing."""

import pytest
import torch

from tests.compile_contract import register_compile_contract
from tests.workload_test_base import TestBase
from tileops.backend import OpNotAvailableError
from tileops.indexing import EmbeddingFwdOp
from workloads.indexing import EmbeddingCall
from workloads.workload_base import manifest_call


def _workload(I, T, strided=False, **row):
    return EmbeddingCall(manifest_call("EmbeddingFwdOp", {"I": I, "T": T}, **row), strided)


@pytest.mark.parametrize(
    "I, T, S, V, D, tune",
    [
        # Each table and id dtype. The token counts end in a partial row tile; D = 1032 and
        # 1152 end in a partial column tile.
        pytest.param("int64", "bfloat16", [2, 33], 1000, 1032, False, marks=pytest.mark.smoke),
        pytest.param("int32", "float16", [65], 500, 1032, False, marks=pytest.mark.smoke),
        pytest.param("int64", "float32", [3, 5, 3], 2304, 1152, False, marks=pytest.mark.smoke),
        pytest.param("int64", "int64", [5, 7], 1000, 6, False, marks=pytest.mark.smoke),
        pytest.param("int32", "int32", [9], 300, 6, False, marks=pytest.mark.smoke),
        # Rank-0 ids; one column, so no vector wider than one element.
        pytest.param("int32", "bfloat16", [], 17, 1, False, marks=pytest.mark.full),
        # A table above 2^31 elements, whose last row needs a 64-bit offset.
        pytest.param("int64", "bfloat16", [2, 8], 1 << 24, 160, False, marks=pytest.mark.full),
        # A table of more than 2^31 - 1 rows.
        pytest.param("int64", "bfloat16", [4], (1 << 31) + 1, 1, False, marks=pytest.mark.full),
        # Rows of 2^25 elements: more column tiles than a CUDA grid's y axis holds, and more
        # than 2^20 tiles in all.
        pytest.param("int32", "bfloat16", [65], 2, 1 << 25, False, marks=pytest.mark.full),
        # Tuned mode, on a row as wide.
        pytest.param("int64", "bfloat16", [3], 2, 1 << 25, True, marks=pytest.mark.full),
    ],
)
def test_embedding(I, T, S, V, D, tune) -> None:
    workload = _workload(I, T, S=S, V=V, D=D)
    op = EmbeddingFwdOp()
    if tune:
        op.request_tune()
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
def test_embedding_strided_inputs() -> None:
    """Every second element of a wider id tensor and of a wider table."""
    workload = _workload("int64", "bfloat16", strided=True, S=[3, 11], V=1000, D=1032)
    TestBase.check(workload, EmbeddingFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("bad", [-1, 64])
def test_embedding_validate_inputs_rejects_out_of_range_ids(bad) -> None:
    input, weight = _workload("int64", "bfloat16", S=[8], V=64, D=128).gen_inputs()
    input[3] = bad
    with pytest.raises(ValueError, match="input"):
        EmbeddingFwdOp(validate_inputs=True)(input, weight)


@pytest.mark.smoke
def test_embedding_rejects_float_ids() -> None:
    input, weight = _workload("int64", "bfloat16", S=[8], V=64, D=128).gen_inputs()
    with pytest.raises(ValueError, match="dtype is outside"):
        EmbeddingFwdOp()(input.float(), weight)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_embedding_refuses_max_norm() -> None:
    """No in-tree kernel renormalizes the table in place; the op refuses before building."""
    input, weight = _workload("int64", "float32", S=[8], V=64, D=128).gen_inputs()
    op = EmbeddingFwdOp(max_norm=1.0)
    with pytest.raises(OpNotAvailableError, match="max_norm"):
        op(input, weight)
    assert not op._entries_built("embedding")


register_compile_contract(EmbeddingFwdOp)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_embedding_compiles_cold() -> None:
    workload = _workload("int64", "bfloat16", S=[2, 33], V=1000, D=1032)
    op = EmbeddingFwdOp()
    TestBase.check(workload, op, *workload.gen_inputs(), runs=torch.compile(op, fullgraph=True))
