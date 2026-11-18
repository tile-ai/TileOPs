import pytest
import torch

from tests.compile_contract import register_compile_contract
from tests.workload_test_base import TestBase
from tileops.indexing import ConcatFwdOp
from workloads.device import run_device
from workloads.indexing import ConcatWorkload, TransposedConcatWorkload
from workloads.workload_base import manifest_call


class ConcatTest(ConcatWorkload, TestBase):
    pass


class TransposedConcatTest(TransposedConcatWorkload, TestBase):
    pass


_BF16, _FP16, _FP32 = "bfloat16", "float16", "float32"
_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
# One case per dtype and per copy path: whole 16-byte reads, narrower vectors, rows that end
# inside a column tile, one input, several launches, and empty elements.
_CASES = (
    # Two inputs that each fill whole column tiles, along a negative dim: the MTP decode-4 row.
    pytest.param([[4, 5120], [4, 5120]], -1, [_BF16] * 2, marks=_SMOKE, id="bf16-tiles"),
    # Each input's row ends inside a column tile.
    pytest.param([[3, 40], [3, 24]], 1, [_FP16] * 2, marks=_SMOKE, id="fp16-tails"),
    pytest.param([[2, 3, 8], [2, 5, 8]], 1, [_FP32] * 2, marks=_SMOKE, id="fp32-middle-dim"),
    # bfloat16 and float16 promote to a float32 output: 8-byte reads, converted.
    pytest.param(
        [[16, 64], [16, 32], [16, 8]], -1, [_BF16, _FP16, _BF16], marks=_FULL, id="promote"
    ),
    # A float32 output of a float16 input: each thread copies two vectors. The float16 input's
    # second vectors end inside a CTA's tile; the float32 input's second vectors are all live
    # in its first tile and all skipped in its second.
    pytest.param(
        [[2, 800], [2, 1032]], -1, [_FP16, _FP32], marks=_FULL, id="promote-second-vector"
    ),
    # The same with widths that shrink the vector to one element: narrower stores.
    pytest.param([[2, 259], [2, 3]], -1, [_FP16, _FP32], marks=_FULL, id="promote-second-element"),
    # Along dim 0 the output is one row.
    pytest.param([[5], [7]], 0, [_FP32] * 2, marks=_FULL, id="rank-1"),
    # One input: a launch with one tensor slot.
    pytest.param([[4, 16]], 1, [_BF16], marks=_FULL, id="single"),
    # More inputs than one launch takes.
    pytest.param(
        [[3, 2], [3, 1], [3, 4], [3, 3], [3, 1], [3, 9]], -1, [_FP16] * 6, marks=_FULL, id="six"
    ),
    # An empty element inside the first launch, and one alone after it.
    pytest.param(
        [[4, 8], [4, 0], [4, 8], [4, 8], [4, 0]], 1, [_BF16] * 5, marks=_FULL, id="empty-element"
    ),
    # More rows than the grid's y extent holds: a second launch takes the last row.
    pytest.param([[65536, 8], [65536, 8]], 1, [_BF16] * 2, marks=_FULL, id="rows-past-grid-y"),
)


@pytest.mark.parametrize("S, dim, D", _CASES)
def test_concat(S, dim, D) -> None:
    call = manifest_call("ConcatFwdOp", {"D": D}, S=S, dim=dim)
    test = ConcatTest(call)
    test.check(ConcatFwdOp(**call.arguments()), *test.gen_inputs())


@pytest.mark.smoke
def test_concat_follows_each_call_shape() -> None:
    """One op joins each call's own shapes: new rows and widths, then the first call again."""
    op = ConcatFwdOp(-1)
    for S in ([[3, 40], [3, 24]], [[4, 5120], [4, 5120]], [[3, 40], [3, 24]]):
        test = ConcatTest(manifest_call("ConcatFwdOp", {"D": [_FP16] * 2}, S=S, dim=-1))
        test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_concat_strided_elements() -> None:
    """Elements that are transposed views are joined by their logical elements."""
    call = manifest_call("ConcatFwdOp", {"D": [_BF16] * 2}, S=[[3, 40], [3, 24]], dim=1)
    test = TransposedConcatTest(call)
    test.check(ConcatFwdOp(1), *test.gen_inputs())


@pytest.mark.smoke
def test_concat_rejects_an_integer_element() -> None:
    """The signature takes float16, bfloat16 and float32 elements only."""
    tensors = [torch.zeros(2, 3, device=run_device(), dtype=torch.int32) for _ in range(2)]
    with pytest.raises(ValueError):
        ConcatFwdOp(1)(tensors)


register_compile_contract(ConcatFwdOp)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_concat_compiles_cold() -> None:
    """A cold ``fullgraph=True`` compile joins the elements as the eager op does."""
    call = manifest_call("ConcatFwdOp", {"D": [_BF16, _FP16]}, S=[[3, 40], [3, 24]], dim=-1)
    test = ConcatTest(call)
    op = ConcatFwdOp(-1)
    test.check(op, *test.gen_inputs(), runs=torch.compile(op, fullgraph=True))
