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
# One case per dtype and per copy path: whole 16-byte reads, narrower vectors, one or several
# vectors per thread, rows that end inside a column tile, one input, several launches, and
# empty elements.
_CASES = (
    # Two inputs that each fill whole column tiles, along a negative dim: the MTP decode-4 row.
    pytest.param([[4, 5120], [4, 5120]], -1, [_BF16] * 2, marks=_SMOKE, id="bf16-tiles"),
    # Each input's row ends inside a column tile.
    pytest.param([[3, 40], [3, 24]], 1, [_FP16] * 2, marks=_SMOKE, id="fp16-tails"),
    pytest.param([[2, 3, 8], [2, 5, 8]], 1, [_FP32] * 2, marks=_SMOKE, id="fp32-middle-dim"),
    # Rows whose one-vector CTAs are all resident at once: each thread takes more vectors
    # until there are at most as many CTAs as SMs.
    pytest.param(
        [[32, 5120], [32, 5120]], -1, [_BF16] * 2, marks=_FULL, id="bf16-resident-32-rows"
    ),
    # The MTP draft row: its CTAs are all resident at once, so each thread copies eight
    # vectors, the last ones past the input's row end.
    pytest.param(
        [[64, 5120], [64, 5120]], -1, [_BF16] * 2, marks=_FULL, id="bf16-eight-per-thread"
    ),
    # Widths that make a vector 8 bytes: two per thread, each an 8-byte store, the second
    # one past the row end in the last tile.
    pytest.param([[4, 1036], [4, 1044]], 1, [_BF16] * 2, marks=_FULL, id="bf16-8-byte-vectors"),
    # Widths that make a vector 4 bytes: four per thread, each a 4-byte store of two elements.
    pytest.param([[4, 1034], [4, 1030]], 1, [_BF16] * 2, marks=_FULL, id="bf16-4-byte-vectors"),
    # bfloat16 and float16 promote to a float32 output: 8-byte reads, two per thread, converted.
    pytest.param(
        [[16, 1200], [16, 1032], [16, 8]], -1, [_BF16, _FP16, _BF16], marks=_FULL, id="promote"
    ),
    # float16 and float32 promote to float32: the float16 input takes twice the float32
    # input's vectors a thread, so each input's tiles take its own count.
    pytest.param(
        [[16, 4096], [16, 4096]], -1, [_FP16, _FP32], marks=_FULL, id="mixed-reads-per-thread"
    ),
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
