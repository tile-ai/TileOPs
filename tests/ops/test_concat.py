import pytest
import torch

from tests.compile_contract import register_compile_contract
from tests.workload_test_base import FixtureBase, TestBase
from tileops.indexing import ConcatFwdOp
from workloads.device import run_device
from workloads.indexing import ConcatWorkload


class ConcatTest(ConcatWorkload, TestBase):
    pass


_BF16, _FP16, _FP32 = torch.bfloat16, torch.float16, torch.float32
_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
_CASES = (
    pytest.param([(4, 5120), (4, 5120)], -1, [_BF16] * 2, marks=_SMOKE, id="mtp-bf16"),
    pytest.param([(3, 40), (3, 24)], 1, [_FP16] * 2, marks=_SMOKE, id="fp16"),
    pytest.param([(2, 3, 8), (2, 5, 8)], 1, [_FP32] * 2, marks=_SMOKE, id="fp32-middle-dim"),
    pytest.param(
        [(16, 64), (16, 32), (16, 8)], -1, [_BF16, _FP16, _BF16], marks=_FULL, id="promote"
    ),
    # One input's width takes 8 fp16 elements a vector, another only 2.
    pytest.param([(5, 4), (5, 8), (5, 2)], 1, [_FP16] * 3, marks=_FULL, id="narrow-vector"),
    pytest.param([(5,), (7,)], 0, [_FP32] * 2, marks=_FULL, id="rank-1"),
    pytest.param([(4, 6)], 1, [_BF16], marks=_FULL, id="single"),
    # More inputs than one launch takes, at widths off every vector width.
    pytest.param(
        [(3, 2), (3, 1), (3, 4), (3, 3), (3, 1), (3, 9)], -1, [_FP16] * 6, marks=_FULL, id="six"
    ),
    # An empty element inside the first launch, and one alone in the second.
    pytest.param(
        [(4, 8), (4, 0), (4, 8), (4, 8), (4, 0)], 1, [_BF16] * 5, marks=_FULL, id="empty-element"
    ),
    # One element per thread, and more than 2**31 of them in the first input.
    pytest.param(
        [(715827883, 3), (715827883, 1)],
        -1,
        [_FP16] * 2,
        marks=_FULL,
        id="int64-count",
    ),
)


class ConcatFixture(FixtureBase):
    PARAMS = [("shapes, dim, dtypes", _CASES)]


@ConcatFixture
def test_concat(shapes: list, dim: int, dtypes: list) -> None:
    test = ConcatTest(shapes, dim, dtypes)
    test.check(ConcatFwdOp(dim), *test.gen_inputs())


@pytest.mark.smoke
def test_concat_follows_each_call_shape() -> None:
    """One op joins each call's own shapes: new row counts and widths, then the first again."""
    op = ConcatFwdOp(-1)
    for shapes in ([(3, 40), (3, 24)], [(4, 5120), (4, 5120)], [(3, 40), (3, 24)]):
        test = ConcatTest(shapes, -1, [_FP16] * 2)
        test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_concat_strided_elements() -> None:
    """Elements that are transposed views read their logical elements."""
    test = ConcatTest([(4, 5120), (4, 5120)], 1, [_BF16] * 2, strided=True)
    test.check(ConcatFwdOp(1), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtypes, shapes",
    [
        pytest.param([torch.int32, torch.int32], [(2, 3), (2, 3)], id="int-dtype"),
        pytest.param([_BF16, _BF16], [(2, 3), (3, 3)], id="unequal-shape"),
    ],
)
def test_concat_rejects(dtypes: list, shapes: list) -> None:
    tensors = [
        torch.zeros(s, device=run_device(), dtype=d) for s, d in zip(shapes, dtypes, strict=True)
    ]
    with pytest.raises(ValueError):
        ConcatFwdOp(1)(tensors)


register_compile_contract(ConcatFwdOp)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_concat_compiles_cold() -> None:
    test = ConcatTest([(3, 40), (3, 24)], -1, [_BF16, _FP16])
    op = ConcatFwdOp(-1)
    test.check(op, *test.gen_inputs(), runs=torch.compile(op, fullgraph=True))
