"""TileOPs refuses a call whose declared outputs would all be empty.

One case per distinct shape of the answer, not one per op: the refusal is decided in
``Op.kernel_for`` for every family, so a second op of the same shape re-tests
the same branch.
"""

import re

import pytest
import torch

from tileops.ops.elementwise import AddFwdOp, ReluFwdOp
from tileops.ops.reduction import SumFwdOp

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DTYPE = torch.float16


def _message(op_class_name: str, input_name: str, shape: tuple) -> str:
    return (
        f"{op_class_name} does not support an empty tensor: input '{input_name}' "
        f"has shape {shape}, which holds no elements."
    )


@pytest.fixture
def empty() -> torch.Tensor:
    return torch.randn(0, 8, device="cuda", dtype=DTYPE)


@pytest.mark.parametrize(
    "call, op_class_name, input_name",
    [
        pytest.param(lambda x: ReluFwdOp()(x), "ReluFwdOp", "input", id="unary"),
        pytest.param(lambda x: AddFwdOp()(x, x), "AddFwdOp", "input", id="binary"),
    ],
)
def test_empty_input_is_refused(empty, call, op_class_name, input_name):
    """The message names the op, the input and its shape."""
    with pytest.raises(ValueError, match=re.escape(_message(op_class_name, input_name, (0, 8)))):
        call(empty)


def test_compiled_call_is_refused_the_same_way(empty):
    """The traced path reaches kernel selection too, so it gets the same message."""
    compiled = torch.compile(ReluFwdOp(), fullgraph=True)
    assert compiled(torch.randn(4, 8, device="cuda", dtype=DTYPE)).shape == (4, 8)
    with pytest.raises(ValueError, match=re.escape(_message("ReluFwdOp", "input", (0, 8)))):
        compiled(empty)


def test_the_op_s_own_validation_precedes_the_refusal():
    """``_eager_forward``'s prelude runs before kernel selection, so it reports first."""
    with pytest.raises(ValueError, match="needs every tensor on one device"):
        AddFwdOp()(torch.empty(0, 2), torch.empty(0, 2, device="cuda"))


def test_the_refusal_precedes_what_the_kernel_states():
    """The empty-input refusal comes before the refusal of a device no kernel runs on."""
    with pytest.raises(ValueError, match=re.escape(_message("ReluFwdOp", "input", (0, 8)))):
        ReluFwdOp()(torch.empty(0, 8))


def test_a_parametric_op_answers_an_empty_input_as_torch_does(empty):
    """A converted entry admits every non-negative extent, so an empty call has a result."""
    torch.testing.assert_close(SumFwdOp(dim=0)(empty), torch.sum(empty, dim=0))
    torch.testing.assert_close(SumFwdOp(dim=1)(empty), torch.sum(empty, dim=1))
