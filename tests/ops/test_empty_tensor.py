"""TileOPs refuses a call whose declared outputs would all be empty.

One case per distinct shape of the answer, not one per op: the refusal is decided in
``Op.get_or_build_kernel`` for every family, so a second op of the same shape re-tests
the same branch.
"""

import re

import pytest
import torch

from tileops.ops.elementwise import AddFwdOp, ReluFwdOp
from tileops.ops.norm import BatchNormFwdOp, RMSNormFwdOp
from tileops.ops.reduction import SumFwdOp, VarMeanFwdOp

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
        pytest.param(lambda x: SumFwdOp(dim=1)(x), "SumFwdOp", "x", id="reduce_to_empty"),
        pytest.param(
            lambda x: RMSNormFwdOp((8,))(x, torch.ones(8, device=x.device, dtype=x.dtype)),
            "RMSNormFwdOp",
            "x",
            id="norm",
        ),
        pytest.param(lambda x: VarMeanFwdOp(dim=1)(x), "VarMeanFwdOp", "x", id="two_outputs"),
    ],
)
def test_empty_input_is_refused(empty, call, op_class_name, input_name):
    """The message names the op, the input and its shape."""
    with pytest.raises(ValueError, match=re.escape(_message(op_class_name, input_name, (0, 8)))):
        call(empty)


def test_batch_norm_training_is_refused():
    """An op reading a private multi-output from its kernel is refused like any other."""
    x = torch.empty(0, 4, 2, 2, device="cuda", dtype=DTYPE)
    stat = lambda: torch.zeros(4, device="cuda")  # noqa: E731
    with pytest.raises(ValueError, match=re.escape(_message("BatchNormFwdOp", "x", (0, 4, 2, 2)))):
        BatchNormFwdOp(training=True)(x, stat(), stat(), stat(), stat())


def test_compiled_call_is_refused_the_same_way(empty):
    """The traced path reaches kernel selection too, so it gets the same message."""
    compiled = torch.compile(ReluFwdOp(), fullgraph=True)
    assert compiled(torch.randn(4, 8, device="cuda", dtype=DTYPE)).shape == (4, 8)
    with pytest.raises(ValueError, match=re.escape(_message("ReluFwdOp", "input", (0, 8)))):
        compiled(empty)


def test_the_op_s_own_validation_precedes_the_refusal():
    """``_eager_forward``'s prelude runs before kernel selection, so it reports first."""
    with pytest.raises(ValueError, match="needs every input on one device"):
        AddFwdOp()(torch.empty(0, 2), torch.empty(0, 2, device="cuda"))


def test_the_refusal_precedes_what_the_kernel_states():
    """The kernel's device statement is made at launch, which this call never reaches."""
    with pytest.raises(ValueError, match=re.escape(_message("ReluFwdOp", "input", (0, 8)))):
        ReluFwdOp()(torch.empty(0, 8))


def test_an_empty_input_with_a_non_empty_output_is_not_refused(empty):
    """An input's zero-length axis is legitimate where the op still produces something."""
    with pytest.raises(ZeroDivisionError):
        SumFwdOp(dim=0)(empty)
