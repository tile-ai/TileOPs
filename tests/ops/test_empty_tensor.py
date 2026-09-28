"""A call that writes no element runs nothing and returns what torch returns.

The rule is decided in ``Op`` for every family, so the cases cover the shapes of the
answer rather than the ops: one output or several, a new output or a written tensor, an op
with or without a compile boundary, and an empty input whose output is not empty.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.ops import FP8QuantFwdOp, FusedAddRMSNormFwdOp, InstanceNormFwdOp
from tileops.ops.elementwise import AddFwdOp, ReluFwdOp
from tileops.ops.moe import ContiguousLayoutSpec, MoePostPermuteFwdOp
from tileops.ops.reduction import SumFwdOp
from workloads.device import run_device, run_device_available

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(not run_device_available(), reason="the run device is not available"),
]

DTYPE = torch.float16


def _tensor(*shape: int, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.randn(*shape, device=run_device()).to(dtype)


def _assert_same(actual: object, expected: object) -> None:
    """Equal structure, shape, dtype and device, and equal values where there are any."""
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            _assert_same(a, e)
        return
    assert (actual.shape, actual.dtype, actual.device) == (
        expected.shape,
        expected.dtype,
        expected.device,
    )
    if expected.numel():
        torch.testing.assert_close(actual, expected)


def _post_permute(out: "torch.Tensor | None" = None) -> torch.Tensor:
    """A routed MoE reduction over no tokens."""
    op = MoePostPermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    weights = torch.empty(0, 2, device=run_device())
    inverse = torch.empty(0, dtype=torch.int32, device=run_device())
    return op(_tensor(0, 64), weights, inverse, out)


@pytest.mark.parametrize(
    "call, reference",
    [
        pytest.param(lambda x: ReluFwdOp()(x), lambda x: torch.relu(x), id="one-output"),
        pytest.param(
            lambda x: FusedAddRMSNormFwdOp()(x, x, _tensor(8)),
            lambda x: (F.rms_norm(x + x, [8]), x + x),
            id="two-outputs",
        ),
        pytest.param(
            lambda x: FP8QuantFwdOp()(x.reshape(0, 4, 1, 2)),
            lambda x: (
                torch.empty(0, 4, 1, device=x.device),
                torch.empty(0, 4, 1, 2, device=x.device, dtype=torch.float8_e4m3fn),
            ),
            id="no-compile-boundary",
        ),
        pytest.param(
            lambda x: SumFwdOp(dim=0)(x), lambda x: torch.sum(x, dim=0), id="non-empty-output"
        ),
    ],
)
def test_an_empty_call_returns_what_torch_returns(call, reference):
    x = _tensor(0, 8)
    _assert_same(call(x), reference(x))


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda t: ReluFwdOp(inplace=True)(t), id="written-input"),
        pytest.param(_post_permute, id="out"),
    ],
)
def test_a_written_empty_tensor_is_returned_as_passed(call):
    written = _tensor(0, 64)
    assert call(written) is written


def test_a_written_input_with_elements_still_runs():
    """A batch of no instances writes the running statistics, as torch writes them."""
    x = _tensor(0, 4, 8, dtype=torch.float32)
    ours = torch.zeros(4, device=run_device()), torch.ones(4, device=run_device())
    theirs = torch.zeros(4, device=run_device()), torch.ones(4, device=run_device())
    InstanceNormFwdOp()(x, *ours)
    F.instance_norm(x, *theirs, use_input_stats=True)
    torch.testing.assert_close(ours, theirs, equal_nan=True)


def test_a_compiled_empty_call_returns_what_torch_returns():
    compiled = torch.compile(ReluFwdOp(), fullgraph=True)
    assert compiled(_tensor(4, 8)).shape == (4, 8)
    _assert_same(compiled(_tensor(0, 8)), torch.relu(_tensor(0, 8)))


@pytest.mark.cuda_only
def test_an_empty_call_is_still_checked():
    """The signature's checks run before the call is found to write nothing."""
    with pytest.raises(ValueError, match="needs every tensor on one device"):
        AddFwdOp()(torch.empty(0, 2), torch.empty(0, 2, device="cuda"))
