"""Conformance tests for the where, clamp and masked_fill ops.

Covers PyTorch-aligned signatures, broadcasting semantics, and split
variants (Tensor-bound clamp / masked_fill, single-bound clamp_min /
clamp_max). Once these tests pass, the corresponding manifest entries
can flip from ``status: spec-only`` to ``status: implemented`` per the
manifest spec rules (.claude/domain-rules/manifest-spec.md).
"""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.ops.elementwise._base import ELEMENTWISE
from workloads.device import run_device
from workloads.elementwise import ElementwiseWorkload

# WhereFwdOp full broadcasting


@pytest.mark.smoke
@pytest.mark.parametrize(
    "cond_shape, inp_shape, other_shape, dtype",
    [
        ((4, 8), (4, 8), (4, 8), torch.float16),  # same shape
        ((1, 8), (4, 1), (1, 1), torch.float32),  # full 3-way broadcast
        ((4, 8), (1, 8), (4, 1), torch.bfloat16),  # mixed broadcast
        ((), (4, 8), (4, 8), torch.float16),  # 0-dim condition
    ],
)
def test_where_broadcast_parity(cond_shape, inp_shape, other_shape, dtype):
    from tileops.ops.elementwise import WhereFwdOp

    cond = (
        torch.randint(0, 2, cond_shape, device=run_device()).bool()
        if cond_shape
        else torch.tensor(True, device=run_device())
    )
    inp = torch.randn(inp_shape, device=run_device(), dtype=dtype)
    other = torch.randn(other_shape, device=run_device(), dtype=dtype)

    op = WhereFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (cond, inp, other))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_dtype",
    [torch.float32, torch.int32],
)
def test_where_rejects_non_bool_condition(bad_dtype):
    from tileops.ops.elementwise import WhereFwdOp

    shape = (4, 8)
    cond = torch.zeros(shape, device=run_device(), dtype=bad_dtype)
    inp = torch.randn(shape, device=run_device(), dtype=torch.float16)
    other = torch.randn(shape, device=run_device(), dtype=torch.float16)
    op = WhereFwdOp()
    with pytest.raises(ValueError, match="condition dtype is not bool"):
        op(cond, inp, other)


# ClampTensorFwdOp Tensor min/max


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, min_shape, max_shape, dtype",
    [
        ((4, 8), (4, 8), (4, 8), torch.float16),
        ((4, 8), (1, 8), (4, 1), torch.float32),
        ((4, 8), (), (), torch.bfloat16),  # 0-dim Tensor bounds
    ],
)
def test_clamp_tensor_bounds_parity(input_shape, min_shape, max_shape, dtype):
    from tileops.ops.elementwise import ClampTensorFwdOp

    inp = torch.randn(input_shape, device=run_device(), dtype=dtype)
    mn = torch.randn(min_shape, device=run_device(), dtype=dtype) - 0.5
    mx = torch.randn(max_shape, device=run_device(), dtype=dtype) + 0.5
    # Make max >= min where tested ranges overlap: PyTorch clamp tolerates a
    # mismatch, but the reference is only meaningful without one.

    op = ClampTensorFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (inp, mn, mx))
    TestBase.check(workload, op, *workload.gen_inputs())


# ClampTensorFwdOp must accept Tensor min with max=None and
# Tensor max with min=None, matching torch.clamp(input, min=tensor, max=None)
# and torch.clamp(input, min=None, max=tensor) on CUDA. The single-bound shape
# matrix is covered by test_clamp_min_only_tensor / test_clamp_max_only_tensor;
# these two verify the None routing at one shape.
@pytest.mark.smoke
def test_clamp_min_only_none_routing():
    from tileops.ops.elementwise import ClampTensorFwdOp

    inp = torch.randn((4, 8), device=run_device(), dtype=torch.float32)
    mn = torch.randn((4, 8), device=run_device(), dtype=torch.float32) - 0.5

    op = ClampTensorFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (inp, mn, None))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
def test_clamp_max_only_none_routing():
    from tileops.ops.elementwise import ClampTensorFwdOp

    inp = torch.randn((4, 8), device=run_device(), dtype=torch.float32)
    mx = torch.randn((4, 8), device=run_device(), dtype=torch.float32) + 0.5

    op = ClampTensorFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (inp, None, mx))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
def test_clamp_both_none_rejected():
    """ClampTensorFwdOp must reject a call with neither bound (a no-op clamp is invalid).

    Which bounds it serves is a fact of the call, so the refusal is too.
    """
    from tileops.ops.elementwise import ClampTensorFwdOp

    inp = torch.randn(4, device=run_device(), dtype=torch.float32)
    with pytest.raises(ValueError, match="ClampOut"):
        ClampTensorFwdOp()(inp)


@pytest.mark.smoke
def test_clamp_scalar_both_none_rejected():
    """ClampScalarFwdOp must reject min=None and max=None.

    Mirrors torch.clamp(input, None, None), which raises
    RuntimeError("At least one of min or max must not be None").
    """
    from tileops.ops.elementwise import ClampScalarFwdOp

    with pytest.raises(ValueError, match=r"present\(min\) or present\(max\)"):
        ClampScalarFwdOp(min=None, max=None)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_one_clamp_instance_serves_clamp_and_both_one_sided_forms():
    """Which bounds a call passes is read off the call, so one instance serves all three.

    Each presence pattern needs its own kernel — ``has_min`` / ``has_max`` change what
    gets built — so this also pins that the three do not share one.
    """
    from tileops.ops.elementwise import ClampTensorFwdOp

    inp = torch.randn(4, device=run_device(), dtype=torch.float32)
    mn = torch.zeros(4, device=run_device(), dtype=torch.float32)
    mx = torch.ones(4, device=run_device(), dtype=torch.float32)

    op = ClampTensorFwdOp()
    workload = ElementwiseWorkload(type(op).__name__, (inp, mn, mx))
    TestBase.check(workload, op, *workload.gen_inputs())
    workload = ElementwiseWorkload(type(op).__name__, (inp, mn, None))
    TestBase.check(workload, op, *workload.gen_inputs())
    workload = ElementwiseWorkload(type(op).__name__, (inp, None, mx))
    TestBase.check(workload, op, *workload.gen_inputs())

    assert len(op.built_kernels(ELEMENTWISE)) == 3, "one kernel per presence pattern"


# ClampScalarFwdOp, and ClampTensorFwdOp with one bound withheld


@pytest.mark.smoke
@pytest.mark.parametrize(
    "min_val, max_val",
    [(-0.5, 0.5), (None, 0.5), (-0.5, None)],
)
def test_clamp_scalar_param_names(min_val, max_val):
    from tileops.ops.elementwise import ClampScalarFwdOp

    inp = torch.randn(1024, device=run_device(), dtype=torch.float32)

    op = ClampScalarFwdOp(min=min_val, max=max_val)

    workload = ElementwiseWorkload(type(op).__name__, (inp,), min=min_val, max=max_val)
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, min_shape",
    [((4, 8), (4, 8)), ((4, 8), (1, 8)), ((4, 8), ())],
)
def test_clamp_min_only_tensor(input_shape, min_shape):
    from tileops.ops.elementwise import ClampTensorFwdOp

    inp = torch.randn(input_shape, device=run_device(), dtype=torch.float32)
    mn = torch.randn(min_shape, device=run_device(), dtype=torch.float32)

    op = ClampTensorFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (inp, mn))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, max_shape",
    [((4, 8), (4, 8)), ((4, 8), (4, 1)), ((4, 8), ())],
)
def test_clamp_max_only_tensor(input_shape, max_shape):
    from tileops.ops.elementwise import ClampTensorFwdOp

    inp = torch.randn(input_shape, device=run_device(), dtype=torch.float32)
    mx = torch.randn(max_shape, device=run_device(), dtype=torch.float32)

    op = ClampTensorFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (inp, None, mx))
    TestBase.check(workload, op, *workload.gen_inputs())


# Regression: NaN propagation for Tensor-bound clamp variants.
#
# torch.clamp / torch.clamp_min / torch.clamp_max propagate NaN: if any of
# input / min / max is NaN at position i, the output at i is NaN. CUDA's
# fmax / fmin (used by T.max / T.min) drop NaN by returning the non-NaN
# operand, so the kernel adds explicit isnan guards. These tests pin the
# semantics so a future refactor cannot regress to non-IEEE behaviour.


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_clamp_tensor_nan_propagation(dtype):
    """ClampTensorFwdOp must match torch.clamp NaN semantics (Tensor min + max)."""
    from tileops.ops.elementwise import ClampTensorFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device=run_device(), dtype=dtype)
    mn = torch.tensor([-1.0, -1.0, float("nan"), -1.0], device=run_device(), dtype=dtype)
    mx = torch.tensor([1.0, 1.0, 1.0, float("nan")], device=run_device(), dtype=dtype)

    op = ClampTensorFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (x, mn, mx))
    TestBase.check(workload, op, *workload.gen_inputs())


# Single-bound NaN behaviour is covered by test_clamp_min_nan_propagation /
# test_clamp_max_only_nan_propagation below exercises the same
# ClampTensorFwdKernel branches (has_min only / has_max only).


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_clamp_min_only_nan_propagation(dtype):
    """A min-only clamp must match torch.clamp_min NaN semantics."""
    from tileops.ops.elementwise import ClampTensorFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device=run_device(), dtype=dtype)
    mn = torch.tensor([-1.0, -1.0, float("nan"), -1.0], device=run_device(), dtype=dtype)

    op = ClampTensorFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (x, mn))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_clamp_max_only_nan_propagation(dtype):
    """A max-only clamp must match torch.clamp_max NaN semantics."""
    from tileops.ops.elementwise import ClampTensorFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device=run_device(), dtype=dtype)
    mx = torch.tensor([1.0, 1.0, 1.0, float("nan")], device=run_device(), dtype=dtype)

    op = ClampTensorFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (x, None, mx))
    TestBase.check(workload, op, *workload.gen_inputs())


# MaskedFillTensorFwdOp (0-dim Tensor) / MaskedFillScalarFwdOp (Number)


_MASKED_FILL_TENSOR_VALUE_FLOAT_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]
_MASKED_FILL_TENSOR_VALUE_INT_DTYPES = [
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]


def _masked_fill_tensor_value_inputs(input_shape, mask_shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, input_shape, device=run_device()).bool()
        value = torch.tensor(True, device=run_device(), dtype=torch.bool)
    elif dtype in _MASKED_FILL_TENSOR_VALUE_INT_DTYPES:
        iinfo = torch.iinfo(dtype)
        lo = max(iinfo.min, -1000)
        hi = min(iinfo.max, 1000) + 1
        inp = torch.randint(lo, hi, input_shape, device=run_device(), dtype=dtype)
        value = torch.tensor(7, device=run_device(), dtype=dtype)
    else:
        inp = torch.randn(input_shape, device=run_device(), dtype=dtype)
        value = torch.tensor(-1.5, device=run_device(), dtype=dtype)
    mask = torch.randint(0, 2, mask_shape, device=run_device()).bool()
    return inp, mask, value


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, mask_shape",
    [((4, 8), (4, 8)), ((1, 8), (4, 8)), ((4, 8), (1, 8)), ((2, 1), (2, 3))],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bool, *_MASKED_FILL_TENSOR_VALUE_INT_DTYPES, *_MASKED_FILL_TENSOR_VALUE_FLOAT_DTYPES],
)
def test_masked_fill_tensor_value(input_shape, mask_shape, dtype):
    from tileops.ops.elementwise import MaskedFillTensorFwdOp

    inp, mask, value = _masked_fill_tensor_value_inputs(input_shape, mask_shape, dtype)

    op = MaskedFillTensorFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (inp, mask, value))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
def test_masked_fill_scalar_param_names():
    from tileops.ops.elementwise import MaskedFillScalarFwdOp

    inp = torch.randn(1024, device=run_device(), dtype=torch.float32)
    mask = torch.randint(0, 2, (1024,), device=run_device()).bool()

    op = MaskedFillScalarFwdOp(value=-1.0)

    workload = ElementwiseWorkload(type(op).__name__, (inp, mask), value=-1.0)
    TestBase.check(workload, op, *workload.gen_inputs())
