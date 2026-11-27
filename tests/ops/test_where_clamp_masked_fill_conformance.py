"""Conformance tests for the where, clamp and masked_fill ops: PyTorch signatures,
broadcasting, and Tensor-bound and one-sided forms."""

import pytest
import torch

from tests.workload_test_base import TestBase
from workloads.device import run_device
from workloads.elementwise import ElementwiseWorkload, Every16BitCase, FullRangeCase

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
    from tileops._ops.elementwise import WhereFwdOp

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
    from tileops._ops.elementwise import WhereFwdOp

    shape = (4, 8)
    cond = torch.zeros(shape, device=run_device(), dtype=bad_dtype)
    inp = torch.randn(shape, device=run_device(), dtype=torch.float16)
    other = torch.randn(shape, device=run_device(), dtype=torch.float16)
    op = WhereFwdOp()
    with pytest.raises(ValueError, match="condition dtype is not bool"):
        op(cond, inp, other)


# ClampFwdOp Tensor min/max


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
    from tileops._ops.elementwise import ClampFwdOp

    inp = torch.randn(input_shape, device=run_device(), dtype=dtype)
    mn = torch.randn(min_shape, device=run_device(), dtype=dtype) - 0.5
    mx = torch.randn(max_shape, device=run_device(), dtype=dtype) + 0.5

    op = ClampFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (inp, mn, mx))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
def test_clamp_both_none_rejected():
    """ClampFwdOp rejects a call with neither bound."""
    from tileops._ops.elementwise import ClampFwdOp

    inp = torch.randn(4, device=run_device(), dtype=torch.float32)
    with pytest.raises(ValueError, match=r"present\(min\) or present\(max\)"):
        ClampFwdOp()(inp)


@pytest.mark.smoke
@pytest.mark.parametrize("number_first", [True, False])
def test_clamp_rejects_a_number_bound_paired_with_a_tensor_bound(number_first):
    """torch.clamp has a Number overload and a Tensor overload; neither mixes the two."""
    from tileops._ops.elementwise import ClampFwdOp

    inp = torch.randn(4, device=run_device(), dtype=torch.float32)
    bounds = (0.0, torch.ones_like(inp)) if number_first else (torch.zeros_like(inp), 1.0)
    with pytest.raises(ValueError, match="member"):
        ClampFwdOp()(inp, *bounds)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_one_clamp_instance_serves_clamp_and_both_one_sided_forms():
    """One instance serves both bounds and either one, each with its own kernel."""
    from tileops._ops.elementwise import ClampFwdOp

    inp = torch.randn(4, device=run_device(), dtype=torch.float32)
    mn = torch.zeros(4, device=run_device(), dtype=torch.float32)
    mx = torch.ones(4, device=run_device(), dtype=torch.float32)

    op = ClampFwdOp()
    workload = ElementwiseWorkload(type(op).__name__, (inp, mn, mx))
    TestBase.check(workload, op, *workload.gen_inputs())
    workload = ElementwiseWorkload(type(op).__name__, (inp, mn, None))
    TestBase.check(workload, op, *workload.gen_inputs())
    workload = ElementwiseWorkload(type(op).__name__, (inp, None, mx))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.parametrize(
    "input_shape, min_shape",
    [
        pytest.param((4, 8), (4, 8), marks=pytest.mark.smoke),
        pytest.param((4, 8), (1, 8), marks=pytest.mark.full),
        pytest.param((4, 8), (), marks=pytest.mark.full),
    ],
)
def test_clamp_min_only_tensor(input_shape, min_shape):
    from tileops._ops.elementwise import ClampFwdOp

    inp = torch.randn(input_shape, device=run_device(), dtype=torch.float32)
    mn = torch.randn(min_shape, device=run_device(), dtype=torch.float32)

    op = ClampFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (inp, mn))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.parametrize(
    "input_shape, max_shape",
    [
        pytest.param((4, 8), (4, 8), marks=pytest.mark.smoke),
        pytest.param((4, 8), (4, 1), marks=pytest.mark.full),
        pytest.param((4, 8), (), marks=pytest.mark.full),
    ],
)
def test_clamp_max_only_tensor(input_shape, max_shape):
    from tileops._ops.elementwise import ClampFwdOp

    inp = torch.randn(input_shape, device=run_device(), dtype=torch.float32)
    mx = torch.randn(max_shape, device=run_device(), dtype=torch.float32)

    op = ClampFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (inp, None, mx))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("min_val, max_val", [(0.1, 0.7), (-1 / 3, None), (None, 2 / 3)])
def test_clamp_number_bounds_every_16bit_value(dtype, min_val, max_val):
    """Number-bound ClampFwdOp equals torch.clamp at every 16-bit pattern, NaN and inf too;
    no bound is exact in either dtype, so one rounded the wrong way moves some element."""
    from tileops._ops.elementwise import ClampFwdOp

    op = ClampFwdOp()

    workload = Every16BitCase(dtype, type(op).__name__, min_val, max_val)
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_min, has_max", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize(
    "broadcast",
    [
        pytest.param(False, id="same-shape", marks=pytest.mark.smoke),
        pytest.param(True, id="row-bounds", marks=pytest.mark.full),
    ],
)
def test_clamp_tensor_bounds_every_16bit_value(dtype, has_min, has_max, broadcast):
    """Tensor-bound ClampFwdOp equals torch.clamp at every 16-bit input pattern, NaN and inf too.

    Same-shape bounds (every pattern, shifted) run the flat program; bounds of one element
    per row run the program that reads them in place.
    """
    from tileops._ops.elementwise import ClampFwdOp

    x = Every16BitCase(dtype, "ClampFwdOp").gen_inputs()[0]
    if broadcast:
        x = x.view(256, 256)
        lo = torch.tensor([float("nan"), float("-inf"), -1 / 3, 0.0], device=run_device())
        hi = torch.tensor([2 / 3, float("nan"), float("inf"), -0.0], device=run_device())
        lo, hi = (b.to(dtype).repeat(64).view(256, 1) for b in (lo, hi))
    else:
        lo, hi = x.roll(1), x.roll(-7)

    op = ClampFwdOp()

    workload = ElementwiseWorkload(
        type(op).__name__, (x, lo if has_min else None, hi if has_max else None)
    )
    TestBase.check(workload, op, *workload.gen_inputs())


# A NaN in the input or a bound gives NaN, as torch.clamp does; float32's min and max drop NaN
# without the kernel's guard. The 16-bit forms are covered by the every-16-bit-value tests.


@pytest.mark.smoke
def test_clamp_tensor_nan_propagation():
    """Tensor min and max."""
    dtype = torch.float32
    from tileops._ops.elementwise import ClampFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device=run_device(), dtype=dtype)
    mn = torch.tensor([-1.0, -1.0, float("nan"), -1.0], device=run_device(), dtype=dtype)
    mx = torch.tensor([1.0, 1.0, 1.0, float("nan")], device=run_device(), dtype=dtype)

    op = ClampFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (x, mn, mx))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
def test_clamp_tensor_nan_propagation_broadcast_bounds():
    """NaN in an input or in a bound read at stride 0 propagates as torch.clamp does."""
    from tileops._ops.elementwise import ClampFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device=run_device()).repeat(4, 1)
    mn = torch.tensor([[-1.0], [float("nan")], [-1.0], [-1.0]], device=run_device())
    mx = torch.tensor([[1.0, 1.0, 1.0, float("nan")]], device=run_device())

    op = ClampFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (x, mn, mx))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
def test_clamp_min_only_nan_propagation():
    """Tensor min only, as torch.clamp_min."""
    dtype = torch.float32
    from tileops._ops.elementwise import ClampFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device=run_device(), dtype=dtype)
    mn = torch.tensor([-1.0, -1.0, float("nan"), -1.0], device=run_device(), dtype=dtype)

    op = ClampFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (x, mn))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
def test_clamp_max_only_nan_propagation():
    """Tensor max only, as torch.clamp_max."""
    dtype = torch.float32
    from tileops._ops.elementwise import ClampFwdOp

    x = torch.tensor([float("nan"), -2.0, 0.0, 2.0], device=run_device(), dtype=dtype)
    mx = torch.tensor([1.0, 1.0, 1.0, float("nan")], device=run_device(), dtype=dtype)

    op = ClampFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (x, None, mx))
    TestBase.check(workload, op, *workload.gen_inputs())


# MaskedFillFwdOp: a 0-dim Tensor or a Number value


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


@pytest.mark.parametrize(
    "input_shape, mask_shape, dtype",
    # Every dtype at one shape in smoke, then each broadcast direction at one dtype.
    [
        pytest.param((4, 8), (4, 8), dtype, marks=pytest.mark.smoke)
        for dtype in (
            torch.bool,
            *_MASKED_FILL_TENSOR_VALUE_INT_DTYPES,
            *_MASKED_FILL_TENSOR_VALUE_FLOAT_DTYPES,
        )
    ]
    + [
        pytest.param(*case, marks=pytest.mark.full)
        for case in (
            ((1, 8), (4, 8), torch.float16),
            ((4, 8), (1, 8), torch.float16),
            ((2, 1), (2, 3), torch.float16),
            ((1, 8), (4, 8), torch.bool),
            ((4, 8), (4, 1), torch.int32),
        )
    ],
)
def test_masked_fill_tensor_value(input_shape, mask_shape, dtype):
    from tileops._ops.elementwise import MaskedFillFwdOp

    inp, mask, value = _masked_fill_tensor_value_inputs(input_shape, mask_shape, dtype)

    op = MaskedFillFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (inp, mask, value))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "input_shape, mask_shape, dtype, value",
    [
        ((64, 96), (96,), torch.bool, True),
        ((64, 96), (64, 1), torch.int32, -7),
        ((64, 96), (1, 96), torch.float16, 0.1),
    ],
)
def test_masked_fill_scalar_broadcast_mask(input_shape, mask_shape, dtype, value):
    """A mask read at stride 0 fills with the scalar converted to the input dtype."""
    from tileops._ops.elementwise import MaskedFillFwdOp

    if dtype == torch.bool:
        inp = torch.randint(0, 2, input_shape, device=run_device()).bool()
    elif dtype == torch.int32:
        inp = torch.randint(-100, 100, input_shape, device=run_device(), dtype=dtype)
    else:
        inp = torch.randn(input_shape, device=run_device(), dtype=dtype)
    mask = torch.randint(0, 2, mask_shape, device=run_device()).bool()

    op = MaskedFillFwdOp()

    workload = ElementwiseWorkload(type(op).__name__, (inp, mask, value))
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
def test_masked_fill_scalar_long_int64():
    """int64 input long enough to fill the device several times over."""
    from tileops._ops.elementwise import MaskedFillFwdOp

    n = 16_777_216
    workload = FullRangeCase((((n,), torch.int64), ((n,), torch.bool)), "MaskedFillFwdOp", 7)
    TestBase.check(workload, MaskedFillFwdOp(), *workload.gen_inputs())


@pytest.mark.parametrize(
    "op_name",
    [
        pytest.param("WhereFwdOp", marks=pytest.mark.smoke),
        pytest.param("tensor-value", marks=pytest.mark.full),
        pytest.param("number-value", marks=pytest.mark.full),
    ],
)
def test_uint8_selector_tuned(op_name):
    """Tuned mode tunes a kernel that takes its bool selector as uint8."""
    import tileops._ops.elementwise as ops

    shape, dtype = (1000, 24), torch.float16
    x, y = (torch.randn(shape, device=run_device(), dtype=dtype) for _ in range(2))
    mask = torch.randint(0, 2, shape, device=run_device()).bool()
    name, inputs = {
        "WhereFwdOp": ("WhereFwdOp", (mask, x, y)),
        "tensor-value": (
            "MaskedFillFwdOp",
            (x, mask, torch.tensor(-1.5, device=run_device(), dtype=dtype)),
        ),
        "number-value": ("MaskedFillFwdOp", (x, mask, -1.5)),
    }[op_name]
    op = getattr(ops, name)()
    op.request_tune()
    workload = ElementwiseWorkload(name, inputs)
    TestBase.check(workload, op, *workload.gen_inputs())
