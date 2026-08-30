"""Manifest-driven benchmarks for elementwise ops.

These cases keep the legacy risk-matrix benchmarks intact while giving each
implemented elementwise manifest entry a benchmark path that is sourced from
``workloads`` and reports roofline data through ``ManifestBenchmark``.

Each row is timed against torch eager and the same reference through inductor.
"""

import functools
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from tileops.manifest import load_workloads
from tileops.ops.elementwise import (
    AddFwdOp,
    BitwiseAndFwdOp,
    BitwiseOrFwdOp,
    BitwiseXorFwdOp,
    ClampScalarFwdOp,
    DivFwdOp,
    EluFwdOp,
    EqFwdOp,
    FloorDivideFwdOp,
    GeFwdOp,
    GeluFwdOp,
    GtFwdOp,
    HardsigmoidFwdOp,
    HardswishFwdOp,
    HardtanhFwdOp,
    LeakyReluFwdOp,
    LeFwdOp,
    LerpFwdOp,
    LerpTensorFwdOp,
    LogicalAndFwdOp,
    LogicalOrFwdOp,
    LtFwdOp,
    MaskedFillFwdOp,
    MaskedFillScalarFwdOp,
    MaximumFwdOp,
    MinimumFwdOp,
    MishFwdOp,
    MulFwdOp,
    NanToNumFwdOp,
    NeFwdOp,
    PowFwdOp,
    PreluFwdOp,
    ReluFwdOp,
    RemainderFwdOp,
    SeluFwdOp,
    SigmoidFwdOp,
    SiluFwdOp,
    SoftplusFwdOp,
    SubFwdOp,
    TanhFwdOp,
    WhereFwdOp,
)
from workloads.elementwise import (
    BinaryManifestWorkload,
    LerpTensorManifestWorkload,
    MaskedFillScalarManifestWorkload,
    MaskedFillTensorManifestWorkload,
    PreluManifestWorkload,
    ShapedRandnWorkload,
    WhereManifestWorkload,
)


def _mark(w: dict, dtype: torch.dtype, index: int) -> tuple:
    """The first row's fp16 case is the smoke case; every other case is full."""
    return (pytest.mark.smoke if index == 0 and dtype is torch.float16 else pytest.mark.full,)


def _shape_args(w: dict, dtype: torch.dtype) -> tuple:
    return (tuple(w["input_shape"]), dtype)


def _binary_args(w: dict, dtype: torch.dtype, rhs_key: str = "other_shape") -> tuple:
    return (tuple(w["input_shape"]), tuple(w[rhs_key]), dtype)


def _prelu_args(w: dict, dtype: torch.dtype) -> tuple:
    return (tuple(w["input_shape"]), tuple(w["weight_shape"]), dtype)


def _masked_fill_tensor_args(w: dict, dtype: torch.dtype) -> tuple:
    return (tuple(w["input_shape"]), tuple(w["mask_shape"]), tuple(w["value_shape"]), dtype)


def _record_unary(
    op,
    bm: ManifestBenchmark,
    inputs: tuple[torch.Tensor, ...],
    baseline_fn: Callable,
) -> None:
    bm.compare(
        {
            "tileops": op,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
    )


def _record_binary(
    op,
    bm: ManifestBenchmark,
    inputs: tuple[torch.Tensor, torch.Tensor],
    baseline_fn: Callable,
) -> None:
    bm.compare(
        {
            "tileops": op,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
    )


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(ReluFwdOp), _shape_args, marks=_mark)
)
def test_relu_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = ReluFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, F.relu)


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(GeluFwdOp), _shape_args, marks=_mark)
)
def test_gelu_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = GeluFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, lambda x: F.gelu(x, approximate="none"))


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(SiluFwdOp), _shape_args, marks=_mark)
)
def test_silu_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = SiluFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, F.silu)


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(HardswishFwdOp), _shape_args, marks=_mark)
)
def test_hardswish_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = HardswishFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, F.hardswish)


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(HardsigmoidFwdOp), _shape_args, marks=_mark)
)
def test_hardsigmoid_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = HardsigmoidFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, F.hardsigmoid)


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(MishFwdOp), _shape_args, marks=_mark)
)
def test_mish_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = MishFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, F.mish)


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(SeluFwdOp), _shape_args, marks=_mark)
)
def test_selu_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = SeluFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, F.selu)


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(LeakyReluFwdOp), _shape_args, marks=_mark)
)
def test_leaky_relu_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = LeakyReluFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, lambda x: F.leaky_relu(x, 0.01))


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(EluFwdOp), _shape_args, marks=_mark)
)
def test_elu_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = EluFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, lambda x: F.elu(x, 1.0))


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(HardtanhFwdOp), _shape_args, marks=_mark)
)
def test_hardtanh_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = HardtanhFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, lambda x: F.hardtanh(x, -1.0, 1.0))


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(SoftplusFwdOp), _shape_args, marks=_mark)
)
def test_softplus_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = SoftplusFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, lambda x: F.softplus(x, 1.0, 20.0))


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(SigmoidFwdOp), _shape_args, marks=_mark)
)
def test_sigmoid_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = SigmoidFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, torch.sigmoid)


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(TanhFwdOp), _shape_args, marks=_mark)
)
def test_tanh_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = TanhFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, torch.tanh)


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(ClampScalarFwdOp), _shape_args, marks=_mark)
)
def test_clamp_scalar_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = ClampScalarFwdOp(min=-0.5, max=0.5)
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, lambda x: torch.clamp(x, -0.5, 0.5))


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(NanToNumFwdOp), _shape_args, marks=_mark)
)
def test_nan_to_num_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = ShapedRandnWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = NanToNumFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_unary(op, bm, inputs, torch.nan_to_num)


@pytest.mark.parametrize(
    "input_shape, weight_shape, dtype",
    workload_params(load_workloads(PreluFwdOp), _prelu_args, marks=_mark),
)
def test_prelu_manifest_bench(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    test = PreluManifestWorkload(input_shape, weight_shape, dtype)
    x, weight = test.gen_inputs()
    op = PreluFwdOp()
    bm = ManifestBenchmark(op, test)
    bm.compare(
        {
            "tileops": op,
            "torch": F.prelu,
            TORCH_COMPILE_TAG: compiled_reference(F.prelu),
        },
        x,
        weight,
    )


@pytest.mark.parametrize(
    "input_shape, mask_shape, value_shape, dtype",
    workload_params(load_workloads(MaskedFillFwdOp), _masked_fill_tensor_args, marks=_mark),
)
def test_masked_fill_tensor_manifest_bench(
    input_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    test = MaskedFillTensorManifestWorkload(input_shape, mask_shape, value_shape, dtype)
    x, mask, value = test.gen_inputs()
    op = MaskedFillFwdOp()
    bm = ManifestBenchmark(op, test)

    def baseline_fn(a, m, v):
        return a.masked_fill(m, v)

    # The baseline is a clone plus an in-place fill, and the clone is a copy, not a
    # kernel; counting copies is what puts all of it in the reading.
    bm.compare(
        {
            "tileops": op,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        x,
        mask,
        value,
        count_copies=True,
    )


@pytest.mark.parametrize(
    "shape, dtype",
    workload_params(load_workloads(MaskedFillScalarFwdOp), _shape_args, marks=_mark),
)
def test_masked_fill_scalar_manifest_bench(
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    test = MaskedFillScalarManifestWorkload(shape, dtype)
    x, mask = test.gen_inputs()
    op = MaskedFillScalarFwdOp(value=-100.0)
    bm = ManifestBenchmark(op, test)

    def baseline_fn(a, m):
        return a.masked_fill(m, -100.0)

    # See the tensor-value case above: the baseline's clone is a copy, not a kernel.
    bm.compare(
        {
            "tileops": op,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        x,
        mask,
        count_copies=True,
    )


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(AddFwdOp), _binary_args, marks=_mark),
)
def test_add_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = AddFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.add)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(SubFwdOp), _binary_args, marks=_mark),
)
def test_sub_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = SubFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.sub)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(MulFwdOp), _binary_args, marks=_mark),
)
def test_mul_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = MulFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.mul)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(DivFwdOp), _binary_args, marks=_mark),
)
def test_div_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, positive=True)
    inputs = test.gen_inputs()
    op = DivFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.div)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(RemainderFwdOp), _binary_args, marks=_mark),
)
def test_remainder_manifest_bench(
    input_shape: tuple,
    other_shape: tuple,
    dtype: torch.dtype,
) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, positive=True)
    inputs = test.gen_inputs()
    op = RemainderFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.remainder)


@pytest.mark.parametrize(
    "input_shape, exponent_shape, dtype",
    workload_params(
        load_workloads(PowFwdOp),
        functools.partial(_binary_args, rhs_key="exponent_shape"),
        marks=_mark,
    ),
)
def test_pow_manifest_bench(
    input_shape: tuple,
    exponent_shape: tuple,
    dtype: torch.dtype,
) -> None:
    test = BinaryManifestWorkload(input_shape, exponent_shape, dtype, positive=True)
    inputs = test.gen_inputs()
    op = PowFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.pow)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(FloorDivideFwdOp), _binary_args, marks=_mark),
)
def test_floor_divide_manifest_bench(
    input_shape: tuple,
    other_shape: tuple,
    dtype: torch.dtype,
) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, positive=True)
    inputs = test.gen_inputs()
    op = FloorDivideFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.floor_divide)


@pytest.mark.parametrize(
    "input_shape, end_shape, dtype",
    workload_params(
        load_workloads(LerpFwdOp),
        functools.partial(_binary_args, rhs_key="end_shape"),
        marks=_mark,
    ),
)
def test_lerp_manifest_bench(input_shape: tuple, end_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, end_shape, dtype)
    inputs = test.gen_inputs()
    op = LerpFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, lambda a, b: torch.lerp(a, b, 0.5))


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(MaximumFwdOp), _binary_args, marks=_mark),
)
def test_maximum_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = MaximumFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.maximum)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(MinimumFwdOp), _binary_args, marks=_mark),
)
def test_minimum_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = MinimumFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.minimum)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(EqFwdOp), _binary_args, marks=_mark),
)
def test_eq_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = EqFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.eq)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(NeFwdOp), _binary_args, marks=_mark),
)
def test_ne_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = NeFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.ne)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(GtFwdOp), _binary_args, marks=_mark),
)
def test_gt_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = GtFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.gt)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(LtFwdOp), _binary_args, marks=_mark),
)
def test_lt_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = LtFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.lt)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(GeFwdOp), _binary_args, marks=_mark),
)
def test_ge_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = GeFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.ge)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(LeFwdOp), _binary_args, marks=_mark),
)
def test_le_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = LeFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.le)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(LogicalAndFwdOp), _binary_args, marks=_mark),
)
def test_logical_and_manifest_bench(
    input_shape: tuple, other_shape: tuple, dtype: torch.dtype
) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, logical=True)
    inputs = test.gen_inputs()
    op = LogicalAndFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.logical_and)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(LogicalOrFwdOp), _binary_args, marks=_mark),
)
def test_logical_or_manifest_bench(
    input_shape: tuple, other_shape: tuple, dtype: torch.dtype
) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, logical=True)
    inputs = test.gen_inputs()
    op = LogicalOrFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.logical_or)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(BitwiseAndFwdOp), _binary_args, marks=_mark),
)
def test_bitwise_and_manifest_bench(
    input_shape: tuple, other_shape: tuple, dtype: torch.dtype
) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, integer=True)
    inputs = test.gen_inputs()
    op = BitwiseAndFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.bitwise_and)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(BitwiseOrFwdOp), _binary_args, marks=_mark),
)
def test_bitwise_or_manifest_bench(
    input_shape: tuple, other_shape: tuple, dtype: torch.dtype
) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, integer=True)
    inputs = test.gen_inputs()
    op = BitwiseOrFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.bitwise_or)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    workload_params(load_workloads(BitwiseXorFwdOp), _binary_args, marks=_mark),
)
def test_bitwise_xor_manifest_bench(
    input_shape: tuple, other_shape: tuple, dtype: torch.dtype
) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, integer=True)
    inputs = test.gen_inputs()
    op = BitwiseXorFwdOp()
    bm = ManifestBenchmark(op, test)
    _record_binary(op, bm, inputs, torch.bitwise_xor)


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(WhereFwdOp), _shape_args, marks=_mark)
)
def test_where_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = WhereManifestWorkload(shape, dtype)
    cond, x, other = test.gen_inputs()
    op = WhereFwdOp()
    bm = ManifestBenchmark(op, test)
    bm.compare(
        {
            "tileops": op,
            "torch": torch.where,
            TORCH_COMPILE_TAG: compiled_reference(torch.where),
        },
        cond,
        x,
        other,
    )


@pytest.mark.parametrize(
    "shape, dtype", workload_params(load_workloads(LerpTensorFwdOp), _shape_args, marks=_mark)
)
def test_lerp_tensor_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = LerpTensorManifestWorkload(shape, dtype)
    x, end, weight = test.gen_inputs()
    op = LerpTensorFwdOp()
    bm = ManifestBenchmark(op, test)
    bm.compare(
        {
            "tileops": op,
            "torch": torch.lerp,
            TORCH_COMPILE_TAG: compiled_reference(torch.lerp),
        },
        x,
        end,
        weight,
    )
