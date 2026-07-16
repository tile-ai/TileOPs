"""Manifest-driven benchmarks for elementwise ops.

These cases keep the legacy risk-matrix benchmarks intact while giving each
implemented elementwise manifest entry a benchmark path that is sourced from
``workloads`` and reports roofline data through ``ManifestBenchmark``.
"""

from math import prod
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
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


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _mark(idx: int, dtype: torch.dtype):
    return pytest.mark.smoke if idx == 0 and dtype is torch.float16 else pytest.mark.full


def _shape_dtype_params(workloads: list[dict], shape_key: str = "input_shape") -> list:
    params = []
    for idx, w in enumerate(workloads):
        shape = tuple(w[shape_key])
        label = w.get("label", "x".join(str(dim) for dim in shape))
        for dtype_name in w["dtypes"]:
            dtype = _dtype(dtype_name)
            params.append(
                pytest.param(shape, dtype, id=f"{label}-{dtype_name}", marks=_mark(idx, dtype))
            )
    return params


def _binary_params(workloads: list[dict], rhs_key: str = "other_shape") -> list:
    params = []
    for idx, w in enumerate(workloads):
        input_shape = tuple(w["input_shape"])
        other_shape = tuple(w[rhs_key])
        label = w.get("label", "x".join(str(dim) for dim in input_shape))
        for dtype_name in w["dtypes"]:
            dtype = _dtype(dtype_name)
            params.append(
                pytest.param(
                    input_shape,
                    other_shape,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=_mark(idx, dtype),
                )
            )
    return params


class UnaryManifestWorkload:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self.shape, device="cuda", dtype=self.dtype),)


class BinaryManifestWorkload:
    def __init__(
        self,
        input_shape: tuple[int, ...],
        other_shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        positive: bool = False,
        integer: bool = False,
        logical: bool = False,
    ):
        self.input_shape = input_shape
        self.other_shape = other_shape
        self.a_shape = input_shape
        self.b_shape = other_shape
        self.shape = tuple(torch.broadcast_shapes(input_shape, other_shape))
        self.n_total = prod(self.shape)
        self.dtype = dtype
        self.positive = positive
        self.integer = integer
        self.logical = logical

    def _tensor(self, shape: tuple[int, ...]) -> torch.Tensor:
        if self.dtype is torch.bool:
            return torch.randint(0, 2, shape, device="cuda", dtype=torch.bool)
        if self.integer:
            return torch.randint(-1000, 1000, shape, device="cuda", dtype=self.dtype)
        if self.positive:
            return torch.rand(shape, device="cuda", dtype=self.dtype) + 0.1
        if self.logical:
            return (torch.randn(shape, device="cuda", dtype=self.dtype) > 0).to(self.dtype)
        return torch.randn(shape, device="cuda", dtype=self.dtype)

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._tensor(self.input_shape), self._tensor(self.other_shape)


class PreluManifestWorkload:
    def __init__(
        self,
        input_shape: tuple[int, ...],
        weight_shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        self.input_shape = input_shape
        self.weight_shape = weight_shape
        self.shape = input_shape
        self.n_total = prod(input_shape)
        self.dtype = dtype

    @property
    def num_channels(self) -> int:
        return self.weight_shape[0] if self.weight_shape else 1

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.input_shape, device="cuda", dtype=self.dtype)
        weight = torch.rand(self.weight_shape, device="cuda", dtype=self.dtype)
        return x, weight


def _prelu_params(workloads: list[dict]) -> list:
    params = []
    for idx, w in enumerate(workloads):
        input_shape = tuple(w["input_shape"])
        weight_shape = tuple(w["weight_shape"])
        label = w.get("label", "prelu")
        for dtype_name in w["dtypes"]:
            dtype = _dtype(dtype_name)
            params.append(
                pytest.param(
                    input_shape,
                    weight_shape,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=_mark(idx, dtype),
                )
            )
    return params


class MaskedFillTensorManifestWorkload:
    def __init__(
        self,
        input_shape: tuple[int, ...],
        mask_shape: tuple[int, ...],
        value_shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        self.input_shape = input_shape
        self.mask_shape = mask_shape
        self.value_shape = value_shape
        self.shape = tuple(torch.broadcast_shapes(input_shape, mask_shape))
        self.n_total = prod(self.shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.input_shape, device="cuda", dtype=self.dtype)
        mask = torch.rand(self.mask_shape, device="cuda") > 0.5
        value = torch.full(self.value_shape, -100.0, device="cuda", dtype=self.dtype)
        return x, mask, value


def _masked_fill_tensor_params(workloads: list[dict]) -> list:
    params = []
    for idx, w in enumerate(workloads):
        input_shape = tuple(w["input_shape"])
        mask_shape = tuple(w["mask_shape"])
        value_shape = tuple(w["value_shape"])
        label = w.get("label", "masked-fill")
        for dtype_name in w["dtypes"]:
            dtype = _dtype(dtype_name)
            params.append(
                pytest.param(
                    input_shape,
                    mask_shape,
                    value_shape,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=_mark(idx, dtype),
                )
            )
    return params


class MaskedFillScalarManifestWorkload:
    def __init__(self, input_shape: tuple[int, ...], dtype: torch.dtype):
        self.input_shape = input_shape
        self.mask_shape = input_shape
        self.shape = input_shape
        self.n_total = prod(input_shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.input_shape, device="cuda", dtype=self.dtype)
        mask = torch.rand(self.mask_shape, device="cuda") > 0.5
        return x, mask


class WhereManifestWorkload:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.condition_shape = shape
        self.input_shape = shape
        self.other_shape = shape
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = torch.rand(self.condition_shape, device="cuda") > 0.5
        x = torch.randn(self.input_shape, device="cuda", dtype=self.dtype)
        y = torch.randn(self.other_shape, device="cuda", dtype=self.dtype)
        return cond, x, y


class LerpTensorManifestWorkload:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.input_shape = shape
        self.end_shape = shape
        self.weight_shape = shape
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.input_shape, device="cuda", dtype=self.dtype)
        end = torch.randn(self.end_shape, device="cuda", dtype=self.dtype)
        weight = torch.rand(self.weight_shape, device="cuda", dtype=self.dtype)
        return x, end, weight


def _numel(shape: tuple[int, ...]) -> int:
    return prod(shape) if shape else 1


def _broadcast_kind(
    input_shape: tuple[int, ...],
    other_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> str:
    if input_shape == output_shape and other_shape == output_shape:
        return "same_shape"
    if _numel(input_shape) == 1 or _numel(other_shape) == 1:
        return "scalar_broadcast"

    def _one_side_kind(dense_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> str | None:
        if dense_shape != output_shape:
            return None
        if (
            len(output_shape) >= 4
            and len(rhs_shape) >= 3
            and rhs_shape[-2:] == (1, 1)
            and rhs_shape[-3] == output_shape[-3]
            and all(dim == 1 for dim in rhs_shape[:-3])
        ):
            return "channel_broadcast"
        if (
            len(output_shape) >= 2
            and len(rhs_shape) >= 1
            and rhs_shape[-1] == output_shape[-1]
            and all(dim == 1 for dim in rhs_shape[:-1])
        ):
            return "last_dim_broadcast"
        return None

    rhs_kind = _one_side_kind(input_shape, other_shape)
    if rhs_kind is not None:
        return rhs_kind
    lhs_kind = _one_side_kind(other_shape, input_shape)
    if lhs_kind is not None:
        return "lhs_" + lhs_kind
    return "broadcast"


def _manifest_params(bm: ManifestBenchmark) -> dict:
    workload = bm.workload
    params = {}
    for attr in (
        "shape",
        "input_shape",
        "other_shape",
        "condition_shape",
        "mask_shape",
        "min_shape",
        "max_shape",
        "weight_shape",
        "end_shape",
        "dtype",
    ):
        if hasattr(workload, attr):
            params[attr] = getattr(workload, attr)

    input_shape = params.get("input_shape")
    other_shape = params.get("other_shape")
    if input_shape is not None and other_shape is not None:
        output_shape = params.get("shape") or tuple(
            torch.broadcast_shapes(input_shape, other_shape)
        )
        params["output_shape"] = output_shape
        params["broadcast_kind"] = _broadcast_kind(
            input_shape, other_shape, output_shape,
        )
    return params


def _record_unary(
    op,
    bm: ManifestBenchmark,
    inputs: tuple[torch.Tensor, ...],
    baseline_fn: Callable,
) -> None:
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, _manifest_params(bm), result, tag="tileops")
    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, _manifest_params(bm), result_bl, tag="torch")


def _record_binary(
    op,
    bm: ManifestBenchmark,
    inputs: tuple[torch.Tensor, torch.Tensor],
    baseline_fn: Callable,
) -> None:
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, _manifest_params(bm), result, tag="tileops")
    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, _manifest_params(bm), result_bl, tag="torch")


_RELU_OP = "ReluFwdOp"
_GELU_OP = "GeluFwdOp"
_SILU_OP = "SiluFwdOp"
_HARDSWISH_OP = "HardswishFwdOp"
_HARDSIGMOID_OP = "HardsigmoidFwdOp"
_MISH_OP = "MishFwdOp"
_SELU_OP = "SeluFwdOp"
_LEAKY_RELU_OP = "LeakyReluFwdOp"
_ELU_OP = "EluFwdOp"
_HARDTANH_OP = "HardtanhFwdOp"
_SOFTPLUS_OP = "SoftplusFwdOp"
_SIGMOID_OP = "SigmoidFwdOp"
_TANH_OP = "TanhFwdOp"
_CLAMP_SCALAR_OP = "ClampScalarFwdOp"
_NAN_TO_NUM_OP = "NanToNumFwdOp"


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_RELU_OP)))
def test_relu_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = ReluFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_RELU_OP, op, test)
    _record_unary(op, bm, inputs, F.relu)


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_GELU_OP)))
def test_gelu_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = GeluFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_GELU_OP, op, test)
    _record_unary(op, bm, inputs, lambda x: F.gelu(x, approximate="none"))


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_SILU_OP)))
def test_silu_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = SiluFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_SILU_OP, op, test)
    _record_unary(op, bm, inputs, F.silu)


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_HARDSWISH_OP)))
def test_hardswish_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = HardswishFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_HARDSWISH_OP, op, test)
    _record_unary(op, bm, inputs, F.hardswish)


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_HARDSIGMOID_OP)))
def test_hardsigmoid_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = HardsigmoidFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_HARDSIGMOID_OP, op, test)
    _record_unary(op, bm, inputs, F.hardsigmoid)


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_MISH_OP)))
def test_mish_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = MishFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_MISH_OP, op, test)
    _record_unary(op, bm, inputs, F.mish)


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_SELU_OP)))
def test_selu_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = SeluFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_SELU_OP, op, test)
    _record_unary(op, bm, inputs, F.selu)


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_LEAKY_RELU_OP)))
def test_leaky_relu_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = LeakyReluFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_LEAKY_RELU_OP, op, test)
    _record_unary(op, bm, inputs, lambda x: F.leaky_relu(x, 0.01))


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_ELU_OP)))
def test_elu_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = EluFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_ELU_OP, op, test)
    _record_unary(op, bm, inputs, lambda x: F.elu(x, 1.0))


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_HARDTANH_OP)))
def test_hardtanh_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = HardtanhFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_HARDTANH_OP, op, test)
    _record_unary(op, bm, inputs, lambda x: F.hardtanh(x, -1.0, 1.0))


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_SOFTPLUS_OP)))
def test_softplus_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = SoftplusFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_SOFTPLUS_OP, op, test)
    _record_unary(op, bm, inputs, lambda x: F.softplus(x, 1.0, 20.0))


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_SIGMOID_OP), "x_shape"))
def test_sigmoid_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = SigmoidFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_SIGMOID_OP, op, test)
    _record_unary(op, bm, inputs, torch.sigmoid)


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_TANH_OP), "x_shape"))
def test_tanh_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = TanhFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_TANH_OP, op, test)
    _record_unary(op, bm, inputs, torch.tanh)


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_CLAMP_SCALAR_OP)))
def test_clamp_scalar_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = ClampScalarFwdOp(input=shape, dtype=dtype, min=-0.5, max=0.5)
    bm = ManifestBenchmark(_CLAMP_SCALAR_OP, op, test)
    _record_unary(op, bm, inputs, lambda x: torch.clamp(x, -0.5, 0.5))


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_NAN_TO_NUM_OP)))
def test_nan_to_num_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = UnaryManifestWorkload(shape, dtype)
    inputs = test.gen_inputs()
    op = NanToNumFwdOp(N_total=test.n_total, dtype=dtype)
    bm = ManifestBenchmark(_NAN_TO_NUM_OP, op, test)
    _record_unary(op, bm, inputs, torch.nan_to_num)


_PRELU_OP = "PreluFwdOp"


@pytest.mark.parametrize(
    "input_shape, weight_shape, dtype",
    _prelu_params(load_workloads(_PRELU_OP)),
)
def test_prelu_manifest_bench(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    test = PreluManifestWorkload(input_shape, weight_shape, dtype)
    x, weight = test.gen_inputs()
    op = PreluFwdOp(shape=input_shape, dtype=dtype, num_channels=test.num_channels)
    bm = ManifestBenchmark(_PRELU_OP, op, test)
    result = bm.profile(op, x, weight)
    BenchmarkReport.record(op, locals(), result, tag="tileops")
    result_bl = bm.profile(F.prelu, x, weight)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


_MASKED_FILL_OP = "MaskedFillFwdOp"
_MASKED_FILL_SCALAR_OP = "MaskedFillScalarFwdOp"


@pytest.mark.parametrize(
    "input_shape, mask_shape, value_shape, dtype",
    _masked_fill_tensor_params(load_workloads(_MASKED_FILL_OP)),
)
def test_masked_fill_tensor_manifest_bench(
    input_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    test = MaskedFillTensorManifestWorkload(input_shape, mask_shape, value_shape, dtype)
    x, mask, value = test.gen_inputs()
    op = MaskedFillFwdOp(input=input_shape, mask=mask_shape, value=value_shape, dtype=dtype)
    bm = ManifestBenchmark(_MASKED_FILL_OP, op, test)
    result = bm.profile(op, x, mask, value)
    BenchmarkReport.record(op, locals(), result, tag="tileops")
    result_bl = bm.profile(lambda a, m, v: a.masked_fill(m, v), x, mask, value)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


@pytest.mark.parametrize(
    "shape, dtype",
    _shape_dtype_params(load_workloads(_MASKED_FILL_SCALAR_OP)),
)
def test_masked_fill_scalar_manifest_bench(
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    test = MaskedFillScalarManifestWorkload(shape, dtype)
    x, mask = test.gen_inputs()
    op = MaskedFillScalarFwdOp(input=shape, mask=shape, value=-100.0, dtype=dtype)
    bm = ManifestBenchmark(_MASKED_FILL_SCALAR_OP, op, test)
    result = bm.profile(op, x, mask)
    BenchmarkReport.record(op, locals(), result, tag="tileops")
    result_bl = bm.profile(lambda a, m: a.masked_fill(m, -100.0), x, mask)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


_ADD_OP = "AddFwdOp"
_SUB_OP = "SubFwdOp"
_MUL_OP = "MulFwdOp"
_DIV_OP = "DivFwdOp"
_REMAINDER_OP = "RemainderFwdOp"
_POW_OP = "PowFwdOp"
_FLOOR_DIVIDE_OP = "FloorDivideFwdOp"
_LERP_OP = "LerpFwdOp"
_MAXIMUM_OP = "MaximumFwdOp"
_MINIMUM_OP = "MinimumFwdOp"


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_ADD_OP)))
def test_add_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = AddFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_ADD_OP, op, test)
    _record_binary(op, bm, inputs, torch.add)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_SUB_OP)))
def test_sub_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = SubFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_SUB_OP, op, test)
    _record_binary(op, bm, inputs, torch.sub)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_MUL_OP)))
def test_mul_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = MulFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_MUL_OP, op, test)
    _record_binary(op, bm, inputs, torch.mul)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_DIV_OP)))
def test_div_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, positive=True)
    inputs = test.gen_inputs()
    op = DivFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_DIV_OP, op, test)
    _record_binary(op, bm, inputs, torch.div)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    _binary_params(load_workloads(_REMAINDER_OP)),
)
def test_remainder_manifest_bench(
    input_shape: tuple,
    other_shape: tuple,
    dtype: torch.dtype,
) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, positive=True)
    inputs = test.gen_inputs()
    op = RemainderFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_REMAINDER_OP, op, test)
    _record_binary(op, bm, inputs, torch.remainder)


@pytest.mark.parametrize(
    "input_shape, exponent_shape, dtype",
    _binary_params(load_workloads(_POW_OP), "exponent_shape"),
)
def test_pow_manifest_bench(
    input_shape: tuple,
    exponent_shape: tuple,
    dtype: torch.dtype,
) -> None:
    test = BinaryManifestWorkload(input_shape, exponent_shape, dtype, positive=True)
    inputs = test.gen_inputs()
    op = PowFwdOp(a_shape=input_shape, b_shape=exponent_shape, dtype=dtype)
    bm = ManifestBenchmark(_POW_OP, op, test)
    _record_binary(op, bm, inputs, torch.pow)


@pytest.mark.parametrize(
    "input_shape, other_shape, dtype",
    _binary_params(load_workloads(_FLOOR_DIVIDE_OP)),
)
def test_floor_divide_manifest_bench(
    input_shape: tuple,
    other_shape: tuple,
    dtype: torch.dtype,
) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, positive=True)
    inputs = test.gen_inputs()
    op = FloorDivideFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_FLOOR_DIVIDE_OP, op, test)
    _record_binary(op, bm, inputs, torch.floor_divide)


@pytest.mark.parametrize("input_shape, end_shape, dtype", _binary_params(load_workloads(_LERP_OP), "end_shape"))
def test_lerp_manifest_bench(input_shape: tuple, end_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, end_shape, dtype)
    inputs = test.gen_inputs()
    op = LerpFwdOp(a_shape=input_shape, b_shape=end_shape, dtype=dtype)
    bm = ManifestBenchmark(_LERP_OP, op, test)
    _record_binary(op, bm, inputs, lambda a, b: torch.lerp(a, b, 0.5))


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_MAXIMUM_OP)))
def test_maximum_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = MaximumFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_MAXIMUM_OP, op, test)
    _record_binary(op, bm, inputs, torch.maximum)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_MINIMUM_OP)))
def test_minimum_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = MinimumFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_MINIMUM_OP, op, test)
    _record_binary(op, bm, inputs, torch.minimum)


_EQ_OP = "EqFwdOp"
_NE_OP = "NeFwdOp"
_GT_OP = "GtFwdOp"
_LT_OP = "LtFwdOp"
_GE_OP = "GeFwdOp"
_LE_OP = "LeFwdOp"
_LOGICAL_AND_OP = "LogicalAndFwdOp"
_LOGICAL_OR_OP = "LogicalOrFwdOp"
_BITWISE_AND_OP = "BitwiseAndFwdOp"
_BITWISE_OR_OP = "BitwiseOrFwdOp"
_BITWISE_XOR_OP = "BitwiseXorFwdOp"


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_EQ_OP)))
def test_eq_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = EqFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_EQ_OP, op, test)
    _record_binary(op, bm, inputs, torch.eq)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_NE_OP)))
def test_ne_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = NeFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_NE_OP, op, test)
    _record_binary(op, bm, inputs, torch.ne)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_GT_OP)))
def test_gt_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = GtFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_GT_OP, op, test)
    _record_binary(op, bm, inputs, torch.gt)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_LT_OP)))
def test_lt_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = LtFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_LT_OP, op, test)
    _record_binary(op, bm, inputs, torch.lt)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_GE_OP)))
def test_ge_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = GeFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_GE_OP, op, test)
    _record_binary(op, bm, inputs, torch.ge)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_LE_OP)))
def test_le_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype)
    inputs = test.gen_inputs()
    op = LeFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_LE_OP, op, test)
    _record_binary(op, bm, inputs, torch.le)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_LOGICAL_AND_OP)))
def test_logical_and_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, logical=True)
    inputs = test.gen_inputs()
    op = LogicalAndFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_LOGICAL_AND_OP, op, test)
    _record_binary(op, bm, inputs, torch.logical_and)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_LOGICAL_OR_OP)))
def test_logical_or_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, logical=True)
    inputs = test.gen_inputs()
    op = LogicalOrFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_LOGICAL_OR_OP, op, test)
    _record_binary(op, bm, inputs, torch.logical_or)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_BITWISE_AND_OP)))
def test_bitwise_and_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, integer=True)
    inputs = test.gen_inputs()
    op = BitwiseAndFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_BITWISE_AND_OP, op, test)
    _record_binary(op, bm, inputs, torch.bitwise_and)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_BITWISE_OR_OP)))
def test_bitwise_or_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, integer=True)
    inputs = test.gen_inputs()
    op = BitwiseOrFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_BITWISE_OR_OP, op, test)
    _record_binary(op, bm, inputs, torch.bitwise_or)


@pytest.mark.parametrize("input_shape, other_shape, dtype", _binary_params(load_workloads(_BITWISE_XOR_OP)))
def test_bitwise_xor_manifest_bench(input_shape: tuple, other_shape: tuple, dtype: torch.dtype) -> None:
    test = BinaryManifestWorkload(input_shape, other_shape, dtype, integer=True)
    inputs = test.gen_inputs()
    op = BitwiseXorFwdOp(a_shape=input_shape, b_shape=other_shape, dtype=dtype)
    bm = ManifestBenchmark(_BITWISE_XOR_OP, op, test)
    _record_binary(op, bm, inputs, torch.bitwise_xor)


_WHERE_OP = "WhereFwdOp"
_LERP_TENSOR_OP = "LerpTensorFwdOp"


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_WHERE_OP)))
def test_where_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = WhereManifestWorkload(shape, dtype)
    cond, x, other = test.gen_inputs()
    op = WhereFwdOp(condition=shape, input=shape, other=shape, dtype=dtype)
    bm = ManifestBenchmark(_WHERE_OP, op, test)
    result = bm.profile(op, cond, x, other)
    BenchmarkReport.record(op, locals(), result, tag="tileops")
    result_bl = bm.profile(torch.where, cond, x, other)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_LERP_TENSOR_OP)))
def test_lerp_tensor_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = LerpTensorManifestWorkload(shape, dtype)
    x, end, weight = test.gen_inputs()
    op = LerpTensorFwdOp(input=shape, end=shape, weight=shape, dtype=dtype)
    bm = ManifestBenchmark(_LERP_TENSOR_OP, op, test)
    result = bm.profile(op, x, end, weight)
    BenchmarkReport.record(op, locals(), result, tag="tileops")
    result_bl = bm.profile(torch.lerp, x, end, weight)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
