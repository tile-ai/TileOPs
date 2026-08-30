"""Benchmarks for binary/comparison/logical/bitwise/fused-gated elementwise ops.

Profiles TileOPs vs PyTorch baselines for each new op category using
DNN-realistic 2D shapes (tokens × hidden_dim) with the default op configuration.

Every row adds the same reference through inductor. The three fused gated ops add
flashinfer's kernel, which takes the concatenated input the reference splits.
"""

from math import prod
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flashinfer_op,
    reference_tolerance,
)
from benchmarks.benchmark_base import (
    BenchmarkBase,
    ManifestBenchmark,
    OpBenchmark,
    workload_params,
)
from tileops.kernels.elementwise import (
    GeluAndMulFwdKernel,
    GeluTanhAndMulFwdKernel,
    SiluAndMulFwdKernel,
)
from tileops.manifest import load_workloads
from tileops.ops.elementwise import (
    BitwiseAndFwdOp,
    BitwiseOrFwdOp,
    BitwiseXorFwdOp,
    DivFwdOp,
    EqFwdOp,
    FloorDivideFwdOp,
    GeFwdOp,
    GeluAndMulFwdOp,
    GeluTanhAndMulFwdOp,
    GtFwdOp,
    LeFwdOp,
    LerpFwdOp,
    LogicalAndFwdOp,
    LogicalOrFwdOp,
    LtFwdOp,
    MaximumFwdOp,
    MinimumFwdOp,
    MulFwdOp,
    NeFwdOp,
    PowFwdOp,
    RemainderFwdOp,
    SiluAndMulFwdOp,
    SubFwdOp,
)
from workloads.elementwise import (
    BinaryBenchCase,
    BroadcastBenchCase,
    FusedGatedBenchCase,
)
from workloads.workload_base import FixtureBase

# DNN-realistic shapes: (tokens, hidden_dim). The third entry is non-pow2
# (LLaMA-7B intermediate=11008) so each op exercises a non-pow2 shape.
_SHAPES = ((1024, 4096), (1024, 10240), (1024, 11008))


# Workloads


class BinaryBenchmark(OpBenchmark[BinaryBenchCase]):
    """Bandwidth-oriented benchmark for binary elementwise ops."""

    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        in_bytes = t.dtype.itemsize
        out_bytes = t.output_dtype.itemsize
        return t.n_total * (2 * in_bytes + out_bytes)


class FusedGatedBenchmark(BenchmarkBase[FusedGatedBenchCase]):
    """Bandwidth-oriented benchmark for fused gated ops."""

    def calculate_flops(self) -> Optional[float]:
        # activation + multiply: ~2 flops per element
        return 2 * self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem = t.dtype.itemsize
        # Read (M, 2N) + write (M, N)
        return t.n_total * 3 * elem


# Input generators
# Binary arithmetic ops (9)


class BinaryArithBenchFixture(FixtureBase):
    PARAMS = [
        (
            "op_name, shape, dtype, output_dtype, op_cls, baseline_fn, gen_inputs",
            [
                # sub
                pytest.param(
                    "sub",
                    _SHAPES[0],
                    torch.float16,
                    torch.float16,
                    SubFwdOp,
                    torch.sub,
                    "normal",
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "sub",
                    _SHAPES[1],
                    torch.float16,
                    torch.float16,
                    SubFwdOp,
                    torch.sub,
                    "normal",
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "sub",
                    _SHAPES[2],
                    torch.float16,
                    torch.float16,
                    SubFwdOp,
                    torch.sub,
                    "normal",
                    marks=pytest.mark.full,
                ),
                # mul
                pytest.param(
                    "mul",
                    _SHAPES[0],
                    torch.float16,
                    torch.float16,
                    MulFwdOp,
                    torch.mul,
                    "normal",
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "mul",
                    _SHAPES[1],
                    torch.float16,
                    torch.float16,
                    MulFwdOp,
                    torch.mul,
                    "normal",
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "mul",
                    _SHAPES[2],
                    torch.float16,
                    torch.float16,
                    MulFwdOp,
                    torch.mul,
                    "normal",
                    marks=pytest.mark.full,
                ),
                # div
                pytest.param(
                    "div",
                    _SHAPES[0],
                    torch.float16,
                    torch.float16,
                    DivFwdOp,
                    torch.div,
                    "positive",
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "div",
                    _SHAPES[1],
                    torch.float16,
                    torch.float16,
                    DivFwdOp,
                    torch.div,
                    "positive",
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "div",
                    _SHAPES[2],
                    torch.float16,
                    torch.float16,
                    DivFwdOp,
                    torch.div,
                    "positive",
                    marks=pytest.mark.full,
                ),
                # remainder
                pytest.param(
                    "remainder",
                    _SHAPES[0],
                    torch.float16,
                    torch.float16,
                    RemainderFwdOp,
                    torch.remainder,
                    "positive",
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "remainder",
                    _SHAPES[1],
                    torch.float16,
                    torch.float16,
                    RemainderFwdOp,
                    torch.remainder,
                    "positive",
                    marks=pytest.mark.full,
                ),
                # pow
                pytest.param(
                    "pow",
                    _SHAPES[0],
                    torch.float16,
                    torch.float16,
                    PowFwdOp,
                    torch.pow,
                    "positive",
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "pow",
                    _SHAPES[1],
                    torch.float16,
                    torch.float16,
                    PowFwdOp,
                    torch.pow,
                    "positive",
                    marks=pytest.mark.full,
                ),
                # floor_divide
                pytest.param(
                    "floor_divide",
                    _SHAPES[0],
                    torch.float16,
                    torch.float16,
                    FloorDivideFwdOp,
                    torch.floor_divide,
                    "positive",
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "floor_divide",
                    _SHAPES[1],
                    torch.float16,
                    torch.float16,
                    FloorDivideFwdOp,
                    torch.floor_divide,
                    "positive",
                    marks=pytest.mark.full,
                ),
                # lerp (weight=0.5 default)
                pytest.param(
                    "lerp",
                    _SHAPES[0],
                    torch.float16,
                    torch.float16,
                    LerpFwdOp,
                    lambda a, b: torch.lerp(a, b, 0.5),
                    "normal",
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "lerp",
                    _SHAPES[1],
                    torch.float16,
                    torch.float16,
                    LerpFwdOp,
                    lambda a, b: torch.lerp(a, b, 0.5),
                    "normal",
                    marks=pytest.mark.full,
                ),
                # maximum
                pytest.param(
                    "maximum",
                    _SHAPES[0],
                    torch.float16,
                    torch.float16,
                    MaximumFwdOp,
                    torch.maximum,
                    "normal",
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "maximum",
                    _SHAPES[1],
                    torch.float16,
                    torch.float16,
                    MaximumFwdOp,
                    torch.maximum,
                    "normal",
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "maximum",
                    _SHAPES[2],
                    torch.float16,
                    torch.float16,
                    MaximumFwdOp,
                    torch.maximum,
                    "normal",
                    marks=pytest.mark.full,
                ),
                # minimum
                pytest.param(
                    "minimum",
                    _SHAPES[0],
                    torch.float16,
                    torch.float16,
                    MinimumFwdOp,
                    torch.minimum,
                    "normal",
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "minimum",
                    _SHAPES[1],
                    torch.float16,
                    torch.float16,
                    MinimumFwdOp,
                    torch.minimum,
                    "normal",
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "minimum",
                    _SHAPES[2],
                    torch.float16,
                    torch.float16,
                    MinimumFwdOp,
                    torch.minimum,
                    "normal",
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@BinaryArithBenchFixture
def test_binary_arith_bench(
    op_name: str,
    shape: tuple,
    dtype: torch.dtype,
    output_dtype: torch.dtype,
    op_cls,
    baseline_fn,
    gen_inputs,
) -> None:
    test = BinaryBenchCase(shape, dtype, output_dtype, gen_inputs)
    inputs = test.gen_inputs()

    op = op_cls()
    bm = BinaryBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
        params=locals(),
    )


# Comparison ops (6)


class ComparisonBenchFixture(FixtureBase):
    PARAMS = [
        (
            "op_name, shape, dtype, baseline_fn",
            [
                pytest.param("eq", _SHAPES[0], torch.float16, torch.eq, marks=pytest.mark.smoke),
                pytest.param("eq", _SHAPES[1], torch.float16, torch.eq, marks=pytest.mark.full),
                pytest.param("ne", _SHAPES[0], torch.float16, torch.ne, marks=pytest.mark.full),
                pytest.param("gt", _SHAPES[0], torch.float16, torch.gt, marks=pytest.mark.full),
                pytest.param("lt", _SHAPES[0], torch.float16, torch.lt, marks=pytest.mark.full),
                pytest.param("ge", _SHAPES[0], torch.float16, torch.ge, marks=pytest.mark.full),
                pytest.param("le", _SHAPES[0], torch.float16, torch.le, marks=pytest.mark.full),
            ],
        ),
    ]


_CMP_OPS = {
    "eq": EqFwdOp,
    "ne": NeFwdOp,
    "gt": GtFwdOp,
    "lt": LtFwdOp,
    "ge": GeFwdOp,
    "le": LeFwdOp,
}


@ComparisonBenchFixture
def test_comparison_bench(
    op_name: str,
    shape: tuple,
    dtype: torch.dtype,
    baseline_fn,
) -> None:
    test = BinaryBenchCase(shape, dtype, torch.bool, "normal")
    inputs = test.gen_inputs()

    op = _CMP_OPS[op_name]()
    bm = BinaryBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
        params=locals(),
    )


# Logical ops (2)


class LogicalBenchFixture(FixtureBase):
    PARAMS = [
        (
            "op_name, shape, dtype, op_cls, baseline_fn",
            [
                pytest.param(
                    "logical_and",
                    _SHAPES[0],
                    torch.float16,
                    LogicalAndFwdOp,
                    torch.logical_and,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "logical_and",
                    _SHAPES[1],
                    torch.float16,
                    LogicalAndFwdOp,
                    torch.logical_and,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "logical_or",
                    _SHAPES[0],
                    torch.float16,
                    LogicalOrFwdOp,
                    torch.logical_or,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "logical_or",
                    _SHAPES[1],
                    torch.float16,
                    LogicalOrFwdOp,
                    torch.logical_or,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@LogicalBenchFixture
def test_logical_bench(
    op_name: str,
    shape: tuple,
    dtype: torch.dtype,
    op_cls,
    baseline_fn,
) -> None:
    test = BinaryBenchCase(shape, dtype, torch.bool, "bool")
    inputs = test.gen_inputs()

    op = op_cls()
    bm = BinaryBenchmark(op, test)
    # The baseline takes the tensors the row declares, as the op does: ``bool`` copies
    # read one byte per element against the op's two, both credited with two.
    functors = {
        "tileops": op,
        "torch": baseline_fn,
        TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
    }
    bm.compare(functors, *inputs, params=locals())


# Bitwise ops (3)


class BitwiseBenchFixture(FixtureBase):
    PARAMS = [
        (
            "op_name, shape, op_cls, baseline_fn",
            [
                pytest.param(
                    "bitwise_and",
                    _SHAPES[0],
                    BitwiseAndFwdOp,
                    torch.bitwise_and,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "bitwise_and",
                    _SHAPES[1],
                    BitwiseAndFwdOp,
                    torch.bitwise_and,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "bitwise_or",
                    _SHAPES[0],
                    BitwiseOrFwdOp,
                    torch.bitwise_or,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "bitwise_xor",
                    _SHAPES[0],
                    BitwiseXorFwdOp,
                    torch.bitwise_xor,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@BitwiseBenchFixture
def test_bitwise_bench(
    op_name: str,
    shape: tuple,
    op_cls,
    baseline_fn,
) -> None:
    dtype = torch.int32
    test = BinaryBenchCase(shape, dtype, dtype, "int")
    inputs = test.gen_inputs()

    op = op_cls()
    bm = BinaryBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
        params=locals(),
    )


# Fused gated ops (2)


_SILU_AND_MUL_OP = "SiluAndMulFwdOp"
_GELU_AND_MUL_OP = "GeluAndMulFwdOp"
_GELU_TANH_AND_MUL_OP = "GeluTanhAndMulFwdOp"


def _fused_gated_args(w: dict, dtype: torch.dtype) -> tuple:
    """``(M, N, dtype)``; the x_shape trailing axis is 2*N."""
    m, two_n = w["x_shape"]
    return (m, two_n // 2, dtype)


class SiluAndMulBenchFixture(FixtureBase):
    PARAMS = [
        (
            "M, N, dtype",
            workload_params(load_workloads(_SILU_AND_MUL_OP), _fused_gated_args, smoke_first=True),
        )
    ]


class GeluAndMulBenchFixture(FixtureBase):
    PARAMS = [
        (
            "M, N, dtype",
            workload_params(load_workloads(_GELU_AND_MUL_OP), _fused_gated_args, smoke_first=True),
        )
    ]


class GeluTanhAndMulBenchFixture(FixtureBase):
    PARAMS = [
        (
            "M, N, dtype",
            workload_params(
                load_workloads(_GELU_TANH_AND_MUL_OP),
                _fused_gated_args,
                smoke_first=True,
            ),
        )
    ]


def _silu_and_mul_baseline(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return F.silu(x[..., :half]) * x[..., half:]


def _gelu_and_mul_baseline(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return F.gelu(x[..., :half]) * x[..., half:]


def _gelu_tanh_and_mul_baseline(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return F.gelu(x[..., :half], approximate="tanh") * x[..., half:]


# flashinfer names its fused gated kernels after the same three activations and
# takes the same concatenated input, so a key here doubles as its entry-point name.
_FUSED_BASELINES = {
    "silu_and_mul": _silu_and_mul_baseline,
    "gelu_and_mul": _gelu_and_mul_baseline,
    "gelu_tanh_and_mul": _gelu_tanh_and_mul_baseline,
}


def _profile_fused_gated(bm: ManifestBenchmark, op, test, baseline_key: str, params: dict) -> None:
    inputs = test.gen_inputs()
    baseline_fn = _FUSED_BASELINES[baseline_key]
    flashinfer_fn = flashinfer_op(baseline_key)
    assert_matches_reference(
        flashinfer_fn, baseline_fn, *inputs, **reference_tolerance(params["dtype"])
    )
    bm.compare(
        {
            "tileops": op,
            FLASHINFER_TAG: flashinfer_fn,
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
        params=params,
    )


@SiluAndMulBenchFixture
def test_silu_and_mul_bench(M: int, N: int, dtype: torch.dtype) -> None:
    test = FusedGatedBenchCase(M, N, dtype)
    op = SiluAndMulFwdOp()
    bm = ManifestBenchmark(op, test)
    _profile_fused_gated(bm, op, test, "silu_and_mul", {"M": M, "N": N, "dtype": dtype})


@GeluAndMulBenchFixture
def test_gelu_and_mul_bench(M: int, N: int, dtype: torch.dtype) -> None:
    test = FusedGatedBenchCase(M, N, dtype)
    op = GeluAndMulFwdOp()
    bm = ManifestBenchmark(op, test)
    _profile_fused_gated(bm, op, test, "gelu_and_mul", {"M": M, "N": N, "dtype": dtype})


@GeluTanhAndMulBenchFixture
def test_gelu_tanh_and_mul_bench(M: int, N: int, dtype: torch.dtype) -> None:
    test = FusedGatedBenchCase(M, N, dtype)
    op = GeluTanhAndMulFwdOp()
    bm = ManifestBenchmark(op, test)
    _profile_fused_gated(bm, op, test, "gelu_tanh_and_mul", {"M": M, "N": N, "dtype": dtype})


# Fused gated strategy benchmark (direct vs explicit_parallel)


_STRATEGY_SHAPES = [(1024, 4096), (1024, 11008), (4096, 4096)]
_STRATEGY_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_STRATEGY_KERNELS = [
    ("silu_and_mul", SiluAndMulFwdKernel),
    ("gelu_and_mul", GeluAndMulFwdKernel),
    ("gelu_tanh_and_mul", GeluTanhAndMulFwdKernel),
]


def _strategy_params():
    """Default-strategy sentinel: shape and dtype axes on the first kernel, plus
    one reference-point direct-vs-explicit sentinel per remaining kernel.

    The three ops share the fused-gated wrapper but bind different activation
    bodies, whose instruction and register cost can flip the direct-vs-explicit
    result — so each kernel keeps a sentinel, without re-sweeping shapes.
    """
    (sweep_op, sweep_cls), sentinels = _STRATEGY_KERNELS[0], _STRATEGY_KERNELS[1:]
    ref_shape, ref_dtype = _STRATEGY_SHAPES[0], torch.float16
    params = []
    for M, N in _STRATEGY_SHAPES:
        mark = pytest.mark.smoke if ref_shape == (M, N) else pytest.mark.full
        params.append(pytest.param(sweep_op, M, N, ref_dtype, sweep_cls, marks=mark))
    for dtype in _STRATEGY_DTYPES[1:]:
        params.append(pytest.param(sweep_op, *ref_shape, dtype, sweep_cls, marks=pytest.mark.full))
    for op_name, kernel_cls in sentinels:
        params.append(
            pytest.param(op_name, *ref_shape, ref_dtype, kernel_cls, marks=pytest.mark.full)
        )
    return params


class FusedGatedStrategyBenchFixture(FixtureBase):
    PARAMS = [("op_name, M, N, dtype, kernel_cls", _strategy_params())]


# How far behind the fastest strategy the default may sit before the choice is
# stale. Measured at 1024x4096 fp16, the two are 8.4us and 16.2us apart, so this
# margin flags a flip without firing on run-to-run spread.
_STRATEGY_MARGIN = 1.25


@FusedGatedStrategyBenchFixture
def test_fused_gated_default_strategy_is_the_fast_one(
    op_name: str,
    M: int,
    N: int,
    dtype: torch.dtype,
    kernel_cls,
) -> None:
    """The kernel's DEFAULT_STRATEGY is the one that runs fastest here.

    A decision, not a tracked number: it publishes no row, because the report's
    rows are ops and a forced strategy is not one — the Op layer has no way to
    ask for it.
    """
    test = FusedGatedBenchCase(M, N, dtype)
    bm = FusedGatedBenchmark(test)
    inputs = test.gen_inputs()

    timings = {}
    for strategy in ("direct", "explicit_parallel"):
        kernel = kernel_cls(M=M, N=N, dtype=dtype, config={"strategy": strategy})
        timings[strategy] = bm.profile(kernel, *inputs)["device_busy_ms"]

    default = kernel_cls.DEFAULT_STRATEGY
    fastest = min(timings, key=timings.get)
    assert timings[default] <= timings[fastest] * _STRATEGY_MARGIN, (
        f"{kernel_cls.__name__} {M}x{N} {dtype}: DEFAULT_STRATEGY is {default} at "
        f"{timings[default] * 1e3:.2f}us, but {fastest} runs at "
        f"{timings[fastest] * 1e3:.2f}us"
    )


# Broadcast benchmark (bias-add pattern)

# DNN bias-add: (tokens, hidden_dim) + (1, hidden_dim). Includes a non-pow2
# hidden (LLaMA-7B intermediate=11008) to exercise tail handling.
_BROADCAST_SHAPES = [
    ((1024, 4096), (1, 4096)),
    ((1024, 10240), (1, 10240)),
    ((1024, 11008), (1, 11008)),
]


class BroadcastBenchmark(OpBenchmark[BroadcastBenchCase]):
    """Bandwidth-oriented benchmark for broadcast binary ops."""

    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem = t.dtype.itemsize
        out_elem = t.output_dtype.itemsize
        # Read a + read b (smaller, broadcast) + write output
        return (prod(t.a_shape) + prod(t.b_shape)) * elem + t.n_total * out_elem


class BroadcastBenchFixture(FixtureBase):
    PARAMS = [
        (
            "op_name, a_shape, b_shape, dtype, op_cls, baseline_fn, gen_inputs",
            [
                # sub — bias-add pattern
                pytest.param(
                    "sub",
                    *_BROADCAST_SHAPES[0],
                    torch.float16,
                    SubFwdOp,
                    torch.sub,
                    "normal",
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    "sub",
                    *_BROADCAST_SHAPES[1],
                    torch.float16,
                    SubFwdOp,
                    torch.sub,
                    "normal",
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "sub",
                    *_BROADCAST_SHAPES[2],
                    torch.float16,
                    SubFwdOp,
                    torch.sub,
                    "normal",
                    marks=pytest.mark.full,
                ),
                # mul — bias-add pattern
                pytest.param(
                    "mul",
                    *_BROADCAST_SHAPES[0],
                    torch.float16,
                    MulFwdOp,
                    torch.mul,
                    "normal",
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "mul",
                    *_BROADCAST_SHAPES[1],
                    torch.float16,
                    MulFwdOp,
                    torch.mul,
                    "normal",
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "mul",
                    *_BROADCAST_SHAPES[2],
                    torch.float16,
                    MulFwdOp,
                    torch.mul,
                    "normal",
                    marks=pytest.mark.full,
                ),
                # div — bias-add pattern
                pytest.param(
                    "div",
                    *_BROADCAST_SHAPES[0],
                    torch.float16,
                    DivFwdOp,
                    torch.div,
                    "positive",
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "div",
                    *_BROADCAST_SHAPES[1],
                    torch.float16,
                    DivFwdOp,
                    torch.div,
                    "positive",
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    "div",
                    *_BROADCAST_SHAPES[2],
                    torch.float16,
                    DivFwdOp,
                    torch.div,
                    "positive",
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@BroadcastBenchFixture
def test_broadcast_bench(
    op_name: str,
    a_shape: tuple,
    b_shape: tuple,
    dtype: torch.dtype,
    op_cls,
    baseline_fn,
    gen_inputs,
) -> None:
    test = BroadcastBenchCase(a_shape, b_shape, dtype, dtype, gen_inputs)
    inputs = test.gen_inputs()

    op = op_cls()
    bm = BroadcastBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
        params=locals(),
    )
