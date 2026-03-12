"""Benchmarks for binary/comparison/logical/bitwise/fused-gated elementwise ops.

Profiles TileOPs vs PyTorch baselines for each new op category using
small, medium, and large 1D shapes with the default op configuration.
"""

from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.test_base import FixtureBase
from tileops.ops.elementwise import (
    BitwiseAndOp,
    BitwiseOrOp,
    BitwiseXorOp,
    DivOp,
    EqOp,
    FloorDivideOp,
    GeluAndMulOp,
    GeluTanhAndMulOp,
    LerpOp,
    LogicalAndOp,
    LogicalOrOp,
    MaximumOp,
    MinimumOp,
    MulOp,
    PowOp,
    RemainderOp,
    SubOp,
)

_SHAPES = (262_144, 1_048_576, 4_000_000)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


class BinaryBenchCase:
    """Minimal test harness for binary ops."""

    def __init__(
        self,
        n_total: int,
        dtype: torch.dtype,
        output_dtype: torch.dtype,
        gen_inputs: Callable,
    ):
        self.n_total = n_total
        self.dtype = dtype
        self.output_dtype = output_dtype
        self._gen_inputs = gen_inputs

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._gen_inputs(self.n_total, self.dtype)


class BinaryBenchmark(BenchmarkBase):
    """Bandwidth-oriented benchmark for binary elementwise ops."""

    def calculate_flops(self) -> Optional[float]:
        return self.test.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        in_bytes = t.dtype.itemsize
        out_bytes = t.output_dtype.itemsize
        return t.n_total * (2 * in_bytes + out_bytes)


class FusedGatedBenchCase:
    """Minimal test harness for fused gated ops."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype
        self.output_dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(1, 2 * self.n_total, device="cuda", dtype=self.dtype),)


class FusedGatedBenchmark(BenchmarkBase):
    """Bandwidth-oriented benchmark for fused gated ops."""

    def calculate_flops(self) -> Optional[float]:
        # activation + multiply: ~2 flops per element
        return 2 * self.test.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        elem = t.dtype.itemsize
        # Read (M, 2N) + write (M, N)
        return t.n_total * 3 * elem


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------


def _randn_pair(n: int, dtype: torch.dtype):
    a = torch.randn(n, device="cuda", dtype=dtype)
    b = torch.randn(n, device="cuda", dtype=dtype)
    return a, b


def _positive_pair(n: int, dtype: torch.dtype):
    a = torch.rand(n, device="cuda", dtype=dtype) + 0.1
    b = torch.rand(n, device="cuda", dtype=dtype) + 0.1
    return a, b


def _int_pair(n: int, dtype: torch.dtype):
    a = torch.randint(-1000, 1000, (n,), device="cuda", dtype=torch.int32)
    b = torch.randint(-1000, 1000, (n,), device="cuda", dtype=torch.int32)
    return a, b


def _bool_pair(n: int, dtype: torch.dtype):
    a = (torch.randn(n, device="cuda", dtype=dtype) > 0).to(dtype)
    b = (torch.randn(n, device="cuda", dtype=dtype) > 0).to(dtype)
    return a, b


# ---------------------------------------------------------------------------
# Binary arithmetic ops (9)
# ---------------------------------------------------------------------------


class BinaryArithBenchFixture(FixtureBase):
    PARAMS = [
        ("op_name, n_total, dtype, output_dtype, op_cls, baseline_fn, gen_inputs", [
            # sub
            pytest.param("sub", _SHAPES[0], torch.float16, torch.float16, SubOp, torch.sub, _randn_pair, marks=pytest.mark.smoke),
            pytest.param("sub", _SHAPES[1], torch.float16, torch.float16, SubOp, torch.sub, _randn_pair, marks=pytest.mark.full),
            pytest.param("sub", _SHAPES[2], torch.float16, torch.float16, SubOp, torch.sub, _randn_pair, marks=pytest.mark.full),
            # mul
            pytest.param("mul", _SHAPES[0], torch.float16, torch.float16, MulOp, torch.mul, _randn_pair, marks=pytest.mark.smoke),
            pytest.param("mul", _SHAPES[1], torch.float16, torch.float16, MulOp, torch.mul, _randn_pair, marks=pytest.mark.full),
            pytest.param("mul", _SHAPES[2], torch.float16, torch.float16, MulOp, torch.mul, _randn_pair, marks=pytest.mark.full),
            # div
            pytest.param("div", _SHAPES[0], torch.float16, torch.float16, DivOp, torch.div, _positive_pair, marks=pytest.mark.smoke),
            pytest.param("div", _SHAPES[1], torch.float16, torch.float16, DivOp, torch.div, _positive_pair, marks=pytest.mark.full),
            pytest.param("div", _SHAPES[2], torch.float16, torch.float16, DivOp, torch.div, _positive_pair, marks=pytest.mark.full),
            # remainder
            pytest.param("remainder", _SHAPES[0], torch.float16, torch.float16, RemainderOp, torch.remainder, _positive_pair, marks=pytest.mark.smoke),
            pytest.param("remainder", _SHAPES[1], torch.float16, torch.float16, RemainderOp, torch.remainder, _positive_pair, marks=pytest.mark.full),
            # pow
            pytest.param("pow", _SHAPES[0], torch.float16, torch.float16, PowOp, torch.pow, _positive_pair, marks=pytest.mark.smoke),
            pytest.param("pow", _SHAPES[1], torch.float16, torch.float16, PowOp, torch.pow, _positive_pair, marks=pytest.mark.full),
            # floor_divide
            pytest.param("floor_divide", _SHAPES[0], torch.float16, torch.float16, FloorDivideOp, torch.floor_divide, _positive_pair, marks=pytest.mark.smoke),
            pytest.param("floor_divide", _SHAPES[1], torch.float16, torch.float16, FloorDivideOp, torch.floor_divide, _positive_pair, marks=pytest.mark.full),
            # lerp (weight=0.5 default)
            pytest.param("lerp", _SHAPES[0], torch.float16, torch.float16, LerpOp, lambda a, b: torch.lerp(a, b, 0.5), _randn_pair, marks=pytest.mark.smoke),
            pytest.param("lerp", _SHAPES[1], torch.float16, torch.float16, LerpOp, lambda a, b: torch.lerp(a, b, 0.5), _randn_pair, marks=pytest.mark.full),
            # maximum
            pytest.param("maximum", _SHAPES[0], torch.float16, torch.float16, MaximumOp, torch.maximum, _randn_pair, marks=pytest.mark.smoke),
            pytest.param("maximum", _SHAPES[1], torch.float16, torch.float16, MaximumOp, torch.maximum, _randn_pair, marks=pytest.mark.full),
            pytest.param("maximum", _SHAPES[2], torch.float16, torch.float16, MaximumOp, torch.maximum, _randn_pair, marks=pytest.mark.full),
            # minimum
            pytest.param("minimum", _SHAPES[0], torch.float16, torch.float16, MinimumOp, torch.minimum, _randn_pair, marks=pytest.mark.smoke),
            pytest.param("minimum", _SHAPES[1], torch.float16, torch.float16, MinimumOp, torch.minimum, _randn_pair, marks=pytest.mark.full),
            pytest.param("minimum", _SHAPES[2], torch.float16, torch.float16, MinimumOp, torch.minimum, _randn_pair, marks=pytest.mark.full),
        ]),
    ]


@BinaryArithBenchFixture
def test_binary_arith_bench(
    op_name: str,
    n_total: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype,
    op_cls,
    baseline_fn,
    gen_inputs,
) -> None:
    test = BinaryBenchCase(n_total, dtype, output_dtype, gen_inputs)
    bm = BinaryBenchmark(test)
    inputs = test.gen_inputs()

    shape = (n_total,)
    op = op_cls(a_shape=shape, b_shape=shape, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="baseline")


# ---------------------------------------------------------------------------
# Comparison ops (6)
# ---------------------------------------------------------------------------


class ComparisonBenchFixture(FixtureBase):
    PARAMS = [
        ("op_name, n_total, dtype, baseline_fn", [
            pytest.param("eq", _SHAPES[1], torch.float16, torch.eq, marks=pytest.mark.smoke),
            pytest.param("eq", _SHAPES[2], torch.float16, torch.eq, marks=pytest.mark.full),
            pytest.param("ne", _SHAPES[1], torch.float16, torch.ne, marks=pytest.mark.full),
            pytest.param("gt", _SHAPES[1], torch.float16, torch.gt, marks=pytest.mark.full),
            pytest.param("lt", _SHAPES[1], torch.float16, torch.lt, marks=pytest.mark.full),
            pytest.param("ge", _SHAPES[1], torch.float16, torch.ge, marks=pytest.mark.full),
            pytest.param("le", _SHAPES[1], torch.float16, torch.le, marks=pytest.mark.full),
        ]),
    ]


_CMP_OPS = {
    "eq": EqOp, "ne": __import__("tileops.ops.elementwise", fromlist=["NeOp"]).NeOp,
    "gt": __import__("tileops.ops.elementwise", fromlist=["GtOp"]).GtOp,
    "lt": __import__("tileops.ops.elementwise", fromlist=["LtOp"]).LtOp,
    "ge": __import__("tileops.ops.elementwise", fromlist=["GeOp"]).GeOp,
    "le": __import__("tileops.ops.elementwise", fromlist=["LeOp"]).LeOp,
}


@ComparisonBenchFixture
def test_comparison_bench(
    op_name: str,
    n_total: int,
    dtype: torch.dtype,
    baseline_fn,
) -> None:
    test = BinaryBenchCase(n_total, dtype, torch.bool, _randn_pair)
    bm = BinaryBenchmark(test)
    inputs = test.gen_inputs()

    shape = (n_total,)
    op = _CMP_OPS[op_name](a_shape=shape, b_shape=shape, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(f"cmp_{op_name}", locals(), result, tag="tileops")

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(f"cmp_{op_name}", locals(), result_bl, tag="baseline")


# ---------------------------------------------------------------------------
# Logical ops (2)
# ---------------------------------------------------------------------------


class LogicalBenchFixture(FixtureBase):
    PARAMS = [
        ("op_name, n_total, dtype, op_cls, baseline_fn", [
            pytest.param("logical_and", _SHAPES[1], torch.float16, LogicalAndOp, torch.logical_and, marks=pytest.mark.smoke),
            pytest.param("logical_and", _SHAPES[2], torch.float16, LogicalAndOp, torch.logical_and, marks=pytest.mark.full),
            pytest.param("logical_or", _SHAPES[1], torch.float16, LogicalOrOp, torch.logical_or, marks=pytest.mark.smoke),
            pytest.param("logical_or", _SHAPES[2], torch.float16, LogicalOrOp, torch.logical_or, marks=pytest.mark.full),
        ]),
    ]


@LogicalBenchFixture
def test_logical_bench(
    op_name: str,
    n_total: int,
    dtype: torch.dtype,
    op_cls,
    baseline_fn,
) -> None:
    test = BinaryBenchCase(n_total, dtype, torch.bool, _bool_pair)
    bm = BinaryBenchmark(test)
    inputs = test.gen_inputs()

    shape = (n_total,)
    op = op_cls(a_shape=shape, b_shape=shape, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    # Baseline uses bool tensors
    a_bool, b_bool = inputs[0].bool(), inputs[1].bool()
    result_bl = bm.profile(baseline_fn, a_bool, b_bool)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="baseline")


# ---------------------------------------------------------------------------
# Bitwise ops (3)
# ---------------------------------------------------------------------------


class BitwiseBenchFixture(FixtureBase):
    PARAMS = [
        ("op_name, n_total, op_cls, baseline_fn", [
            pytest.param("bitwise_and", _SHAPES[1], BitwiseAndOp, torch.bitwise_and, marks=pytest.mark.smoke),
            pytest.param("bitwise_and", _SHAPES[2], BitwiseAndOp, torch.bitwise_and, marks=pytest.mark.full),
            pytest.param("bitwise_or", _SHAPES[1], BitwiseOrOp, torch.bitwise_or, marks=pytest.mark.full),
            pytest.param("bitwise_xor", _SHAPES[1], BitwiseXorOp, torch.bitwise_xor, marks=pytest.mark.full),
        ]),
    ]


@BitwiseBenchFixture
def test_bitwise_bench(
    op_name: str,
    n_total: int,
    op_cls,
    baseline_fn,
) -> None:
    dtype = torch.int32
    test = BinaryBenchCase(n_total, dtype, dtype, _int_pair)
    bm = BinaryBenchmark(test)
    inputs = test.gen_inputs()

    shape = (n_total,)
    op = op_cls(a_shape=shape, b_shape=shape, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="baseline")


# ---------------------------------------------------------------------------
# Fused gated ops (2)
# ---------------------------------------------------------------------------


class FusedGatedBenchFixture(FixtureBase):
    PARAMS = [
        ("op_name, n_total, dtype, op_cls", [
            pytest.param("gelu_and_mul", _SHAPES[0], torch.float16, GeluAndMulOp, marks=pytest.mark.smoke),
            pytest.param("gelu_and_mul", _SHAPES[1], torch.float16, GeluAndMulOp, marks=pytest.mark.full),
            pytest.param("gelu_and_mul", _SHAPES[2], torch.float16, GeluAndMulOp, marks=pytest.mark.full),
            pytest.param("gelu_tanh_and_mul", _SHAPES[0], torch.float16, GeluTanhAndMulOp, marks=pytest.mark.smoke),
            pytest.param("gelu_tanh_and_mul", _SHAPES[1], torch.float16, GeluTanhAndMulOp, marks=pytest.mark.full),
            pytest.param("gelu_tanh_and_mul", _SHAPES[2], torch.float16, GeluTanhAndMulOp, marks=pytest.mark.full),
        ]),
    ]


def _gelu_and_mul_baseline(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return F.gelu(x[..., :half]) * x[..., half:]


def _gelu_tanh_and_mul_baseline(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return F.gelu(x[..., :half], approximate="tanh") * x[..., half:]


_FUSED_BASELINES = {
    "gelu_and_mul": _gelu_and_mul_baseline,
    "gelu_tanh_and_mul": _gelu_tanh_and_mul_baseline,
}


@FusedGatedBenchFixture
def test_fused_gated_bench(
    op_name: str,
    n_total: int,
    dtype: torch.dtype,
    op_cls,
) -> None:
    test = FusedGatedBenchCase(n_total, dtype)
    bm = FusedGatedBenchmark(test)
    inputs = test.gen_inputs()

    op = op_cls(M=1, N=n_total, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    baseline_fn = _FUSED_BASELINES[op_name]
    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
