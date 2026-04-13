"""Benchmarks for representative unary elementwise ops.

Profiles TileOPs vs PyTorch baselines for each new elementwise category using
small, medium, and large 1D shapes with the default op configuration.
"""

from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport, BenchmarkWorkload
from tileops.ops.elementwise import (
    BitwiseNotOp,
    ExpOp,
    GeluOp,
    IsnanOp,
    LogicalNotOp,
)
from workloads.workload_base import FixtureBase

_SHAPES = (262_144, 1_048_576, 4_000_000)


class UnaryElementwiseBenchCase:
    """Minimal test harness shared by BenchmarkBase."""

    def __init__(
        self,
        n_total: int,
        dtype: torch.dtype,
        output_dtype: torch.dtype,
        gen_inputs: Callable[[int, torch.dtype], tuple[torch.Tensor]],
    ):
        self.n_total = n_total
        self.dtype = dtype
        self.output_dtype = output_dtype
        self._gen_inputs = gen_inputs

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return self._gen_inputs(self.n_total, self.dtype)


class UnaryElementwiseBenchmark(BenchmarkBase[BenchmarkWorkload]):
    """Bandwidth-oriented benchmark for unary elementwise ops."""

    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        return self.workload.n_total * (
            self.workload.dtype.itemsize + self.workload.output_dtype.itemsize
        )


def _randn(n_total: int, dtype: torch.dtype) -> tuple[torch.Tensor]:
    return (torch.randn(n_total, device="cuda", dtype=dtype),)


def _logical_inputs(n_total: int, dtype: torch.dtype) -> tuple[torch.Tensor]:
    x = torch.randn(n_total, device="cuda", dtype=dtype)
    mask = torch.rand(n_total, device="cuda") > 0.5
    x[mask] = 0
    return (x,)


def _bitwise_inputs(n_total: int, dtype: torch.dtype) -> tuple[torch.Tensor]:
    x = torch.randint(-128, 128, (n_total,), device="cuda", dtype=dtype)
    return (x,)


def _special_inputs(n_total: int, dtype: torch.dtype) -> tuple[torch.Tensor]:
    x = torch.randn(n_total, device="cuda", dtype=dtype)
    quarter = n_total // 4
    x[:quarter] = float("nan")
    x[quarter:2 * quarter] = float("inf")
    x[2 * quarter:3 * quarter] = float("-inf")
    return (x,)


class UnaryElementwiseBenchFixture(FixtureBase):
    PARAMS = [
        ("op_name, n_total, dtype, output_dtype, op_cls, baseline_fn, gen_inputs", [
            pytest.param(
                "exp", _SHAPES[0], torch.float16, torch.float16,
                ExpOp, torch.exp, _randn,
            ),
            pytest.param(
                "exp", _SHAPES[1], torch.float16, torch.float16,
                ExpOp, torch.exp, _randn,
            ),
            pytest.param(
                "exp", _SHAPES[2], torch.float16, torch.float16,
                ExpOp, torch.exp, _randn,
            ),
            pytest.param(
                "exp", _SHAPES[0], torch.bfloat16, torch.bfloat16,
                ExpOp, torch.exp, _randn,
            ),
            pytest.param(
                "exp", _SHAPES[1], torch.bfloat16, torch.bfloat16,
                ExpOp, torch.exp, _randn,
            ),
            pytest.param(
                "exp", _SHAPES[2], torch.bfloat16, torch.bfloat16,
                ExpOp, torch.exp, _randn,
            ),
            pytest.param(
                "gelu", _SHAPES[0], torch.float16, torch.float16,
                GeluOp, F.gelu, _randn,
            ),
            pytest.param(
                "gelu", _SHAPES[1], torch.float16, torch.float16,
                GeluOp, F.gelu, _randn,
            ),
            pytest.param(
                "gelu", _SHAPES[2], torch.float16, torch.float16,
                GeluOp, F.gelu, _randn,
            ),
            pytest.param(
                "gelu", _SHAPES[0], torch.bfloat16, torch.bfloat16,
                GeluOp, F.gelu, _randn,
            ),
            pytest.param(
                "gelu", _SHAPES[1], torch.bfloat16, torch.bfloat16,
                GeluOp, F.gelu, _randn,
            ),
            pytest.param(
                "gelu", _SHAPES[2], torch.bfloat16, torch.bfloat16,
                GeluOp, F.gelu, _randn,
            ),
            pytest.param(
                "logical_not", _SHAPES[0], torch.float16, torch.bool,
                LogicalNotOp, torch.logical_not, _logical_inputs,
            ),
            pytest.param(
                "logical_not", _SHAPES[1], torch.float16, torch.bool,
                LogicalNotOp, torch.logical_not, _logical_inputs,
            ),
            pytest.param(
                "logical_not", _SHAPES[2], torch.float16, torch.bool,
                LogicalNotOp, torch.logical_not, _logical_inputs,
            ),
            pytest.param(
                "bitwise_not", _SHAPES[0], torch.int32, torch.int32,
                BitwiseNotOp, torch.bitwise_not, _bitwise_inputs,
            ),
            pytest.param(
                "bitwise_not", _SHAPES[1], torch.int32, torch.int32,
                BitwiseNotOp, torch.bitwise_not, _bitwise_inputs,
            ),
            pytest.param(
                "bitwise_not", _SHAPES[2], torch.int32, torch.int32,
                BitwiseNotOp, torch.bitwise_not, _bitwise_inputs,
            ),
            pytest.param(
                "isnan", _SHAPES[0], torch.float16, torch.bool,
                IsnanOp, torch.isnan, _special_inputs,
            ),
            pytest.param(
                "isnan", _SHAPES[1], torch.float16, torch.bool,
                IsnanOp, torch.isnan, _special_inputs,
            ),
            pytest.param(
                "isnan", _SHAPES[2], torch.float16, torch.bool,
                IsnanOp, torch.isnan, _special_inputs,
            ),
        ]),
    ]


@UnaryElementwiseBenchFixture
def test_unary_elementwise_bench(
    op_name: str,
    n_total: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype,
    op_cls,
    baseline_fn,
    gen_inputs,
) -> None:
    test = UnaryElementwiseBenchCase(
        n_total=n_total,
        dtype=dtype,
        output_dtype=output_dtype,
        gen_inputs=gen_inputs,
    )
    bm = UnaryElementwiseBenchmark(test)
    inputs = test.gen_inputs()

    op = op_cls(N_total=n_total, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
