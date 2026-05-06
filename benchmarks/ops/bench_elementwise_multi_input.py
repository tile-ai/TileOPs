"""Benchmarks for the elementwise_multi_input family.

Currently scopes ``LerpTensorFwdOp`` (Tensor-weight ``torch.lerp``).
``WhereFwdOp`` is benched in ``bench_independent_elementwise.py`` already.

Shapes follow LLaMA-family defaults: (tokens x hidden_dim) where
hidden_dim covers small=4096, medium=10240, and non-pow2=11008
(LLaMA-7B intermediate). All three manifest-declared dtypes are
exercised.
"""

from math import prod
from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops.elementwise import LerpTensorFwdOp
from workloads.workload_base import FixtureBase

_LERP_SHAPES = [(1024, 4096), (1024, 10240), (1024, 11008)]
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class LerpTensorBenchCase:
    """Same-shape input/end/weight; output broadcast equals the shape."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = torch.randn(self.shape, device="cuda", dtype=self.dtype)
        b = torch.randn(self.shape, device="cuda", dtype=self.dtype)
        # Keep weight in [0, 1] to stay close to typical lerp usage.
        w = torch.rand(self.shape, device="cuda", dtype=self.dtype)
        return a, b, w


class LerpTensorBenchmark(BenchmarkBase[LerpTensorBenchCase]):
    """Bandwidth-oriented benchmark for ``LerpTensorFwdOp``.

    Per output element: 3 flops (sub + mul + add). Bytes: 3 reads +
    1 write at ``elem_bytes`` (matches
    ``tileops.perf.formulas.lerp_tensor_fwd_roofline``).
    """

    def calculate_flops(self) -> Optional[float]:
        return 3 * self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        return 4 * t.n_total * t.dtype.itemsize


def _lerp_tensor_params() -> list:
    params = []
    for shape in _LERP_SHAPES:
        for dtype in _DTYPES:
            mark = (
                pytest.mark.smoke
                if (shape == _LERP_SHAPES[0] and dtype == torch.float16)
                else pytest.mark.full
            )
            params.append(pytest.param(shape, dtype, marks=mark))
    return params


class LerpTensorBenchFixture(FixtureBase):
    PARAMS = [("shape, dtype", _lerp_tensor_params())]


@LerpTensorBenchFixture
def test_lerp_tensor_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = LerpTensorBenchCase(shape, dtype)
    bm = LerpTensorBenchmark(test)
    a, b, w = test.gen_inputs()

    op = LerpTensorFwdOp(
        input=tuple(shape), end=tuple(shape), weight=tuple(shape), dtype=dtype,
    )
    result = bm.profile(op, a, b, w)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(torch.lerp, a, b, w)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")
