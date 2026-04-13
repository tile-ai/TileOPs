"""Benchmarks for DropoutOp.

Profiles TileOPs dropout vs torch.nn.functional.dropout on DNN-realistic shapes.
Uses p=0.5 (default) as representative drop rate.
"""

from math import prod
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport, BenchmarkWorkload
from tileops.ops.dropout import DropoutOp
from workloads.workload_base import FixtureBase

_SHAPES = [(1024, 4096), (1024, 10240), (1024, 20480)]
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_P = 0.5


class DropoutBenchCase:
    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, device="cuda", dtype=self.dtype),)


class DropoutBenchmark(BenchmarkBase[BenchmarkWorkload]):
    def calculate_flops(self) -> Optional[float]:
        # RNG + compare + scale — not compute-bound, return element count
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        # Read x + write y
        return self.workload.n_total * self.workload.dtype.itemsize * 2


def _dropout_params():
    params = []
    for shape in _SHAPES:
        for dtype in _DTYPES:
            mark = pytest.mark.smoke if (shape == _SHAPES[0] and dtype == torch.float16) else pytest.mark.full
            params.append(pytest.param(shape, dtype, marks=mark))
    return params


class DropoutBenchFixture(FixtureBase):
    PARAMS = [("shape, dtype", _dropout_params())]


@DropoutBenchFixture
def test_dropout_bench(shape: tuple, dtype: torch.dtype) -> None:
    n_total = prod(shape)
    test = DropoutBenchCase(shape, dtype)
    bm = DropoutBenchmark(test)
    (x,) = test.gen_inputs()

    op = DropoutOp(N_total=n_total, dtype=dtype, p=_P, seed=42)
    result = bm.profile(op, x)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return F.dropout(x, p=_P, training=True)

    result_bl = bm.profile(baseline_fn, x)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
