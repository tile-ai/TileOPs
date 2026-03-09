"""Benchmark for InstanceNormOp.

Compares TileOPs InstanceNorm vs PyTorch F.instance_norm on representative shapes.

Run:
    PYTHONPATH="$PWD" python -m pytest benchmarks/ops/bench_instance_norm.py -vvs
"""

import math
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_instance_norm import InstanceNormFixture, InstanceNormTest
from tileops.ops.norm.instance_norm import InstanceNormOp


class InstanceNormBenchmark(BenchmarkBase):

    def __init__(self, test, N, C, spatial):
        super().__init__(test)
        self.N = N
        self.C = C
        self.spatial = spatial

    def calculate_flops(self) -> Optional[float]:
        spatial_size = math.prod(self.spatial) if self.spatial else 1
        total_elements = self.N * self.C * spatial_size
        return 5.0 * total_elements

    def calculate_memory(self) -> Optional[float]:
        spatial_size = math.prod(self.spatial) if self.spatial else 1
        elem_bytes = torch.tensor([], dtype=self.test.dtype).element_size()
        total_elements = self.N * self.C * spatial_size
        return (2 * total_elements + 2 * self.C) * elem_bytes


@InstanceNormFixture
def test_instance_norm_bench(N: int, C: int, spatial: tuple, dtype: torch.dtype) -> None:
    test = InstanceNormTest(N, C, spatial, dtype)
    bm = InstanceNormBenchmark(test, N, C, spatial)
    inputs = test.gen_inputs()

    op = InstanceNormOp(N=N, C=C, spatial=spatial, dtype=dtype)
    result = bm.profile(op, *inputs)
    spatial_str = str(spatial)
    BenchmarkReport.record("instance_norm", {
        "N": N, "C": C, "spatial": spatial_str, "dtype": dtype,
    }, result, tag="tileops")

    # Baseline: torch.nn.functional.instance_norm
    def baseline_fn(x, weight, bias):
        return F.instance_norm(x, weight=weight, bias=bias, eps=1e-5)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record("instance_norm", {
        "N": N, "C": C, "spatial": spatial_str, "dtype": dtype,
    }, result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
