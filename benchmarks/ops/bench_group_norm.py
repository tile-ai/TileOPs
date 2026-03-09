"""Benchmark for GroupNormOp.

Compares TileOPs GroupNorm vs PyTorch F.group_norm on representative shapes.

Run:
    PYTHONPATH="$PWD" python -m pytest benchmarks/ops/bench_group_norm.py -vvs
"""

import math
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_group_norm import GroupNormFixture, GroupNormTest
from tileops.ops.norm.group_norm import GroupNormOp


class GroupNormBenchmark(BenchmarkBase):

    def __init__(self, test, N, C, spatial, G):
        super().__init__(test)
        self.N = N
        self.C = C
        self.spatial = spatial
        self.G = G

    def calculate_flops(self) -> Optional[float]:
        spatial_size = math.prod(self.spatial) if self.spatial else 1
        total_elements = self.N * self.C * spatial_size
        # Per element: mean, variance, normalize, scale, bias ~ 5 flops
        return 5.0 * total_elements

    def calculate_memory(self) -> Optional[float]:
        spatial_size = math.prod(self.spatial) if self.spatial else 1
        elem_bytes = torch.tensor([], dtype=self.test.dtype).element_size()
        total_elements = self.N * self.C * spatial_size
        # Read x + write y + read weight (C) + read bias (C)
        return (2 * total_elements + 2 * self.C) * elem_bytes


@GroupNormFixture
def test_group_norm_bench(N: int, C: int, spatial: tuple, G: int, dtype: torch.dtype) -> None:
    test = GroupNormTest(N, C, spatial, G, dtype)
    bm = GroupNormBenchmark(test, N, C, spatial, G)
    inputs = test.gen_inputs()

    op = GroupNormOp(N=N, C=C, spatial=spatial, G=G, dtype=dtype)
    result = bm.profile(op, *inputs)
    spatial_str = str(spatial)
    BenchmarkReport.record("group_norm", {
        "N": N, "C": C, "spatial": spatial_str, "G": G, "dtype": dtype,
    }, result, tag="tileops")

    # Baseline: torch.nn.functional.group_norm
    def baseline_fn(x, weight, bias):
        return F.group_norm(x, G, weight=weight, bias=bias, eps=1e-5)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record("group_norm", {
        "N": N, "C": C, "spatial": spatial_str, "G": G, "dtype": dtype,
    }, result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
