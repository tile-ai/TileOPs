import math
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_group_norm import GroupNormTest
from tileops.ops.norm.group_norm import GroupNormOp


class GroupNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        spatial_size = math.prod(t.spatial)
        total_elems = t.n * t.c * spatial_size
        # Per element: subtract mean, square for var, normalize, scale, bias => ~5 flops
        return 5 * total_elems

    def calculate_memory(self) -> Optional[float]:
        """Useful bytes only. Read x + read weight + read bias + write y."""
        t = self.test
        spatial_size = math.prod(t.spatial)
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        total_elems = t.n * t.c * spatial_size
        # Read x + write y + read weight (C, broadcast) + read bias (C, broadcast)
        return (2 * total_elems + 2 * t.c) * elem_bytes


_GROUP_NORM_BENCH_PARAMS = [
    pytest.param(2, 32, (8, 8), 8, torch.float16, True, marks=pytest.mark.full, id="bench-fp16-2d"),
    pytest.param(2, 16, (4, 4, 4), 4, torch.float16, True, marks=pytest.mark.full, id="bench-fp16-3d"),
    pytest.param(2, 32, (7, 7), 8, torch.bfloat16, True, marks=pytest.mark.nightly, id="bench-bf16-unaligned"),
    pytest.param(2, 32, (4, 4), 32, torch.float16, True, marks=pytest.mark.nightly, id="bench-instance-like"),
]


@pytest.mark.parametrize("n, c, spatial, g, dtype, tune", _GROUP_NORM_BENCH_PARAMS)
def test_group_norm_bench(n: int, c: int, spatial: tuple, g: int,
                          dtype: torch.dtype, tune: bool) -> None:
    test = GroupNormTest(n, c, spatial, g, dtype)
    bm = GroupNormBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupNormOp(N=n, C=c, spatial=spatial, G=g, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("group_norm", locals(), result, tag="tileops")

    # Baseline: torch.nn.functional.group_norm
    def baseline_fn(x, weight, bias):
        return F.group_norm(x, g, weight=weight, bias=bias, eps=1e-5)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record("group_norm", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
