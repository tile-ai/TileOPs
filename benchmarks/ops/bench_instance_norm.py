import math
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_instance_norm import InstanceNormTest
from tileops.ops.norm.instance_norm import InstanceNormOp


class InstanceNormBenchmark(BenchmarkBase):

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


_INSTANCE_NORM_BENCH_PARAMS = [
    pytest.param(8, 128, (32, 32), torch.float16, True, id="image-fp16"),
    pytest.param(8, 128, (32, 32), torch.bfloat16, True, id="image-bf16"),
    pytest.param(4, 256, (28, 28), torch.float16, True, id="wider-channel"),
    pytest.param(4, 64, (30, 30), torch.float16, True, id="tail-spatial"),
]


@pytest.mark.parametrize("n, c, spatial, dtype, tune", _INSTANCE_NORM_BENCH_PARAMS)
def test_instance_norm_bench(n: int, c: int, spatial: tuple,
                             dtype: torch.dtype, tune: bool) -> None:
    test = InstanceNormTest(n, c, spatial, dtype)
    bm = InstanceNormBenchmark(test)
    inputs = test.gen_inputs()

    op = InstanceNormOp(N=n, C=c, spatial=spatial, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: torch.nn.functional.instance_norm
    def baseline_fn(x, weight, bias):
        return F.instance_norm(x, weight=weight, bias=bias, eps=1e-5)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
