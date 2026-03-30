import math
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_group_norm import GroupNormTest
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.norm.group_norm import GroupNormOp

_OP_NAME = "groupnorm_fwd"


class GroupNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        spatial_size = math.prod(t.spatial)
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        flops, _ = eval_roofline(_OP_NAME, N=t.n, C=t.c,
                                 spatial_size=spatial_size, elem_bytes=elem_bytes)
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        spatial_size = math.prod(t.spatial)
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        _, mem_bytes = eval_roofline(_OP_NAME, N=t.n, C=t.c,
                                     spatial_size=spatial_size, elem_bytes=elem_bytes)
        return mem_bytes


def _manifest_params():
    params = []
    for w in load_workloads(_OP_NAME):
        shape = w["x_shape"]
        n, c, spatial = shape[0], shape[1], tuple(shape[2:])
        g = w["groups"]
        label = w.get("label", f"{n}x{c}x{'x'.join(map(str, spatial))}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(n, c, spatial, g, dtype, True,
                                       id=f"{label}-{dtype_str}"))
    return params


@pytest.mark.parametrize("n, c, spatial, g, dtype, tune", _manifest_params())
def test_group_norm_bench(n: int, c: int, spatial: tuple, g: int,
                          dtype: torch.dtype, tune: bool) -> None:
    test = GroupNormTest(n, c, spatial, g, dtype)
    bm = GroupNormBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupNormOp(N=n, C=c, spatial=spatial, G=g, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: torch.nn.functional.group_norm
    def baseline_fn(x, weight, bias):
        return F.group_norm(x, g, weight=weight, bias=bias, eps=1e-5)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
