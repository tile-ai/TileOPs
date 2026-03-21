from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_mhc_pre import MhcPreTest
from tileops.ops import ManifoldConstrainedHyperConnectionPreOp


class MhcPreBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops = 2 * t.batch * (
            (t.n_expand * t.n_expand * t.c_x * t.c_x) *
            (t.n_expand * t.n_expand + 2 * t.n_expand) + t.n_expand * t.c_x)
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return (t.n_expand * 3 + 1) * t.c_x + (t.n_expand * t.c_x) * (
            t.n_expand * t.n_expand + 2 * t.n_expand)


_MHC_PRE_BENCH_PARAMS = [
    pytest.param(1, 4, 1280, torch.bfloat16, True, id="small"),
    pytest.param(2, 4, 1920, torch.bfloat16, True, id="medium"),
    pytest.param(4, 4, 2560, torch.bfloat16, True, id="large"),
]


@pytest.mark.parametrize("batch, n_expand, c_x, dtype, tune", _MHC_PRE_BENCH_PARAMS)
def test_mhc_pre_bench(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                       tune: bool) -> None:
    test = MhcPreTest(batch, n_expand, c_x, dtype)
    bm = MhcPreBenchmark(test)
    inputs = test.gen_inputs()

    op = ManifoldConstrainedHyperConnectionPreOp(batch, n_expand, c_x, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
