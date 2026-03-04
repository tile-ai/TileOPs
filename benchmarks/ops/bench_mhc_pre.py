from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_mhc_pre import MhcPreFixture, MhcPreTest
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


@MhcPreFixture
def test_mhc_pre_bench(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                       tune: bool) -> None:
    test = MhcPreTest(batch, n_expand, c_x, dtype)
    bm = MhcPreBenchmark(test)
    inputs = test.gen_inputs()

    op = ManifoldConstrainedHyperConnectionPreOp(batch, n_expand, c_x, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("mhc_pre", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("mhc_pre", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
