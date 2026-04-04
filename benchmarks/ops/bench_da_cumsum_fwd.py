from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.da_cumsum_fwd import DaCumsumFwdOp
from workloads.ops.da_cumsum_fwd import (
    DaCumsumFwdFixture,
    DaCumsumFwdTest,
    da_cumsum_fwd_ref,
)


class DaCumsumFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, c, L, h = t.batch, t.num_chunks, t.chunk_len, t.n_heads
        # One multiply (dt * A) and one add per element for the inclusive scan
        # Total: 2 * b * c * L * h
        return float(2 * b * c * L * h)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, c, L, h = t.batch, t.num_chunks, t.chunk_len, t.n_heads
        # float32 throughout
        elem = 4
        # Reads: dt (b, c*L, h) + A (h,)
        reads = (b * c * L * h + h) * elem
        # Writes: dA_cumsum (b, h, c, L)
        writes = b * h * c * L * elem
        return float(reads + writes)


@DaCumsumFwdFixture
def test_da_cumsum_fwd_bench(batch, num_chunks, chunk_len, n_heads, tune):
    test = DaCumsumFwdTest(batch, num_chunks, chunk_len, n_heads)
    bm = DaCumsumFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = DaCumsumFwdOp(batch, num_chunks, chunk_len, n_heads, seq_len=num_chunks * chunk_len, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(dt, A):
        return da_cumsum_fwd_ref(dt, A, num_chunks, chunk_len)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record("da_cumsum_fwd", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
