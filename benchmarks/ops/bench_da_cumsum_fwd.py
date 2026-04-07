from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.da_cumsum_fwd import DaCumsumFwdOp
from workloads.ops.da_cumsum_fwd import DaCumsumFwdTest


def da_cumsum_fwd_ref(
    dt: torch.Tensor,
    A: torch.Tensor,
    num_chunks: int,
    chunk_len: int,
) -> torch.Tensor:
    """PyTorch reference for da_cumsum_fwd (benchmark-local copy)."""
    b, S, h = dt.shape
    Q = chunk_len
    C = num_chunks
    dt_chunked = dt.float().reshape(b, C, Q, h)
    dA = dt_chunked * A.float()
    dA_cumsum = dA.cumsum(dim=2)
    return dA_cumsum.permute(0, 3, 1, 2).contiguous()


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


_DA_CUMSUM_FWD_BENCH_PARAMS = [
    pytest.param(1, 2,  64,  4, False, id="b1-c2-L64-h4"),
    pytest.param(2, 4,  64,  8, False, id="b2-c4-L64-h8"),
    pytest.param(1, 2, 128,  4, False, id="b1-c2-L128-h4"),
    pytest.param(2, 4, 128, 16, False, id="b2-c4-L128-h16"),
]


@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_heads, tune",
    _DA_CUMSUM_FWD_BENCH_PARAMS,
)
def test_da_cumsum_fwd_bench(
    batch: int, num_chunks: int, chunk_len: int, n_heads: int, tune: bool,
) -> None:
    test = DaCumsumFwdTest(batch, num_chunks, chunk_len, n_heads)
    bm = DaCumsumFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = DaCumsumFwdOp(batch, num_chunks, chunk_len, n_heads, seq_len=num_chunks * chunk_len, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(dt, A):
        return da_cumsum_fwd_ref(dt, A, num_chunks, chunk_len)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
