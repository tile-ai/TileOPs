from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ssd_chunk_state_fwd import (
    SsdChunkStateFwdFixture,
    SsdChunkStateFwdTest,
    ssd_chunk_state_fwd_ref,
)
from tileops.ops.ssd_chunk_state_fwd import SsdChunkStateFwdOp


class SsdChunkStateFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        b, c, Q, h, p, n = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state,
        )
        # For each (b, c, h) block we do a rank-1 outer-product accumulation
        # over Q positions: Q * (p + n) multiply-adds, giving Q * p * n * 2 FLOPs
        # (treating the outer product as n*p MACs per position).
        flops = b * c * h * Q * n * p * 2
        return float(flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        b, c, Q, h, p, n, g = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state, t.n_groups,
        )
        seq_len = c * Q
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): x + Bmat
        reads = (
            b * seq_len * h * p      # x
            + b * seq_len * g * n    # Bmat
        ) * elem
        # Reads (float32): dt + dA_cumsum
        reads += b * h * c * Q * 4 * 2
        # Writes (float32): out
        writes = b * c * h * n * p * 4
        return float(reads + writes)


@SsdChunkStateFwdFixture
def test_ssd_chunk_state_fwd_bench(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx,
):
    test = SsdChunkStateFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, has_seq_idx,
    )
    bm = SsdChunkStateFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = SsdChunkStateFwdOp(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
        has_seq_idx=has_seq_idx, tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(x, Bmat, dt, dA_cumsum, seq_idx):
        return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, n_groups=n_groups, seq_idx=seq_idx)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record("ssd_chunk_state_fwd", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
