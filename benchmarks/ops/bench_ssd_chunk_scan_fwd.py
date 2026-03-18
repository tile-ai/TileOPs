from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ssd_chunk_scan_fwd import (
    SsdChunkScanFwdFixture,
    SsdChunkScanFwdTest,
    ssd_chunk_scan_fwd_ref,
)
from tileops.ops.ssd_chunk_scan_fwd import SsdChunkScanFwdOp


class SsdChunkScanFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        b, c, L, h, p, n = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state,
        )
        # History path (Step 4): C @ prev_states per token
        #   b * c * L * h matmuls of shape (1, n) x (n, p) -> 2*n*p FLOPs each
        history_flops = b * c * L * h * 2 * n * p
        # Intra-chunk path (Step 1): lower-triangular lcb @ x
        #   b * c * h causal GEMMs of size (L, L) x (L, p) -> L*(L+1)/2 * 2*p FLOPs each
        diag_flops = b * c * h * (L * (L + 1) // 2) * 2 * p
        return float(history_flops + diag_flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        b, c, L, h, p, n = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state,
        )
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): x + cb + C + prev_states + dt
        reads = (
            b * c * L * h * p          # x
            + b * c * h * L * L        # cb
            + b * c * L * h * n        # C
            + b * c * h * n * p        # prev_states
            + b * c * L * h            # dt
        ) * elem
        # Reads (float32): dA_cumsum
        reads += b * h * c * L * 4
        # Writes (float32): out
        writes = b * c * L * h * p * 4
        return float(reads + writes)


@SsdChunkScanFwdFixture
def test_ssd_chunk_scan_fwd_bench(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune):
    test = SsdChunkScanFwdTest(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype)
    bm = SsdChunkScanFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = SsdChunkScanFwdOp(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("ssd_chunk_scan_fwd", locals(), result, tag="tileops")

    def baseline(*args):
        return ssd_chunk_scan_fwd_ref(*args)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record("ssd_chunk_scan_fwd", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
