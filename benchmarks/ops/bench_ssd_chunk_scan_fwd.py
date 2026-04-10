from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.ssd_chunk_scan import SsdChunkScanFwdOp
from workloads.ops.ssd_chunk_scan_fwd import SsdChunkScanFwdFixture, SsdChunkScanFwdTest


def ssd_chunk_scan_fwd_torch(x, cb, dA_cumsum, C, prev_states, dt):
    """Triton-aligned PyTorch reference for chunk scan."""
    b, c, L, h, p = x.shape

    y_off = torch.einsum("bclhn,bchnp->bclhp", C.float(), prev_states.float())
    a_l = dA_cumsum.permute(0, 2, 3, 1).unsqueeze(-1)
    y_off = y_off * torch.exp(a_l)

    a_lhs = dA_cumsum.unsqueeze(-1)
    a_rhs = dA_cumsum.unsqueeze(-2)
    decay = torch.exp(a_lhs - a_rhs)
    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask, 0)
    decay = decay.permute(0, 2, 1, 3, 4)
    dt_s = dt.float().permute(0, 1, 3, 2).unsqueeze(-2)
    lcb = cb.float() * decay * dt_s
    y_diag = torch.einsum("bchls,bcshp->bclhp", lcb, x.float())

    return y_off + y_diag


class SsdChunkScanFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
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
        t = self.workload
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
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(*args):
        return ssd_chunk_scan_fwd_torch(*args)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record("ssd_chunk_scan_fwd", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
