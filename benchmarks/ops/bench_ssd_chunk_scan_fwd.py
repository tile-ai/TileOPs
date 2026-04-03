"""Benchmark for SsdChunkScanFwdOp vs mamba-ssm Triton baseline.

Baselines:
  - mamba-ssm (optional): mamba-ssm's _chunk_scan_fwd Triton kernel; only runs
      when mamba_ssm is installed (`cd /path/to/mamba &&
      MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-deps --no-build-isolation -e .`).
  - torch-ref: hand-written PyTorch reference, always available as fallback.

Usage:
    conda run -n flashmlaenv python -m pytest benchmarks/ops/bench_ssd_chunk_scan_fwd.py -vvs
    conda run -n flashmlaenv python benchmarks/ops/bench_ssd_chunk_scan_fwd.py
"""

from typing import Optional

import pytest
import torch

try:
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd as _mamba_chunk_scan_fwd
except ImportError:
    _mamba_chunk_scan_fwd = None

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

    # TileOPs — warmup then profile
    op = SsdChunkScanFwdOp(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune=tune)
    op(*inputs)
    torch.cuda.synchronize()
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if _mamba_chunk_scan_fwd is not None:
        # mamba-ssm _chunk_scan_fwd expects:
        #   cb:          (b, c, h, L, L)        ← our layout, already correct
        #   x:           (b, seqlen=c*L, h, p)  ← reshape from (b, c, L, h, p)
        #   dt:          (b, h, c, L)            ← permute from (b, c, L, h)
        #   dA_cumsum:   (b, h, c, L)            ← our layout, already correct
        #   C:           (b, seqlen=c*L, h, n)   ← reshape from (b, c, L, h, n)
        #   states:      (b, c, h, p, n)         ← permute from (b, c, h, n, p)
        # returns out:   (b, seqlen=c*L, h, p)   ← reshape back to (b, c, L, h, p)
        x, cb, dA_cumsum, C, prev_states, dt = inputs
        b, c, L, h, p, n = batch, num_chunks, chunk_len, n_heads, d_head, d_state

        x_m           = x.reshape(b, c * L, h, p)
        dt_m          = dt.permute(0, 3, 1, 2).contiguous()       # (b, h, c, L)
        C_m           = C.reshape(b, c * L, h, n)
        states_m      = prev_states.permute(0, 1, 2, 4, 3).contiguous()  # (b, c, h, p, n)

        def _mamba_fn(x, cb, dA_cumsum, C, prev_states, dt):
            out, _ = _mamba_chunk_scan_fwd(cb, x_m, dt_m, dA_cumsum, C_m, states_m)
            return out.reshape(b, c, L, h, p)

        _mamba_fn(*inputs)
        torch.cuda.synchronize()
        result_bl = bm.profile(_mamba_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="mamba-ssm")
    else:
        def _torch_ref_fn(*args):
            return ssd_chunk_scan_fwd_ref(*args)

        _torch_ref_fn(*inputs)
        torch.cuda.synchronize()
        result_bl = bm.profile(_torch_ref_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
