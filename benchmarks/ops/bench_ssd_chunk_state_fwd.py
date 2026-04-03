"""Benchmark for SsdChunkStateFwdOp vs mamba-ssm Triton baseline.

Baselines:
  - mamba-ssm (optional): mamba-ssm's _chunk_state_fwd Triton kernel; only runs
      when mamba_ssm is installed (`cd /path/to/mamba &&
      MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-deps --no-build-isolation -e .`).
  - torch-ref: hand-written PyTorch reference, always available as fallback.

Usage:
    conda run -n flashmlaenv python -m pytest benchmarks/ops/bench_ssd_chunk_state_fwd.py -vvs
    conda run -n flashmlaenv python benchmarks/ops/bench_ssd_chunk_state_fwd.py
"""

from typing import Optional

import pytest
import torch

try:
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd as _mamba_chunk_state_fwd
except ImportError:
    _mamba_chunk_state_fwd = None

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

    # TileOPs — warmup then profile
    op = SsdChunkStateFwdOp(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
        has_seq_idx=has_seq_idx, tune=tune,
    )
    op(*inputs)
    torch.cuda.synchronize()
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if _mamba_chunk_state_fwd is not None:
        # mamba-ssm _chunk_state_fwd expects:
        #   B (Bmat):     (b, seqlen, ngroups, dstate)       ← our layout, already correct
        #   x:            (b, seqlen, nheads, headdim)        ← our layout, already correct
        #   dt:           (b, nheads, nchunks, chunk_size)    ← our layout, already correct
        #   dA_cumsum:    (b, nheads, nchunks, chunk_size)    ← our layout, already correct
        #   seq_idx:      (b, seqlen) or None                 ← our layout, already correct
        # returns states: (b, nchunks, nheads, headdim, dstate) = our (b, c, h, p, n) ✓
        x, Bmat, dt, dA_cumsum, seq_idx = inputs

        def _mamba_fn(x, Bmat, dt, dA_cumsum, seq_idx):
            return _mamba_chunk_state_fwd(Bmat, x, dt, dA_cumsum, seq_idx=seq_idx)

        _mamba_fn(*inputs)
        torch.cuda.synchronize()
        result_bl = bm.profile(_mamba_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="mamba-ssm")
    else:
        def _torch_ref_fn(x, Bmat, dt, dA_cumsum, seq_idx):
            return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, n_groups=n_groups, seq_idx=seq_idx)

        _torch_ref_fn(*inputs)
        torch.cuda.synchronize()
        result_bl = bm.profile(_torch_ref_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
