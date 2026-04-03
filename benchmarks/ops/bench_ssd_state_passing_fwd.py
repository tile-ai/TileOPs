"""Benchmark for SsdStatePassingFwdOp vs mamba-ssm Triton baseline.

Baselines:
  - mamba-ssm (optional): mamba-ssm's _state_passing_fwd Triton kernel; only runs
      when mamba_ssm is installed (`cd /path/to/mamba &&
      MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-deps --no-build-isolation -e .`).
  - torch-ref: hand-written PyTorch reference, always available as fallback.

Usage:
    conda run -n flashmlaenv python -m pytest benchmarks/ops/bench_ssd_state_passing_fwd.py -vvs
    conda run -n flashmlaenv python benchmarks/ops/bench_ssd_state_passing_fwd.py
"""

from typing import Optional

import pytest
import torch

try:
    from mamba_ssm.ops.triton.ssd_state_passing import (
        _state_passing_fwd as _mamba_state_passing_fwd,
    )
except ImportError:
    _mamba_state_passing_fwd = None

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ssd_state_passing_fwd import (
    SsdStatePassingFwdFixture,
    SsdStatePassingFwdTest,
    ssd_state_passing_fwd_ref,
)
from tileops.ops.ssd_state_passing_fwd import SsdStatePassingFwdOp


class SsdStatePassingFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        b, c, h, d = t.batch, t.num_chunks, t.n_heads, t.d_state
        # Per chunk: scale multiply + add for each (b, h, d) element
        # 2 FLOPs (mul + add) per element per chunk
        flops = b * c * h * d * 2
        return float(flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        b, c, h, d = t.batch, t.num_chunks, t.n_heads, t.d_state
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): states
        reads = b * c * h * d * elem
        # Reads (float32): dA_chunk_cumsum + initial_states
        reads += b * h * c * 4 + b * h * d * 4
        # Writes (float32): out + final_states
        writes = (b * c * h * d + b * h * d) * 4
        return float(reads + writes)


@SsdStatePassingFwdFixture
def test_ssd_state_passing_fwd_bench(batch, num_chunks, n_heads, d_state, dtype, tune):
    test = SsdStatePassingFwdTest(batch, num_chunks, n_heads, d_state, dtype)
    bm = SsdStatePassingFwdBenchmark(test)
    inputs = test.gen_inputs()

    # TileOPs — warmup then profile
    op = SsdStatePassingFwdOp(batch, num_chunks, n_heads, d_state, dtype=dtype, tune=tune)
    op(*inputs)
    torch.cuda.synchronize()
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if _mamba_state_passing_fwd is not None:
        # mamba-ssm _state_passing_fwd expects:
        #   states:           (b, c, h, d)  ← our layout, already correct
        #   dA_chunk_cumsum:  (b, h, c)     ← our layout, already correct
        #   initial_states:   (b, h, d)     ← our layout, already correct
        # returns (out, final_states) both float32 — same as ref

        def _mamba_fn(states, dA_chunk_cumsum, initial_states):
            return _mamba_state_passing_fwd(states, dA_chunk_cumsum, initial_states)

        _mamba_fn(*inputs)
        torch.cuda.synchronize()
        result_bl = bm.profile(_mamba_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="mamba-ssm")
    else:
        def _torch_ref_fn(states, dA_chunk_cumsum, initial_states):
            return ssd_state_passing_fwd_ref(states, dA_chunk_cumsum, initial_states)

        _torch_ref_fn(*inputs)
        torch.cuda.synchronize()
        result_bl = bm.profile(_torch_ref_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
