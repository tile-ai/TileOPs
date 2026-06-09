"""End-to-end Mamba-2 SSD forward benchmark: TileOPs vs mamba_ssm official.

Benchmarks the full Mamba-2 SSD forward pass (DaCumsum → SSDChunkState →
SSDStatePassing → SSDChunkScan) against mamba_chunk_scan_combined from
the official mamba_ssm library (Triton baseline), with a PyTorch fallback.

Run:
    pytest benchmarks/ops/bench_mamba2_e2e.py -m smoke
    pytest benchmarks/ops/bench_mamba2_e2e.py -m full --benchmark-json=results.json
"""

from typing import Optional

import pytest

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tests.ops.test_mamba import mamba2_fwd_ref as _mamba2_fwd_ref
from tileops.ops.mamba2_fwd import Mamba2FwdOp
from workloads.mamba2_e2e import Mamba2FwdFixture, Mamba2FwdTest

# ---------------------------------------------------------------------------
# Optional mamba_ssm Triton baseline
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined as _mamba_chunk_scan_combined,
    )
except ImportError:
    _mamba_chunk_scan_combined = None


# ---------------------------------------------------------------------------
# FLOPS / memory calculators
# ---------------------------------------------------------------------------

class Mamba2FwdBenchmark(BenchmarkBase["Mamba2FwdTest"]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, S, h, p, n = t.batch, t.seqlen, t.n_heads, t.d_head, t.d_state
        Q = t.chunk_size
        C = t.num_chunks

        # da_cumsum: ~7 ops/elem (bias + softplus + clamp + mul + cumsum)
        flops_dacumsum = 7 * b * S * h

        # chunk_state: GEMM per chunk: b*C * h * Q * p * n * 2
        flops_chunk_state = 2 * b * C * h * Q * p * n

        # state passing: b * C * h * p * n multiplies
        flops_state_pass = b * C * h * p * n

        # chunk_scan history: b * C * Q * h * n * p * 2 (C @ prev_states)
        flops_scan_hist = 2 * b * C * Q * h * n * p

        # chunk_scan intra: b * C * h * Q^2 * p * 2
        flops_scan_intra = 2 * b * C * h * Q * Q * p

        return float(flops_dacumsum + flops_chunk_state + flops_state_pass + flops_scan_hist + flops_scan_intra)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, S, h, p, n, g = t.batch, t.seqlen, t.n_heads, t.d_head, t.d_state, t.n_groups
        elem2 = 2  # bfloat16
        elem4 = 4  # float32

        reads = (
            b * S * h * p * elem2 +   # x
            b * S * h * elem4 +        # dt
            h * elem4 +                # A
            b * S * g * n * elem2 * 2  # B + C
        )
        writes = b * S * h * p * elem4  # y

        return float(reads + writes)


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------

@Mamba2FwdFixture
def test_mamba2_fwd_bench(batch, seqlen, n_heads, d_head, d_state, n_groups,
                           dtype, chunk_size, dt_softplus, tune):
    test = Mamba2FwdTest(
        batch, seqlen, n_heads, d_head, d_state, n_groups,
        dtype, chunk_size, dt_softplus,
    )
    bm = Mamba2FwdBenchmark(test)
    inputs = test.gen_inputs()  # (x, dt, A, B, C, dt_bias)
    x, dt, A, B, C, dt_bias = inputs

    # ── TileOPs ──────────────────────────────────────────────────────────────
    op = Mamba2FwdOp(
        batch=batch,
        seqlen=seqlen,
        n_heads=n_heads,
        d_head=d_head,
        d_state=d_state,
        n_groups=n_groups,
        dtype=dtype,
        chunk_size=chunk_size,
        dt_softplus=dt_softplus,
        has_initial_states=False,
        tune=tune,
    )

    # Pass inputs directly so bench_kernel clones them each iteration,
    # giving accurate per-clone addressing and fair kernel-only timing.
    # Include dt_bias so all three paths see the same input distribution.
    result = bm.profile(op.forward, x, dt, A, B, C, dt_bias)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # ── Official mamba_ssm Triton baseline ───────────────────────────────────
    # Pass the same 5 tensor inputs so both paths get identical clone treatment.
    # Non-tensor kwargs (chunk_size, dt_softplus, dt_bias) are captured by the
    # partial wrapper below — bench_kernel only clones the positional tensors.
    if _mamba_chunk_scan_combined is not None:
        def _mamba_wrapper(x, dt, A, B, C):
            return _mamba_chunk_scan_combined(
                x, dt, A, B, C,
                chunk_size,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
            )

        result_mamba = bm.profile(_mamba_wrapper, x, dt, A, B, C)
        BenchmarkReport.record(op, locals(), result_mamba, tag="mamba")
    else:
        def _torch_wrapper(x, dt, A, B, C):
            return _mamba2_fwd_ref(x, dt, A, B, C, dt_bias, chunk_size, dt_softplus)

        result_torch = bm.profile(_torch_wrapper, x, dt, A, B, C)
        BenchmarkReport.record(op, locals(), result_torch, tag="torch-ref")
