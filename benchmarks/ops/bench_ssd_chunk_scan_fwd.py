"""
Benchmark: ssd_chunk_scan_fwd

Measures the TileOPs kernel against the Mamba-2 Triton reference
(_chunk_scan_fwd from mamba_ssm), falling back to the PyTorch
reference when mamba_ssm is not installed.

Kernel contract (official mamba_ssm interface)
----------------------------------------------
  out[b, c*L+l, h, p]
    = exp(dA_cumsum[b,h,c,l]) * sum_n C[b,c*L+l,g(h),n] * prev_states[b,c,h,p,n]
    + sum_{s<=l} cb[b,c,g(h),l,s]
                 * exp(dA_cumsum[b,h,c,l] - dA_cumsum[b,h,c,s])
                 * dt[b,h,c,s]
                 * x[b,c*L+s,h,p]

Layout (official, seqlen-fused)
--------------------------------
  x           [B, S, H, P]        dtype    S = num_chunks * chunk_len
  cb          [B, C, G, L, L]     dtype    group-owned
  dA_cumsum   [B, H, C, L]        float32
  C           [B, S, G, N]        dtype    group-owned
  prev_states [B, C, H, P, N]     dtype    P before N
  dt          [B, H, C, L]        dtype
  out         [B, S, H, P]        float32
"""
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.ssd_chunk_scan_fwd import SsdChunkScanFwdOp
from workloads.ops.ssd_chunk_scan_fwd import SsdChunkScanFwdTest

# ---------------------------------------------------------------------------
# Optional mamba_ssm Triton baseline
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd as _mamba_chunk_scan_fwd
except ImportError:
    _mamba_chunk_scan_fwd = None


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class SsdChunkScanFwdBenchmark(BenchmarkBase):
    """FLOPs and memory bandwidth for ssd_chunk_scan_fwd."""

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, c, L, h, p, n = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state,
        )
        # History path: C @ prev_states per token
        #   b * c * L * h matmuls of shape (1, n) x (n, p) -> 2*n*p FLOPs each
        history_flops = b * c * L * h * 2 * n * p
        # Intra-chunk path: lower-triangular lcb @ x
        #   b * c * h causal GEMMs of size (L, L) x (L, p) -> L*(L+1)/2 * 2*p FLOPs each
        diag_flops = b * c * h * (L * (L + 1) // 2) * 2 * p
        return float(history_flops + diag_flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, c, L, h, p, n, g = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state, t.n_groups,
        )
        S = c * L
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): x + cb + C + prev_states + dt
        reads = (
            b * S * h * p           # x          [B, S, H, P]
            + b * c * g * L * L     # cb         [B, C, G, L, L]
            + b * S * g * n         # C          [B, S, G, N]
            + b * c * h * p * n     # prev_states [B, C, H, P, N]
            + b * h * c * L         # dt         [B, H, C, L]
        ) * elem
        # Reads (float32): dA_cumsum [B, H, C, L]
        reads += b * h * c * L * 4
        # Writes (float32): out [B, S, H, P]
        writes = b * S * h * p * 4
        return float(reads + writes)


# ---------------------------------------------------------------------------
# Benchmark parameters
#
# Model-to-shape mapping (Mamba-2 defaults):
#   n_heads = d_model / 32,  head_dim = 64,  d_state = 128,  chunk_len = 256
#   num_chunks = seq_len // chunk_len  (chunk_len=256: 2k->8, 4k->16, 32k->128)
#   n_groups = 1 (Mamba-2 standard)
#
#   130M -> n_heads=24   370M -> n_heads=32   780M -> n_heads=48
#   1.3B -> n_heads=64   2.7B -> n_heads=80
#
# Schema: (batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune)
# ---------------------------------------------------------------------------
_SSD_CHUNK_SCAN_FWD_BENCH_PARAMS = [
    # ── unit-scale ──
    pytest.param(1, 2,  64, 4,  64,  32, 1, torch.float16,  False, id="b1-c2-L64-h4-p64-n32-fp16"),
    pytest.param(2, 4,  64, 8,  64,  64, 2, torch.float16,  False, id="b2-c4-L64-h8-p64-n64-fp16"),
    pytest.param(1, 2, 128, 4, 128,  32, 1, torch.bfloat16, False, id="b1-c2-L128-h4-p128-n32-bf16"),
    pytest.param(2, 2,  64, 4,  64,  32, 2, torch.bfloat16, False, id="b2-c2-L64-h4-p64-n32-bf16"),
    # ── 130M (n_heads=24) ──
    pytest.param(1,  16, 256, 24, 64, 128, 1, torch.float16, True, id="latency-130m-4k"),
    pytest.param(8,  16, 256, 24, 64, 128, 1, torch.float16, True, id="serving-130m-4k"),
    pytest.param(4, 128, 256, 24, 64, 128, 1, torch.float16, True, id="longctx-130m-32k"),
    # ── 370M (n_heads=32) ──
    pytest.param(1,  16, 256, 32, 64, 128, 1, torch.float16, True, id="latency-370m-4k"),
    pytest.param(8,  16, 256, 32, 64, 128, 1, torch.float16, True, id="serving-370m-4k"),
    pytest.param(4, 128, 256, 32, 64, 128, 1, torch.float16, True, id="longctx-370m-32k"),
    pytest.param(32,  8, 256, 32, 64, 128, 1, torch.float16, True, id="throughput-370m-2k"),
    # ── 780M (n_heads=48) ──
    pytest.param(1,  16, 256, 48, 64, 128, 1, torch.float16, True, id="latency-780m-4k"),
    pytest.param(8,  16, 256, 48, 64, 128, 1, torch.float16, True, id="serving-780m-4k"),
    pytest.param(4, 128, 256, 48, 64, 128, 1, torch.float16, True, id="longctx-780m-32k"),
    pytest.param(16,  8, 256, 48, 64, 128, 1, torch.float16, True, id="throughput-780m-2k"),
    # ── 1.3B (n_heads=64) ──
    pytest.param(1,  16, 256, 64, 64, 128, 1, torch.float16, True, id="latency-1p3b-4k"),
    pytest.param(8,  16, 256, 64, 64, 128, 1, torch.float16, True, id="serving-1p3b-4k"),
    pytest.param(2, 128, 256, 64, 64, 128, 1, torch.float16, True, id="longctx-1p3b-32k"),
    pytest.param(8,   8, 256, 64, 64, 128, 1, torch.float16, True, id="throughput-1p3b-2k"),
    # ── 2.7B (n_heads=80) ──
    pytest.param(1,  16, 256, 80, 64, 128, 1, torch.float16, True, id="latency-2p7b-4k"),
    pytest.param(4,  16, 256, 80, 64, 128, 1, torch.float16, True, id="serving-2p7b-4k"),
    pytest.param(2, 128, 256, 80, 64, 128, 1, torch.float16, True, id="longctx-2p7b-32k"),
    pytest.param(4,   8, 256, 80, 64, 128, 1, torch.float16, True, id="throughput-2p7b-2k"),
]


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune",
    _SSD_CHUNK_SCAN_FWD_BENCH_PARAMS,
)
def test_ssd_chunk_scan_fwd_bench(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = SsdChunkScanFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
    )
    bm = SsdChunkScanFwdBenchmark(test)
    inputs = test.gen_inputs()  # x, cb, dA_cumsum, C, prev_states, dt

    # ── TileOPs kernel ──
    op = SsdChunkScanFwdOp(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # ── Mamba-2 Triton baseline ──
    if _mamba_chunk_scan_fwd is not None:
        x, cb, dA_cumsum, C, prev_states, dt = inputs
        # All tensors are already in official mamba_ssm layout
        # mamba signature: _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, ...)
        def mamba_fwd():
            return _mamba_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, prev_states)

        result_mamba = bm.profile(mamba_fwd)
        BenchmarkReport.record(op, locals(), result_mamba, tag="mamba")
    else:
        from tests.ops.test_ssd_chunk_scan_fwd import ssd_chunk_scan_fwd_ref

        def torch_ref(x, cb, dA_cumsum, C, prev_states, dt):
            return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, n_groups)

        result_bl = bm.profile(torch_ref, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
