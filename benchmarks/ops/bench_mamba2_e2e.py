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
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
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
# PyTorch reference (benchmark-local, no oracle-leakage from tests/)
# ---------------------------------------------------------------------------

def _mamba2_fwd_ref(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt_bias: Optional[torch.Tensor],
    chunk_size: int,
    dt_softplus: bool,
) -> torch.Tensor:
    """Pure-PyTorch reference for the Mamba-2 SSD forward pass.

    Inputs:
        x:       (B, S, H, P)       dtype
        dt:      (B, S, H)          float32
        A:       (H,)               float32  (log-space, ≤ 0)
        B:       (B, S, G, N)       dtype
        C:       (B, S, G, N)       dtype
        dt_bias: (H,)               float32, optional
        chunk_size: int
        dt_softplus: bool

    Returns:
        y: (B, S, H, P)  float32
    """
    b, S, h, p = x.shape
    n = B.shape[-1]
    g = B.shape[2]
    heads_per_group = h // g
    Q = chunk_size
    num_chunks = S // Q

    # --- Step 1: DaCumsum ---
    dt_val = dt.float()
    if dt_bias is not None:
        dt_val = dt_val + dt_bias.float()
    if dt_softplus:
        dt_val = F.softplus(dt_val)
    dt_val = torch.clamp(dt_val, min=0.0)
    # reshape to (B, H, C, Q)
    dt_chunked = dt_val.reshape(b, num_chunks, Q, h).permute(0, 3, 1, 2)  # (B, H, C, Q)
    dA = dt_chunked * A.float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)   # (B, H, C, Q)
    dA_cumsum = dA.cumsum(dim=-1)  # (B, H, C, Q)

    # --- Step 2: CB matrix per group ---
    # Pick representative head per group
    dA_group = dA_cumsum[:, ::heads_per_group, :, :]  # (B, G, C, Q)
    dA_g = dA_group.permute(0, 2, 1, 3)               # (B, C, G, Q)
    dA_l = dA_g.unsqueeze(-1)   # (B, C, G, Q, 1)
    dA_s = dA_g.unsqueeze(-2)   # (B, C, G, 1, Q)
    cb = torch.exp(dA_l - dA_s)
    mask = torch.ones(Q, Q, device=x.device, dtype=torch.bool).tril()
    cb = cb * mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # --- Step 3: SSDChunkState ---
    # chunk_states[b,c,h,p,n] = sum_l x[b,c*Q+l,h,p] * B[b,c*Q+l,g(h),n]
    #                             * exp(dA_cumsum[-1] - dA_cumsum[l]) * dt[l]
    x_chunked = x.float().reshape(b, num_chunks, Q, h, p)   # (B, C, Q, H, P)
    B_chunked = B.float().reshape(b, num_chunks, Q, g, n)   # (B, C, Q, G, N)
    dt_c = dt_chunked.permute(0, 2, 3, 1)  # (B, C, Q, H)

    # decay = exp(dA_cumsum[-1] - dA_cumsum[l])  (B, H, C, Q) -> (B, C, Q, H)
    dA_last = dA_cumsum[:, :, :, -1].unsqueeze(-1)  # (B, H, C, 1)
    decay = torch.exp(dA_last - dA_cumsum)          # (B, H, C, Q)
    decay_c = decay.permute(0, 2, 3, 1)             # (B, C, Q, H)

    # x weighted by decay * dt: (B, C, Q, H, P)
    wx = x_chunked * (decay_c * dt_c).unsqueeze(-1)

    # broadcast B from groups to heads: (B, C, Q, H, N)
    B_heads = B_chunked[:, :, :, torch.arange(h, device=x.device) // heads_per_group, :]

    # chunk_states: (B, C, H, P, N)
    chunk_states = torch.einsum("bcqhp,bcqhn->bchpn", wx, B_heads)

    # --- Step 4: SSDStatePassing (recurrent scan) ---
    dA_chunk_cumsum = dA_cumsum[:, :, :, -1]  # (B, H, C)
    exp_dA_chunk = torch.exp(dA_chunk_cumsum)  # (B, H, C)

    s = torch.zeros(b, h, p, n, dtype=torch.float32, device=x.device)
    prev_states = []
    for c_idx in range(num_chunks):
        prev_states.append(s.unsqueeze(1))  # (B, 1, H, P, N)
        decay_c_val = exp_dA_chunk[:, :, c_idx].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        s = decay_c_val * s + chunk_states[:, c_idx, :, :, :]

    prev_states = torch.cat(prev_states, dim=1).to(x.dtype)  # (B, C, H, P, N)

    # --- Step 5: SSDChunkScan ---
    C_chunked = C.float().reshape(b, num_chunks, Q, g, n)   # (B, C, Q, G, N)
    C_heads = C_chunked[:, :, :, torch.arange(h, device=x.device) // heads_per_group, :]
    dA_c = dA_cumsum.permute(0, 2, 3, 1)  # (B, C, Q, H)

    # History: exp(dA_l) * C[l] @ prev_states
    y_hist = torch.einsum("bcqhn,bchpn->bcqhp", C_heads, prev_states.float())
    y_hist = y_hist * torch.exp(dA_c).unsqueeze(-1)

    # Intra: sum_{s<=l} cb[l,s] * exp(dA_l-dA_s) * dt[s] * x[s]
    cb_heads = cb.float()[:, :, torch.arange(h, device=x.device) // heads_per_group, :, :]
    # (B, C, H, Q, Q)
    wx_intra = x_chunked.float() * dt_c.unsqueeze(-1)  # (B, C, Q, H, P)
    # wx_intra transposed for matmul: (B, C, H, Q, P)
    wx_t = wx_intra.permute(0, 1, 3, 2, 4)
    # y_intra[b,c,h,l,p] = sum_s cb_heads[b,c,h,l,s] * wx_t[b,c,h,s,p]
    # result is (B, C, H, L, P); transpose to (B, C, Q, H, P) to match y_hist
    y_intra = torch.einsum("bchlq,bchqp->bchlp", cb_heads, wx_t).permute(0, 1, 3, 2, 4)

    y = (y_hist + y_intra).reshape(b, S, h, p)
    return y


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
