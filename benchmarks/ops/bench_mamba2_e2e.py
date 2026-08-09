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
from workloads.mamba2_e2e import Mamba2FwdFixture, Mamba2FwdWorkload

# Optional mamba_ssm Triton baseline
try:
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined as _mamba_chunk_scan_combined,
    )
except ImportError:
    _mamba_chunk_scan_combined = None


def mamba2_fwd_ref(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt_bias: torch.Tensor | None,
    chunk_size: int,
    dt_softplus: bool,
) -> torch.Tensor:
    """Pure-PyTorch baseline for the Mamba-2 State-Space Dual (SSD) forward pass.

    Computes the same result as mamba_chunk_scan_combined from mamba_ssm:
      out[l,p] = exp(dA[l]) * C[l] @ prev_state
               + sum_{s<=l} (C[l]@B[s]) * exp(dA[l]-dA[s]) * dt[s] * x[s,p]

    Inputs:
        x:           (B, S, H, P)     dtype
        dt:          (B, S, H)        float32
        A:           (H,)             float32  (log-space, <= 0)
        B:           (B, S, G, N)     dtype
        C:           (B, S, G, N)     dtype
        dt_bias:     (H,)             float32, optional
        chunk_size:  int
        dt_softplus: bool

    Returns:
        y: (B, S, H, P)  float32
    """
    b, S, h, p = x.shape
    n = B.shape[-1]
    g = B.shape[2]
    hpg = h // g
    Q = chunk_size
    num_chunks = S // Q

    # Step 1: DaCumsum
    dt_val = dt.float()
    if dt_bias is not None:
        dt_val = dt_val + dt_bias.float()
    if dt_softplus:
        dt_val = F.softplus(dt_val)
    dt_val = torch.clamp(dt_val, min=0.0)
    dt_chunked = dt_val.reshape(b, num_chunks, Q, h).permute(0, 3, 1, 2)
    dA = dt_chunked * A.float().view(1, h, 1, 1)
    dA_cumsum = dA.cumsum(dim=-1)

    # Step 2: CB = C[l] @ B[s]^T per chunk, lower-triangular, group-owned.
    B_c = B.float().reshape(b, num_chunks, Q, g, n)
    C_c = C.float().reshape(b, num_chunks, Q, g, n)
    cb = torch.einsum("bcqgn,bcsgn->bcgqs", C_c, B_c)
    mask = torch.ones(Q, Q, device=x.device, dtype=torch.bool).tril()
    cb = cb * mask.view(1, 1, 1, Q, Q)

    # Step 3: SSDChunkState
    decay = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)
    decay_c = decay.permute(0, 2, 3, 1)
    dt_c = dt_chunked.permute(0, 2, 3, 1)
    x_c = x.float().reshape(b, num_chunks, Q, h, p)
    B_heads = B_c[:, :, :, torch.arange(h, device=x.device) // hpg, :]
    wx = x_c * (decay_c * dt_c).unsqueeze(-1)
    chunk_states = torch.einsum("bcqhp,bcqhn->bchpn", wx, B_heads)

    # Step 4: SSDStatePassing
    exp_dA_chunk = torch.exp(dA_cumsum[:, :, :, -1])
    s = torch.zeros(b, h, p, n, device=x.device, dtype=torch.float32)
    prev_states_list = []
    for ci in range(num_chunks):
        prev_states_list.append(s.unsqueeze(1))
        scale = exp_dA_chunk[:, :, ci].view(b, h, 1, 1)
        s = scale * s + chunk_states[:, ci]
    prev_states = torch.cat(prev_states_list, dim=1)

    # Step 5: SSDChunkScan
    dA_c = dA_cumsum.permute(0, 2, 3, 1)
    C_heads = C_c[:, :, :, torch.arange(h, device=x.device) // hpg, :]

    y_hist = torch.einsum("bcqhn,bchpn->bcqhp", C_heads, prev_states.float())
    y_hist = y_hist * torch.exp(dA_c).unsqueeze(-1)

    dA_l = dA_cumsum.unsqueeze(-1)
    dA_s = dA_cumsum.unsqueeze(-2)
    decay_ls = torch.exp(dA_l - dA_s).masked_fill(
        ~mask.view(1, 1, 1, Q, Q), 0.0
    ).permute(0, 2, 1, 3, 4)
    cb_heads = cb[:, :, torch.arange(h, device=x.device) // hpg, :, :]
    lcb = cb_heads * decay_ls * dt_c.permute(0, 1, 3, 2).unsqueeze(-2)
    wx_t = x_c.permute(0, 1, 3, 2, 4)
    y_intra = torch.einsum("bchls,bchsp->bchlp", lcb, wx_t).permute(0, 1, 3, 2, 4)

    return (y_hist + y_intra).reshape(b, S, h, p)


# FLOPS / memory calculators

class Mamba2FwdBenchmark(BenchmarkBase["Mamba2FwdWorkload"]):

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


# Benchmark test

@Mamba2FwdFixture
def test_mamba2_fwd_bench(batch, seqlen, n_heads, d_head, d_state, n_groups,
                           dtype, chunk_size, dt_softplus, tune):
    test = Mamba2FwdWorkload(
        batch, seqlen, n_heads, d_head, d_state, n_groups,
        dtype, chunk_size, dt_softplus,
    )
    bm = Mamba2FwdBenchmark(test)
    inputs = test.gen_inputs()  # (x, dt, A, B, C, dt_bias)
    x, dt, A, B, C, dt_bias = inputs

    # ── TileOPs ──────────────────────────────────────────────────────────────
    op = Mamba2FwdOp(
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
            return mamba2_fwd_ref(x, dt, A, B, C, dt_bias, chunk_size, dt_softplus)

        result_torch = bm.profile(_torch_wrapper, x, dt, A, B, C)
        BenchmarkReport.record(op, locals(), result_torch, tag="torch-ref")
