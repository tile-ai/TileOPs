"""End-to-end Mamba-2 SSD forward benchmark: TileOPs vs mamba_ssm official.

Benchmarks the full Mamba-2 SSD forward pass (DaCumsum → SSDChunkState →
SSDStatePassing → SSDChunkScan) against mamba_chunk_scan_combined from
the official mamba_ssm library (Triton baseline), with a PyTorch fallback.

Run:
    pytest benchmarks/ops/bench_mamba2_e2e.py -m smoke
    pytest benchmarks/ops/bench_mamba2_e2e.py -m full --benchmark-json=results.json
"""

import pytest
import torch
import torch.nn.functional as F

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    then_dtype,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops.mamba.mamba2_fwd import Mamba2FwdOp
from workloads.mamba2_e2e import Mamba2FwdWorkload

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
    initial_states: torch.Tensor | None = None,
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
    s = (
        torch.zeros(b, h, p, n, device=x.device, dtype=torch.float32)
        if initial_states is None
        else initial_states.float().clone()
    )
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
    decay_ls = (
        torch.exp(dA_l - dA_s).masked_fill(~mask.view(1, 1, 1, Q, Q), 0.0).permute(0, 2, 1, 3, 4)
    )
    cb_heads = cb[:, :, torch.arange(h, device=x.device) // hpg, :, :]
    lcb = cb_heads * decay_ls * dt_c.permute(0, 1, 3, 2).unsqueeze(-2)
    wx_t = x_c.permute(0, 1, 3, 2, 4)
    y_intra = torch.einsum("bchls,bchsp->bchlp", lcb, wx_t).permute(0, 1, 3, 2, 4)

    return (y_hist + y_intra).reshape(b, S, h, p)


# FLOPS / memory calculators

# Benchmark test


def _mamba2_args(workload: dict) -> tuple:
    """Constructor arguments for one manifest workload row."""
    batch, seqlen, n_heads, d_head = workload["x_shape"]
    n_groups, d_state = workload["B_shape"][2], workload["B_shape"][3]
    return (
        batch,
        seqlen,
        n_heads,
        d_head,
        d_state,
        n_groups,
        workload.get("chunk_size", 256),
        bool(workload.get("dt_softplus", True)),
        "dt_bias_shape" in workload,
        "initial_states_shape" in workload,
    )


@pytest.mark.parametrize(
    "batch, seqlen, n_heads, d_head, d_state, n_groups, chunk_size, dt_softplus,"
    " has_dt_bias, has_initial_states, dtype, tune",
    workload_params(load_workloads(Mamba2FwdOp), then_dtype(_mamba2_args, tune=False)),
)
def test_mamba2_fwd_bench(
    batch,
    seqlen,
    n_heads,
    d_head,
    d_state,
    n_groups,
    chunk_size,
    dt_softplus,
    has_dt_bias,
    has_initial_states,
    dtype,
    tune,
):
    test = Mamba2FwdWorkload(
        batch,
        seqlen,
        n_heads,
        d_head,
        d_state,
        n_groups,
        dtype,
        chunk_size,
        dt_softplus,
    )
    inputs = test.gen_inputs()  # (x, dt, A, B, C, dt_bias)
    x, dt, A, B, C, dt_bias = inputs
    # A row that does not declare the optional input does not pass it, so the
    # sub-ops build the kernels without those branches.
    if not has_dt_bias:
        dt_bias = None
    initial_states = (
        torch.randn(batch, n_heads, d_head, d_state, dtype=torch.float32, device=x.device) * 0.1
        if has_initial_states
        else None
    )

    # ── TileOPs ──────────────────────────────────────────────────────────────
    op = Mamba2FwdOp(
        chunk_size=chunk_size,
        dt_softplus=dt_softplus,
        tune=tune,
    )
    bm = ManifestBenchmark(op, test)

    # Pass inputs directly so bench_kernel clones them each iteration,
    # giving accurate per-clone addressing and fair kernel-only timing.
    # Include dt_bias so all three paths see the same input distribution.
    functors = {"tileops": op.forward}

    # ── Official mamba_ssm Triton baseline ───────────────────────────────────
    # Pass the same 5 tensor inputs so both paths get identical clone treatment.
    # Non-tensor kwargs (chunk_size, dt_softplus, dt_bias) are captured by the
    # partial wrapper below — bench_kernel only clones the positional tensors.
    if _mamba_chunk_scan_combined is not None:

        def _mamba_wrapper(x, dt, A, B, C):
            return _mamba_chunk_scan_combined(
                x,
                dt,
                A,
                B,
                C,
                chunk_size,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
                initial_states=initial_states,
            )

        functors["mamba"] = (
            _mamba_wrapper,
            (
                x,
                dt,
                A,
                B,
                C,
            ),
        )

    def _torch_wrapper(x, dt, A, B, C):
        return mamba2_fwd_ref(x, dt, A, B, C, dt_bias, chunk_size, dt_softplus, initial_states)

    reference_args = (x, dt, A, B, C)
    functors["torch-ref"] = (_torch_wrapper, reference_args)
    functors[TORCH_COMPILE_TAG] = (compiled_reference(_torch_wrapper), reference_args)

    bm.compare(functors, x, dt, A, B, C, dt_bias, initial_states)
