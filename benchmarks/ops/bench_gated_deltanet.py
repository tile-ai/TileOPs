"""Benchmark: TileOPs Gated DeltaNet vs FLA chunk_gated_delta_rule.

Compares forward and backward latency across sequence lengths and dtypes.

When FLA is not installed, the forward benchmark still runs against a pure-torch
reference baseline (tagged "torch"), so a missing optional dependency never blocks
the nightly. The backward benchmark has no such fallback: a reference backward
reached through autograd runs on the engine's thread, where the timer cannot tell
which iteration launched a kernel.

Layout convention:
    TileOPs uses BHSD: q/k [B, H, S, DK], v [B, H, S, DV], g/beta [B, H, S].
    FLA uses BTHK:     q/k [B, T, H, K],  v [B, T, H, V],  g/beta [B, T, H].
    Tensors are permuted before calling FLA to ensure both implementations
    compute the same function.
"""

import warnings
from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport, backward_of
from tileops.ops import GatedDeltaNetBwdOp, GatedDeltaNetFwdOp, GatedDeltaNetOp
from workloads.linear_attention import GatedDeltaNetFwdWorkload
from workloads.workload_base import FixtureBase


def compute_w_u_torch(Aw, Au, k, v, beta, chunk_size):
    B, H, S, DK = k.shape
    _, _, _, DV = v.shape
    BC = chunk_size
    num_chunks = S // BC
    k_beta = k.float() * beta.unsqueeze(-1)
    v_beta = v.float() * beta.unsqueeze(-1)
    Aw_ = Aw.reshape(B, H, num_chunks, BC, BC)
    Au_ = Au.reshape(B, H, num_chunks, BC, BC)
    k_beta_ = k_beta.reshape(B, H, num_chunks, BC, DK)
    v_beta_ = v_beta.reshape(B, H, num_chunks, BC, DV)
    w = torch.einsum("bhcij,bhcjd->bhcid", Aw_, k_beta_).reshape(B, H, S, DK)
    u = torch.einsum("bhcij,bhcjd->bhcid", Au_, v_beta_).reshape(B, H, S, DV)
    return w, u


def kernel2_gated_deltanet_torch(q, k, g, w, u, S_0, chunk_size):
    B, H, S_len, DK = q.shape
    _, _, _, DV = u.shape
    BC = chunk_size
    num_chunks = S_len // BC
    q, k, g, w, u = q.float(), k.float(), g.float(), w.float(), u.float()
    h = S_0.float().clone()

    o = torch.zeros(B, H, S_len, DV, dtype=torch.float32, device=q.device)
    for c in range(num_chunks):
        i0, i1 = c * BC, (c + 1) * BC
        q_c = q[:, :, i0:i1, :]
        k_c = k[:, :, i0:i1, :]
        g_c = g[:, :, i0:i1]
        w_c = w[:, :, i0:i1, :]
        u_c = u[:, :, i0:i1, :]

        g_last_val = g_c[:, :, -1:]
        v_new_c = u_c - (w_c * torch.exp(g_c + g_last_val).unsqueeze(-1)) @ h

        o_part = torch.einsum("bhnk,bhkv->bhnv", q_c, h)
        o_part = o_part * torch.exp(g_c).unsqueeze(-1)
        attn = torch.einsum("bhnk,bhmk->bhnm", q_c, k_c)
        Gamma_causal = torch.exp(g_c.unsqueeze(-1) - g_c.unsqueeze(-2))
        mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.bool), diagonal=0)
        attn = (attn * Gamma_causal).masked_fill(~mask.unsqueeze(0).unsqueeze(0), 0.0)
        o_c = o_part + torch.einsum("bhnm,bhmv->bhnv", attn, v_new_c)
        o[:, :, i0:i1, :] = o_c

        g_last = g_c[:, :, -1:]
        k_scaled = k_c * torch.exp(g_last - g_c).unsqueeze(-1)
        h = h * torch.exp(g_last).view(B, H, 1, 1)
        h = h + torch.einsum("bhnk,bhnv->bhkv", k_scaled, v_new_c)
    return h, o


def prepare_wy_repr_gated_torch(k, g_cum, beta, chunk_size):
    B, H, S, DK = k.shape
    assert S % chunk_size == 0
    BC = chunk_size
    NC = S // BC
    kc = k.float().reshape(B, H, NC, BC, DK)
    gc = g_cum.float().reshape(B, H, NC, BC)
    bc = beta.float().reshape(B, H, NC, BC)
    gram = kc @ kc.transpose(-2, -1)
    gamma = torch.exp(gc.unsqueeze(-1) - gc.unsqueeze(-2))
    m = bc.unsqueeze(-1) * (gamma * gram)
    eye = torch.eye(BC, dtype=torch.float32, device=k.device)
    a_g = eye + torch.tril(m, diagonal=-1)
    a_g_inv = torch.linalg.inv(a_g).reshape(B, H, S, BC)
    return a_g_inv, a_g_inv.clone()


class GatedDeltaNetFwdTestBaseline(GatedDeltaNetFwdWorkload):
    """Times the batched WY-representation idiom, not the workload reference.

    ``prepare_wy_repr_gated_torch`` in this module inverts every chunk in one
    batched call; the workload reference loops over (batch, head, chunk) so the
    test can read it. Same math, different cost.
    """

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        B, H, S, DK = k.shape
        _, _, _, DV = v.shape
        # Chunk-local cumulative sum of g (paper requires cumulated gates)
        BC = self.chunk_size
        g_cum = g.float().reshape(B, H, S // BC, BC).cumsum(-1).reshape(B, H, S).to(g.dtype)
        Aw, Au = prepare_wy_repr_gated_torch(k, g_cum, beta, self.chunk_size)
        w, u = compute_w_u_torch(Aw, Au, k, v, beta, self.chunk_size)
        S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)
        _S, o = kernel2_gated_deltanet_torch(q, k, g_cum, w, u, S_0, self.chunk_size)
        return o.to(self.dtype)

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None


def _to_fla_layout(q, k, v, g, beta):
    """Convert TileOPs BHSD tensors to FLA BTHK layout."""
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        g.permute(0, 2, 1).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )


# Forward benchmark

class GatedDeltaNetFwdBenchmark(BenchmarkBase[GatedDeltaNetFwdWorkload]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 2.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (2 * DK + 2 * DV + 2) * elem


class GatedDeltaNetVsFlaFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            # chunk_size=32
            (2, 4096, 4, 64, 64, 32, torch.float16, False),
            (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
            # chunk_size=64
            (2, 2048, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.float16, False),
            (2, 8192, 4, 64, 64, 64, torch.float16, False),
            (2, 16384, 4, 64, 64, 64, torch.float16, False),
            (2, 32768, 4, 64, 64, 64, torch.float16, False),
        ]),
    ]


@GatedDeltaNetVsFlaFwdFixture
def test_gated_deltanet_vs_fla_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GatedDeltaNetFwdTestBaseline(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetFwdBenchmark(test)
    inputs = test.gen_inputs()  # q, k, v, g, beta  (BHSD)

    # --- TileOPs (BHSD) ---
    op = GatedDeltaNetFwdOp(chunk_size=chunk_size, tune=tune)
    functors = {"tileops": op}

    if chunk_gated_delta_rule is not None:
        # --- FLA (BTHK) ---
        q, k, v, g, beta = inputs
        scale = dim_k ** -0.5
        q_fla, k_fla, v_fla, g_fla, beta_fla = _to_fla_layout(q, k, v, g, beta)

        def fla_fwd():
            return chunk_gated_delta_rule(q_fla, k_fla, v_fla, g_fla, beta_fla, scale=scale)

        functors["fla"] = (fla_fwd, ())
    else:
        # --- Torch reference baseline ---
        functors["torch"] = test.ref_program

    bm.compare(functors, *inputs, record_as=op, params=locals())


# Backward benchmark

class GatedDeltaNetBwdBenchmark(BenchmarkBase[GatedDeltaNetFwdWorkload]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 4.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (4 * DK + 3 * DV + 4) * elem


class GatedDeltaNetVsFlaBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            # chunk_size=32
            (2, 4096, 4, 64, 64, 32, torch.float16, False),
            (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
            # chunk_size=64
            (2, 2048, 4, 64, 64, 64, torch.float16, False),
            (2, 4096, 4, 64, 64, 64, torch.float16, False),
            (2, 8192, 4, 64, 64, 64, torch.float16, False),
            (2, 16384, 4, 64, 64, 64, torch.float16, False),
        ]),
    ]


@GatedDeltaNetVsFlaBwdFixture
def test_gated_deltanet_vs_fla_bwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GatedDeltaNetFwdTestBaseline(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetBwdBenchmark(test)

    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size
    q = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, H, S, device="cuda", dtype=dtype)
    beta = torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    # --- TileOPs: fwd to get S, then profile bwd only ---
    fwd_op = GatedDeltaNetFwdOp(chunk_size=BC)
    _o, S_fwd, _Aw, _Au = fwd_op.forward(q, k, v, g, beta)

    bwd_op = GatedDeltaNetBwdOp(chunk_size=BC, tune=tune)
    functors = {"tileops": bwd_op.forward}

    if chunk_gated_delta_rule is not None:
        # --- FLA: bwd only via autograd (BTHK layout) ---
        scale = DK ** -0.5
        q_fla, k_fla, v_fla, g_fla, beta_fla = _to_fla_layout(q, k, v, g, beta)
        do_fla = do.permute(0, 2, 1, 3).contiguous()  # [B,H,S,DV] -> [B,S,H,DV]

        q_fla = q_fla.detach().requires_grad_(True)
        k_fla = k_fla.detach().requires_grad_(True)
        v_fla = v_fla.detach().requires_grad_(True)
        g_fla = g_fla.detach().requires_grad_(True)
        beta_fla = beta_fla.detach().requires_grad_(True)

        # Run fwd once to build the graph, then time the backward node directly
        o_fla, _ = chunk_gated_delta_rule(q_fla, k_fla, v_fla, g_fla, beta_fla, scale=scale)
        # One grad per forward output: fla returns (o, final_state).
        fla_backward = backward_of(o_fla)

        def fla_bwd():
            return fla_backward(do_fla, None)

        functors["fla"] = (fla_bwd, ())
    else:
        warnings.warn(
            "fla is unavailable and the torch autograd reference cannot be timed "
            "(its backward runs on autograd's engine thread, where the timer cannot "
            "attribute the kernels); the baseline column will be omitted from results.",
            RuntimeWarning,
            stacklevel=2,
        )

    bm.compare(functors, do, q, k, v, g, beta, S_fwd,
               record_as=bwd_op, params=locals())


# Combined fwd+bwd benchmark (fair comparison: both measure fwd+bwd total)
