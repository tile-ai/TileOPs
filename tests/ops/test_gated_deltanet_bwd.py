from typing import List, Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GatedDeltaNetBwdOp

# =============================================================================
# Torch reference implementations (test-only)
# =============================================================================

def _prepare_wy_repr_torch_ref(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, S, DK = k.shape
    assert S % chunk_size == 0
    BC = chunk_size
    Aw = torch.empty(B, H, S, BC, dtype=torch.float32, device=k.device)
    Au = torch.empty(B, H, S, BC, dtype=torch.float32, device=k.device)

    for b in range(B):
        for h in range(H):
            for c in range(S // BC):
                i0, i1 = c * BC, (c + 1) * BC
                kc = k[b, h, i0:i1, :].float()
                gc = g[b, h, i0:i1].float()
                bc = beta[b, h, i0:i1].float()
                k_beta = kc * bc.unsqueeze(-1)
                Gram = k_beta @ k_beta.T

                L = torch.tril(Gram, diagonal=-1).neg()
                L.diagonal(0).fill_(1.0)
                L_inv = torch.linalg.inv(L)
                Aw[b, h, i0:i1, :] = L_inv

                Gram_g = Gram * torch.exp(gc.unsqueeze(1) - gc.unsqueeze(0))
                L_u = torch.tril(Gram_g, diagonal=-1).neg()
                L_u.diagonal(0).fill_(1.0)
                L_u_inv = torch.linalg.inv(L_u)
                Au[b, h, i0:i1, :] = L_u_inv

    return Aw, Au


def _compute_w_u_torch_ref(
    Aw: torch.Tensor,
    Au: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def _compute_w_u_bwd_torch_ref(
    dw: torch.Tensor,
    du: torch.Tensor,
    Aw: torch.Tensor,
    Au: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, S, DK = k.shape
    _, _, _, DV = v.shape
    BC = chunk_size
    num_chunks = S // BC

    k_f, v_f = k.float(), v.float()
    beta_f = beta.float()
    dw_f, du_f = dw.float(), du.float()
    Aw_f, Au_f = Aw.float(), Au.float()

    k_beta = k_f * beta_f.unsqueeze(-1)
    v_beta = v_f * beta_f.unsqueeze(-1)

    Aw_ = Aw_f.reshape(B, H, num_chunks, BC, BC)
    Au_ = Au_f.reshape(B, H, num_chunks, BC, BC)
    k_beta_ = k_beta.reshape(B, H, num_chunks, BC, DK)
    v_beta_ = v_beta.reshape(B, H, num_chunks, BC, DV)
    dw_ = dw_f.reshape(B, H, num_chunks, BC, DK)
    du_ = du_f.reshape(B, H, num_chunks, BC, DV)

    dAw_ = torch.einsum("bhcid,bhcjd->bhcij", dw_, k_beta_)
    dAw = dAw_.reshape(B, H, S, BC)

    d_k_beta_ = torch.einsum("bhcji,bhcjd->bhcid", Aw_, dw_)
    d_k_beta = d_k_beta_.reshape(B, H, S, DK)

    dAu_ = torch.einsum("bhcid,bhcjd->bhcij", du_, v_beta_)
    dAu = dAu_.reshape(B, H, S, BC)

    d_v_beta_ = torch.einsum("bhcji,bhcjd->bhcid", Au_, du_)
    d_v_beta = d_v_beta_.reshape(B, H, S, DV)

    dk = d_k_beta * beta_f.unsqueeze(-1)
    dbeta = (d_k_beta * k_f).sum(dim=-1)

    dv = d_v_beta * beta_f.unsqueeze(-1)
    dbeta = dbeta + (d_v_beta * v_f).sum(dim=-1)

    return dAw, dAu, dk, dv, dbeta


def _kernel2_torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    S_0: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

        v_new_c = u_c - (w_c * torch.exp(g_c).unsqueeze(-1)) @ h

        o_part = torch.einsum("bhnk,bhkv->bhnv", q_c, h)
        o_part = o_part * torch.exp(g_c).unsqueeze(-1)
        attn = torch.einsum("bhnk,bhmk->bhnm", q_c, k_c)
        mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.bool), diagonal=0)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), 0.0)
        o_c = o_part + torch.einsum("bhnm,bhmv->bhnv", attn, v_new_c)
        o[:, :, i0:i1, :] = o_c

        g_last = g_c[:, :, -1:]
        k_scaled = k_c * torch.exp(g_last - g_c).unsqueeze(-1)
        h = h * torch.exp(g_last).view(B, H, 1, 1)
        h = h + torch.einsum("bhnk,bhnv->bhkv", k_scaled, v_new_c)
    return h, o


def _kernel2_bwd_torch_ref(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    S_0: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, S_len, DK = q.shape
    _, _, _, DV = u.shape
    BC = chunk_size
    num_chunks = S_len // BC
    device = q.device
    dtype = torch.float32
    q = q.float()
    k = k.float()
    g = g.float()
    w = w.float()
    u = u.float()
    do = do.float()
    S_0 = S_0.float()

    h = S_0.clone()
    saved: List[dict] = []
    for c in range(num_chunks):
        i0, i1 = c * BC, (c + 1) * BC
        q_c = q[:, :, i0:i1, :]
        k_c = k[:, :, i0:i1, :]
        g_c = g[:, :, i0:i1]
        w_c = w[:, :, i0:i1, :]
        u_c = u[:, :, i0:i1, :]

        v_new_c = u_c - (w_c * torch.exp(g_c).unsqueeze(-1)) @ h
        o_part = torch.einsum("bhnk,bhkv->bhnv", q_c, h) * torch.exp(g_c).unsqueeze(-1)
        attn = torch.einsum("bhnk,bhmk->bhnm", q_c, k_c)
        mask = torch.tril(torch.ones(BC, BC, device=device, dtype=torch.bool), diagonal=0)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), 0.0)

        g_last = g_c[:, :, -1:]
        k_scaled = k_c * torch.exp(g_last - g_c).unsqueeze(-1)
        h_next = h * torch.exp(g_last).view(B, H, 1, 1) + torch.einsum("bhnk,bhnv->bhkv", k_scaled, v_new_c)

        saved.append({
            "h_c": h.clone(), "q_c": q_c, "k_c": k_c, "g_c": g_c, "w_c": w_c,
            "u_c": u_c, "v_new_c": v_new_c, "o_part": o_part, "attn": attn,
            "g_last": g_last, "k_scaled": k_scaled,
        })
        h = h_next

    dq = torch.zeros_like(q, dtype=dtype, device=device)
    dk = torch.zeros_like(k, dtype=dtype, device=device)
    dg = torch.zeros_like(g, dtype=dtype, device=device)
    dw = torch.zeros_like(w, dtype=dtype, device=device)
    du = torch.zeros_like(u, dtype=dtype, device=device)

    mask = torch.tril(torch.ones(BC, BC, device=device, dtype=torch.bool), diagonal=0)
    mask_bhw = mask.unsqueeze(0).unsqueeze(0)

    dh_from_next = None
    for t in range(num_chunks - 1, -1, -1):
        i0, i1 = t * BC, (t + 1) * BC
        s = saved[t]
        h_c, q_c, k_c, g_c, w_c = s["h_c"], s["q_c"], s["k_c"], s["g_c"], s["w_c"]
        v_new_c, o_part, attn = s["v_new_c"], s["o_part"], s["attn"]
        g_last, k_scaled = s["g_last"], s["k_scaled"]

        do_c = do[:, :, i0:i1, :]
        dh = dh_from_next if dh_from_next is not None else torch.zeros(B, H, DK, DV, dtype=dtype, device=device)

        d_v_new_c = torch.einsum("bhnm,bhnv->bhmv", attn, do_c)
        d_attn = torch.einsum("bhnv,bhmv->bhnm", do_c, v_new_c)
        d_attn = d_attn.masked_fill(~mask_bhw, 0.0)

        exp_g = torch.exp(g_c).unsqueeze(-1)
        d_x = do_c * exp_g
        dg_c = (do_c * o_part).sum(dim=-1)
        d_q_c = torch.einsum("bhnv,bhkv->bhnk", d_x, h_c)
        dh = dh + torch.einsum("bhnk,bhnv->bhkv", q_c, d_x)

        d_q_c = d_q_c + torch.einsum("bhnm,bhmk->bhnk", d_attn, k_c)
        d_k_c = torch.einsum("bhnm,bhnk->bhmk", d_attn, q_c)

        P = w_c * exp_g
        dP = -torch.einsum("bhnv,bhkv->bhnk", d_v_new_c, h_c)
        dh = dh - torch.einsum("bhnk,bhnv->bhkv", P, d_v_new_c)
        d_w_c = dP * exp_g
        dg_c = dg_c + (P * dP).sum(dim=-1)

        exp_g_last = torch.exp(g_last).view(B, H, 1, 1)
        if dh_from_next is not None:
            dh = dh + dh_from_next * exp_g_last
            d_v_new_c = d_v_new_c + torch.einsum("bhnk,bhkv->bhnv", k_scaled, dh_from_next)
            d_k_scaled = torch.einsum("bhnv,bhkv->bhnk", v_new_c, dh_from_next)
            d_k_c = d_k_c + d_k_scaled * torch.exp(g_last - g_c).unsqueeze(-1)
            dg_c = dg_c - (d_k_scaled * k_scaled).sum(dim=-1)
            d_g_last = (dh_from_next * h_c * exp_g_last).sum(dim=(-2, -1)) + (d_k_scaled * k_scaled).sum(dim=(-2, -1))
            dg_c[:, :, -1] = dg_c[:, :, -1] + d_g_last

        dq[:, :, i0:i1, :] = d_q_c
        dk[:, :, i0:i1, :] = d_k_c
        dg[:, :, i0:i1] = dg_c
        dw[:, :, i0:i1, :] = d_w_c
        du[:, :, i0:i1, :] = d_v_new_c
        dh_from_next = dh

    return dq, dk, dg, dw, du


def _gated_deltanet_bwd_torch_ref(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aw: torch.Tensor,
    Au: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, S, DK = k.shape
    _, _, _, DV = v.shape
    S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)

    w, u = _compute_w_u_torch_ref(Aw, Au, k, v, beta, chunk_size)
    dq, dk_k2, dg, dw, du = _kernel2_bwd_torch_ref(do, q, k, g, w, u, S_0, chunk_size)
    _dAw, _dAu, dk_wu, dv, dbeta = _compute_w_u_bwd_torch_ref(dw, du, Aw, Au, k, v, beta, chunk_size)

    dk = dk_k2 + dk_wu
    return dq, dk, dv, dg, dbeta


# =============================================================================
# Backward correctness tests
# =============================================================================

def _get_tolerances(dtype: torch.dtype) -> dict:
    # Tolerances are looser than DEVELOPMENT.md defaults (fp16: 1e-3, bf16: 1.6e-2)
    # because the backward pass chains multiple sequential chunk recurrences
    # (forward state, reverse dh propagation, compute_w_u_bwd), compounding fp32
    # rounding errors across every chunk boundary. See test_gated_deltanet_fwd.py
    # for additional rationale.
    if dtype == torch.float32:
        return {"atol": 1e-2, "rtol": 1e-2}
    elif dtype == torch.float16:
        return {"atol": 5e-2, "rtol": 5e-2}
    else:  # bfloat16
        return {"atol": 1e-1, "rtol": 1e-1}


class GatedDeltaNetBwdTest(TestBase):

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, ...]:
        B, H, S, DK, DV = self.batch, self.heads, self.seq_len, self.dim_k, self.dim_v
        torch.manual_seed(42)
        q = torch.randn(B, H, S, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, H, S, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, H, S, DV, device="cuda", dtype=self.dtype) * 0.1
        g = -torch.rand(B, H, S, device="cuda", dtype=self.dtype)
        beta = torch.rand(B, H, S, device="cuda", dtype=self.dtype) * 0.5
        Aw, Au = _prepare_wy_repr_torch_ref(k, g, beta, self.chunk_size)
        w, u = _compute_w_u_torch_ref(Aw, Au, k, v, beta, self.chunk_size)
        S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device="cuda")
        _S, o = _kernel2_torch_ref(q, k, g, w, u, S_0, self.chunk_size)
        do = torch.randn_like(o) * 0.1
        return do, q, k, v, g, beta

    def ref_program(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        BC = self.chunk_size
        Aw, Au = _prepare_wy_repr_torch_ref(k, g, beta, BC)
        dq, dk, dv, dg, dbeta = _gated_deltanet_bwd_torch_ref(do, q, k, v, g, beta, Aw, Au, BC)
        return dq.to(self.dtype), dk.to(self.dtype), dv.to(self.dtype), dg.to(self.dtype), dbeta.to(self.dtype)


class GatedDeltaNetBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(2, 64, 2, 64, 64, 32, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@GatedDeltaNetBwdFixture
def test_gated_deltanet_bwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size
    q = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, H, S, device="cuda", dtype=dtype)
    beta = torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5

    Aw, Au = _prepare_wy_repr_torch_ref(k, g, beta, BC)
    w, u = _compute_w_u_torch_ref(Aw, Au, k, v, beta, BC)
    S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device="cuda")
    _S, o = _kernel2_torch_ref(q, k, g, w, u, S_0, BC)
    do = torch.randn_like(o) * 0.1

    ref_outputs = _gated_deltanet_bwd_torch_ref(do, q, k, v, g, beta, Aw, Au, BC)

    op = GatedDeltaNetBwdOp(B, H, S, DK, DV, BC, dtype, tune=tune)
    op_outputs = op.forward(do, q, k, v, g, beta)

    tols = _get_tolerances(dtype)
    for ref_out, op_out in zip(ref_outputs, op_outputs, strict=True):
        torch.testing.assert_close(op_out, ref_out.to(dtype), **tols)
    print("All checks passed for GatedDeltaNetBwdOp.")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
