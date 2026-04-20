
import pytest
import torch

from tests.test_base import FixtureBase
from tileops.ops import GatedDeltaNetBwdOp


def _differentiable_fwd(q, k, v, g_raw, beta, chunk_size):
    """Fully differentiable chunked forward matching paper (Eq. 10 via WY)."""
    B, H, S, DK = q.shape
    DV = v.shape[-1]
    BC = chunk_size
    NC = S // BC
    g_cum = g_raw.float().reshape(B, H, NC, BC).cumsum(-1).reshape(B, H, S)
    h = q.new_zeros(B, H, DK, DV)
    o_chunks = []
    eye = torch.eye(BC, device=q.device, dtype=torch.float32)
    mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.float32))
    for c in range(NC):
        sl = slice(c * BC, (c + 1) * BC)
        qc = q[:, :, sl, :].float()
        kc = k[:, :, sl, :].float()
        vc = v[:, :, sl, :].float()
        gc = g_cum[:, :, sl]
        bc = beta[:, :, sl].float()
        Gram = torch.einsum("bhik,bhjk->bhij", kc, kc)
        Gamma = torch.exp(gc.unsqueeze(-1) - gc.unsqueeze(-2))
        M = bc.unsqueeze(-1) * (Gamma * Gram)
        A = eye + torch.tril(M, diagonal=-1)
        A_inv = torch.linalg.inv(A)
        wc = A_inv @ (kc * bc.unsqueeze(-1))
        uc = A_inv @ (vc * bc.unsqueeze(-1))
        g_last = gc[:, :, -1:]
        v_new = uc - (wc * torch.exp(gc + g_last).unsqueeze(-1)) @ h
        o_part = (qc @ h) * torch.exp(gc).unsqueeze(-1)
        attn = (qc @ kc.transpose(-2, -1)) * Gamma * mask
        o_c = o_part + attn @ v_new
        o_chunks.append(o_c)
        k_sc = kc * torch.exp(g_last - gc).unsqueeze(-1)
        h = h * torch.exp(g_last).unsqueeze(-1) + k_sc.transpose(-2, -1) @ v_new
    return torch.cat(o_chunks, dim=2)


def gated_deltanet_autograd_bwd_torch(do, q, k, v, g, beta, chunk_size):
    """Compute backward gradients via autograd on the differentiable forward."""
    q_ = q.float().detach().requires_grad_(True)
    k_ = k.float().detach().requires_grad_(True)
    v_ = v.float().detach().requires_grad_(True)
    g_ = g.float().detach().requires_grad_(True)
    beta_ = beta.float().detach().requires_grad_(True)

    o = _differentiable_fwd(q_, k_, v_, g_, beta_, chunk_size)
    loss = (o * do.float()).sum()
    dq, dk, dv, dg, dbeta = torch.autograd.grad(loss, [q_, k_, v_, g_, beta_])
    return dq, dk, dv, dg, dbeta

# =============================================================================
# Autograd-based reference: differentiable forward → torch.autograd.grad
# =============================================================================


# =============================================================================
# Backward correctness tests
# =============================================================================

def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-2, "rtol": 1e-2}
    elif dtype == torch.float16:
        return {"atol": 5e-2, "rtol": 5e-2}
    else:  # bfloat16 — wider tolerance due to compounding chunk-boundary
        # rounding in bf16 (7-bit mantissa); validated against FLA at 0.998+ cosine.
        return {"atol": 1e-1, "rtol": 1e-1}


class GatedDeltaNetBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(2, 64, 2, 64, 64, 32, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(2, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(2, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.smoke),
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

    # Forward to get S for backward kernel
    from tileops.ops import GatedDeltaNetFwdOp
    fwd_op = GatedDeltaNetFwdOp(B, H, S, DK, DV, BC, dtype)
    _o, S_fwd, _Aw, _Au = fwd_op.forward(q, k, v, g, beta)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    # Reference via autograd
    ref_dq, ref_dk, ref_dv, ref_dg, ref_dbeta = gated_deltanet_autograd_bwd_torch(do, q, k, v, g, beta, BC)
    ref_outputs = (ref_dq, ref_dk, ref_dv, ref_dg, ref_dbeta)

    # Kernel
    op = GatedDeltaNetBwdOp(B, H, S, DK, DV, BC, dtype, tune=tune)
    op_outputs = op.forward(do, q, k, v, g, beta, S_fwd)

    tols = _get_tolerances(dtype)
    names = ["dq", "dk", "dv", "dg", "dbeta"]
    for name, ref_out, op_out in zip(names, ref_outputs, op_outputs, strict=True):
        torch.testing.assert_close(
            op_out, ref_out.to(dtype), **tols,
            msg=lambda m, n=name: f"{n}: {m}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
