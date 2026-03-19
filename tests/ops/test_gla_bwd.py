from typing import Tuple

import pytest
import torch

from tests.ops.gla_test_utils import cosine_sim, get_tolerances
from tests.test_base import FixtureBase
from tileops.ops import GLABwdOp, GLAFwdOp

try:
    from fla.ops.gla import chunk_gla
except ImportError:
    chunk_gla = None

# =============================================================================
# Pure-torch differentiable forward (BTHD layout) for autograd-based reference
# =============================================================================


def _gla_fwd_torch_ref(q, k, v, g, chunk_size, scale=None):
    """Fully differentiable chunked GLA forward in float32.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H, K] — log-space forget gates (per head per dim_k)
        chunk_size: int
        scale: float or None (defaults to dim_k**-0.5)

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BC = chunk_size
    NC = T // BC

    if scale is None:
        scale = K ** -0.5

    q = q.float() * scale
    k = k.float()
    v = v.float()
    g = g.float()

    # Intra-chunk cumulative sum of g: [B, T, H, K]
    g_cum = g.reshape(B, NC, BC, H, K).cumsum(dim=2).reshape(B, T, H, K)

    # h: hidden state [B, H, K, V]
    h = q.new_zeros(B, H, K, V)
    mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.float32))

    o_chunks = []
    for c in range(NC):
        sl = slice(c * BC, (c + 1) * BC)
        qc = q[:, sl, :, :]  # [B, BC, H, K]
        kc = k[:, sl, :, :]
        vc = v[:, sl, :, :]
        gc = g_cum[:, sl, :, :]  # [B, BC, H, K]
        g_last = gc[:, -1:, :, :]  # [B, 1, H, K]

        # Inter-chunk: o_inter = sum_k (q * exp(g_cum)) @ h
        q_gated = qc * torch.exp(gc)  # [B, BC, H, K]
        o_inter = torch.einsum("bthk,bhkv->bthv", q_gated, h)

        # Intra-chunk: causal attention with gating
        k_ungated = kc * torch.exp(-gc)
        A = torch.einsum("bihk,bjhk->bhij", q_gated, k_ungated)
        A = A * mask.unsqueeze(0).unsqueeze(0)
        o_intra = torch.einsum("bhij,bjhv->bihv", A, vc)

        o_chunks.append(o_inter + o_intra)

        # State update: h = h * exp(g_last) + k_adj^T @ v
        k_adj = kc * torch.exp(g_last - gc)
        h = h * torch.exp(g_last).permute(0, 2, 3, 1).squeeze(-1).unsqueeze(-1)
        h = h + torch.einsum("bthk,bthv->bhkv", k_adj, vc)

    return torch.cat(o_chunks, dim=1)


def _gla_autograd_bwd_ref(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
    scale: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GLA backward gradients via autograd on the differentiable forward.

    Returns:
        (dq, dk, dv, dg) all in float32.
    """
    sc = (q.shape[-1] ** -0.5) if scale <= 0 else scale

    q_ = q.float().detach().requires_grad_(True)
    k_ = k.float().detach().requires_grad_(True)
    v_ = v.float().detach().requires_grad_(True)
    g_ = g.float().detach().requires_grad_(True)

    o = _gla_fwd_torch_ref(q_, k_, v_, g_, chunk_size, scale=sc)
    loss = (o * do.float()).sum()
    dq, dk, dv, dg = torch.autograd.grad(loss, [q_, k_, v_, g_])
    return dq, dk, dv, dg


def _fla_autograd_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GLA backward gradients via FLA's chunk_gla + autograd.

    FLA uses the same BTHD layout and g shape [B, T, H, K] as TileOPs.

    Returns:
        (dq, dk, dv, dg) all in float32.
    """
    q_ = q.float().detach().requires_grad_(True)
    k_ = k.float().detach().requires_grad_(True)
    v_ = v.float().detach().requires_grad_(True)
    g_ = g.float().detach().requires_grad_(True)

    o, _ = chunk_gla(q_, k_, v_, g_, scale=scale)
    loss = (o * do.float()).sum()
    dq, dk, dv, dg = torch.autograd.grad(loss, [q_, k_, v_, g_])
    return dq, dk, dv, dg


# =============================================================================
# Backward correctness tests
# =============================================================================


class GLABwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(2, 64, 2, 64, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(1, 128, 4, 64, 64, 64, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@GLABwdFixture
def test_gla_bwd(
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
    B, T, H, K, V, BC = batch, seq_len, heads, dim_k, dim_v, chunk_size

    # GLA layout: BTHD — [B, T, H, K/V]
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype)
    do = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1

    scale = K ** -0.5

    # --- Torch reference via autograd ---
    ref_dq, ref_dk, ref_dv, ref_dg = _gla_autograd_bwd_ref(
        do, q, k, v, g, BC, scale=scale
    )
    ref_grads = {"dq": ref_dq, "dk": ref_dk, "dv": ref_dv, "dg": ref_dg}

    # --- FLA reference via autograd (if available) ---
    if chunk_gla is not None:
        fla_dq, fla_dk, fla_dv, fla_dg = _fla_autograd_bwd(
            do, q, k, v, g, scale=scale
        )
        fla_grads = {"dq": fla_dq, "dk": fla_dk, "dv": fla_dv, "dg": fla_dg}

        # Validate FLA vs torch reference alignment
        tols = get_tolerances(torch.float32)
        for name in ["dq", "dk", "dv", "dg"]:
            cos = cosine_sim(ref_grads[name], fla_grads[name])
            print(f"  FLA vs ref {name}: cosine={cos:.6f}")
            assert cos > 0.99, f"FLA vs ref {name} cosine too low: {cos:.6f}"

    # --- TileOPs kernel backward ---
    fwd_op = GLAFwdOp(B, T, H, K, V, BC, scale=scale,
                       output_final_state=False, dtype=dtype)
    o_fwd, _ = fwd_op.forward(q, k, v, g)
    h = fwd_op.kernel._h_out  # [B, NT+1, H, K, V] in fp32

    dht = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)
    bwd_op = GLABwdOp(B, T, H, K, V, BC, scale=scale, dtype=dtype, tune=tune)
    op_dq, op_dk, op_dv, op_dg = bwd_op.forward(q, k, v, g, h, do, dht)
    op_grads = {"dq": op_dq, "dk": op_dk, "dv": op_dv, "dg": op_dg}

    # Validate TileOPs vs torch reference
    tols = get_tolerances(dtype)
    for name in ["dq", "dk", "dv", "dg"]:
        cos = cosine_sim(ref_grads[name], op_grads[name])
        print(f"  TileOPs vs ref {name}: cosine={cos:.6f}")
        torch.testing.assert_close(
            op_grads[name].float(), ref_grads[name].float(), **tols,
            msg=lambda m, n=name: f"{n}: {m}",
        )

    # Validate TileOPs vs FLA (if available)
    if chunk_gla is not None:
        for name in ["dq", "dk", "dv", "dg"]:
            cos = cosine_sim(fla_grads[name], op_grads[name])
            print(f"  TileOPs vs FLA {name}: cosine={cos:.6f}")
            assert cos > 0.99, f"TileOPs vs FLA {name} cosine too low: {cos:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
