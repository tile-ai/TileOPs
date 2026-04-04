from typing import Tuple

import torch


def _differentiable_fwd(q, k, v, g_raw, beta, chunk_size):
    """Fully differentiable chunked forward matching paper (Eq. 10 via WY).

    Inputs use raw per-step g (cumsum is computed inside).
    """
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
        # WY: A_g = I + strictLower(diag(β) · (Γ ⊙ KK^T))
        Gram = torch.einsum("bhik,bhjk->bhij", kc, kc)
        Gamma = torch.exp(gc.unsqueeze(-1) - gc.unsqueeze(-2))
        M = bc.unsqueeze(-1) * (Gamma * Gram)
        A = eye + torch.tril(M, diagonal=-1)
        A_inv = torch.linalg.inv(A)
        wc = A_inv @ (kc * bc.unsqueeze(-1))
        uc = A_inv @ (vc * bc.unsqueeze(-1))
        g_last = gc[:, :, -1:]
        # v_new = u - w * exp(g + g_last) @ h
        v_new = uc - (wc * torch.exp(gc + g_last).unsqueeze(-1)) @ h
        # output = q @ h * exp(g) + causal(Γ * q @ k^T) @ v_new
        o_part = (qc @ h) * torch.exp(gc).unsqueeze(-1)
        attn = (qc @ kc.transpose(-2, -1)) * Gamma * mask
        o_c = o_part + attn @ v_new
        o_chunks.append(o_c)
        # state update
        k_sc = kc * torch.exp(g_last - gc).unsqueeze(-1)
        h = h * torch.exp(g_last).unsqueeze(-1) + k_sc.transpose(-2, -1) @ v_new
    return torch.cat(o_chunks, dim=2)


def _autograd_bwd_ref(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
