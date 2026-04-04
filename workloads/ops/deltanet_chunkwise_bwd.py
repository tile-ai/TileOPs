from typing import Tuple

import torch


def _differentiable_fwd(q, k, v, beta, chunk_size):
    """Fully differentiable chunked forward matching DeltaNet (ungated).

    No gate parameter g, no cumsum, no exp scaling.
    """
    B, H, S, DK = q.shape
    DV = v.shape[-1]
    BC = chunk_size
    NC = S // BC
    h = q.new_zeros(B, H, DK, DV)
    o_chunks = []
    eye = torch.eye(BC, device=q.device, dtype=torch.float32)
    mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.float32))
    for c in range(NC):
        sl = slice(c * BC, (c + 1) * BC)
        qc = q[:, :, sl, :].float()
        kc = k[:, :, sl, :].float()
        vc = v[:, :, sl, :].float()
        bc = beta[:, :, sl].float()
        # WY: A = I + strictLower(diag(beta) * KK^T) (no Gamma)
        Gram = torch.einsum("bhik,bhjk->bhij", kc, kc)
        M = bc.unsqueeze(-1) * Gram
        A = eye + torch.tril(M, diagonal=-1)
        A_inv = torch.linalg.inv(A)
        wc = A_inv @ (kc * bc.unsqueeze(-1))
        uc = A_inv @ (vc * bc.unsqueeze(-1))
        # v_new = u - w @ h (no exp scaling)
        v_new = uc - wc @ h
        # output = q @ h + causal(q @ k^T) @ v_new (no exp scaling)
        o_part = qc @ h
        attn = (qc @ kc.transpose(-2, -1)) * mask
        o_c = o_part + attn @ v_new
        o_chunks.append(o_c)
        # state update: h = h + k^T @ v_new (no decay)
        h = h + kc.transpose(-2, -1) @ v_new
    return torch.cat(o_chunks, dim=2)


def _autograd_bwd_ref(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute backward gradients via autograd on the differentiable forward."""
    q_ = q.float().detach().requires_grad_(True)
    k_ = k.float().detach().requires_grad_(True)
    v_ = v.float().detach().requires_grad_(True)
    beta_ = beta.float().detach().requires_grad_(True)

    o = _differentiable_fwd(q_, k_, v_, beta_, chunk_size)
    loss = (o * do.float()).sum()
    dq, dk, dv, dbeta = torch.autograd.grad(loss, [q_, k_, v_, beta_])
    return dq, dk, dv, dbeta
