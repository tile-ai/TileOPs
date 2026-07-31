import torch
import torch.nn.functional as F


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
    """Pure-PyTorch reference for the Mamba-2 SSD forward pass.

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
