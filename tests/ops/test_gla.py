"""Correctness unit tests for GLAFwdOp.

Reference:
    https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.ops import GLAFwdOp


def ref_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for GLA chunked forward.

    Implements the same 4-stage algorithm as the TileLang kernel:
      1. Within-chunk cumulative sum of log-space gates
      2. Inter-chunk hidden state recurrence with gated decay
      3. Intra-chunk causal attention matrix
      4. Output = inter-chunk (q*exp(g_cs) @ h) + intra-chunk (A @ v)

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H, K]  log-space gates
        scale: query scale factor
        initial_state: [B, H, K, V] float32, optional
        chunk_size: BT

    Returns:
        (o [B, T, H, V], final_state [B, H, K, V] float32)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = (T + chunk_size - 1) // chunk_size

    # work in float32 for numerical stability
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()

    # Stage 1: within-chunk cumulative sum of gates
    # g_cs[b, t, h, k] = sum of g[b, chunk_start..t, h, k]
    g_cs = torch.zeros_like(g)
    for i_c in range(NT):
        cs = i_c * chunk_size
        ce = min(cs + chunk_size, T)
        g_cs[:, cs:ce] = torch.cumsum(g[:, cs:ce], dim=1)

    # Stage 2: inter-chunk hidden state recurrence
    # h[b, i_c, h, K, V] = state entering chunk i_c
    h_states = torch.zeros(B, NT, H, K, V, dtype=torch.float32, device=q.device)
    b_h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        b_h = initial_state.float().clone()

    for i_c in range(NT):
        cs = i_c * chunk_size
        ce = min(cs + chunk_size, T)
        h_states[:, i_c] = b_h

        # g_last: cumsum at last position of this chunk [B, H, K]
        g_last = g_cs[:, ce - 1]  # [B, H, K]

        # decay existing state
        # b_h[b, h, k, v] *= exp(g_last[b, h, k])
        b_h = b_h * torch.exp(g_last).unsqueeze(-1)  # [B, H, K, V]

        # accumulate: sum over t in chunk of k_adj[t]^T @ v[t]
        # k_adj[b, t, h, k] = k[b, t, h, k] * exp(g_last[b, h, k] - g_cs[b, t, h, k])
        k_chunk = k[:, cs:ce]  # [B, L, H, K]
        v_chunk = v[:, cs:ce]  # [B, L, H, V]
        g_cs_chunk = g_cs[:, cs:ce]  # [B, L, H, K]
        g_last_exp = torch.exp(g_last).unsqueeze(1)  # [B, 1, H, K]
        k_adj = k_chunk * (g_last_exp / torch.exp(g_cs_chunk).clamp(min=1e-30))
        # b_h += einsum('blhk,blhv->bhkv', k_adj, v_chunk)
        b_h = b_h + torch.einsum('blhk,blhv->bhkv', k_adj, v_chunk)

    final_state = b_h  # [B, H, K, V]

    # Stage 3 + 4: intra-chunk attention and output
    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)

    for i_c in range(NT):
        cs = i_c * chunk_size
        ce = min(cs + chunk_size, T)
        L = ce - cs

        q_c = q[:, cs:ce]  # [B, L, H, K]
        k_c = k[:, cs:ce]  # [B, L, H, K]
        v_c = v[:, cs:ce]  # [B, L, H, V]
        g_cs_c = g_cs[:, cs:ce]  # [B, L, H, K]
        h_c = h_states[:, i_c]  # [B, H, K, V]

        # intra-chunk attention matrix A[b, i, h, j] = scale * sum_k(
        #   q[i,k]*exp(g_cs[i,k]) * k[j,k]*exp(-g_cs[j,k]) ), causal
        q_gated = q_c * torch.exp(g_cs_c)  # [B, L, H, K]
        k_gated = k_c * torch.exp(-g_cs_c)  # [B, L, H, K]
        # A[b, h, i, j] = scale * q_gated[b,i,h,:] @ k_gated[b,j,h,:]^T
        # rearrange to [B, H, L, K] for bmm
        qg = q_gated.permute(0, 2, 1, 3)  # [B, H, L, K]
        kg = k_gated.permute(0, 2, 1, 3)  # [B, H, L, K]
        A = scale * torch.bmm(qg.reshape(B * H, L, K),
                              kg.reshape(B * H, L, K).transpose(1, 2)).reshape(B, H, L,
                                                                               L)  # [B, H, L, L]

        # causal mask
        causal_mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
        A = A * causal_mask.unsqueeze(0).unsqueeze(0)  # [B, H, L, L]

        # intra-chunk output: A @ v  [B, H, L, V]
        vc = v_c.permute(0, 2, 1, 3).reshape(B * H, L, V)
        o_intra = torch.bmm(A.reshape(B * H, L, L),
                            vc).reshape(B, H, L, V).permute(0, 2, 1, 3)  # [B, L, H, V]

        # inter-chunk output: scale * (q*exp(g_cs)) @ h  [B, L, H, V]
        # q_gated [B, L, H, K], h_c [B, H, K, V]
        o_inter = scale * torch.einsum('blhk,bhkv->blhv', q_gated, h_c)

        o[:, cs:ce] = o_intra + o_inter

    return o, final_state


@pytest.mark.parametrize(
    "batch, seq_len, heads, dim_k, dim_v, chunk_size, output_final_state, dtype, tune",
    [
        (1, 64, 4, 64, 64, 64, False, torch.float16, False),
        (2, 128, 8, 64, 64, 64, False, torch.bfloat16, False),
        (1, 256, 4, 128, 128, 64, False, torch.float16, False),
        (2, 128, 8, 64, 128, 32, False, torch.bfloat16, False),
        (4, 256, 16, 64, 64, 64, False, torch.float16, True),
        (1, 64, 4, 64, 64, 64, True, torch.float16, False),  # with initial_state + final_state
        (2, 128, 8, 64, 64, 64, True, torch.bfloat16, False),
    ],
)
def test_gla_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    output_final_state: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)

    scale = dim_k**-0.5

    op = GLAFwdOp(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        dim_k=dim_k,
        dim_v=dim_v,
        chunk_size=chunk_size,
        scale=scale,
        output_final_state=output_final_state,
        dtype=dtype,
        tune=tune,
    )

    q = torch.randn(batch, seq_len, heads, dim_k, device='cuda', dtype=dtype)
    k = torch.randn(batch, seq_len, heads, dim_k, device='cuda', dtype=dtype)
    v = torch.randn(batch, seq_len, heads, dim_v, device='cuda', dtype=dtype)
    g = F.logsigmoid(torch.randn(batch, seq_len, heads, dim_k, device='cuda', dtype=dtype))

    initial_state = None
    if output_final_state:
        initial_state = torch.randn(
            batch, heads, dim_k, dim_v, device='cuda', dtype=torch.float32) * 0.1

    with torch.no_grad():
        out, out_final = op(q, k, v, g, initial_state)

    ref_o, ref_final = ref_gla_fwd(
        q, k, v, g, scale=scale, initial_state=initial_state, chunk_size=chunk_size)
    ref_o = ref_o.to(dtype)

    assert torch.allclose(out, ref_o, atol=1e-2, rtol=1e-2), \
        f"output mismatch: max err = {(out.float() - ref_o.float()).abs().max():.6f}"

    if output_final_state:
        assert out_final is not None
        assert torch.allclose(out_final.float(), ref_final.float(), atol=1e-2, rtol=1e-2), \
            f"final_state mismatch: max err = {(out_final.float() - ref_final.float()).abs().max():.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
