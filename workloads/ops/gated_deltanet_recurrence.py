from typing import Tuple

import torch

from workloads.base import WorkloadBase


def _gated_deltanet_decode_torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step gated delta rule.

    Args:
        q: [B, H, DK]
        k: [B, H, DK]
        v: [B, H, DV]
        g: [B, H]       (log-space gate)
        beta: [B, H]    (writing strength)
        state: [B, H, DK, DV]

    Returns:
        o: [B, H, DV]
        new_state: [B, H, DK, DV]
    """
    q, k, v = q.float(), k.float(), v.float()
    g, beta = g.float(), beta.float()
    state = state.float()

    alpha = torch.exp(g)  # [B, H]

    # Step 1: old_val = state @ k  ->  [B, H, DV]
    # state: [B,H,DK,DV], k: [B,H,DK] -> einsum "bhkv,bhk->bhv"
    old_val = torch.einsum("bhkv,bhk->bhv", state, k)

    # Step 2: v_new = beta * v - alpha * beta * old_val  ->  [B, H, DV]
    beta_unsq = beta.unsqueeze(-1)   # [B, H, 1]
    alpha_unsq = alpha.unsqueeze(-1)  # [B, H, 1]
    v_new = beta_unsq * v - alpha_unsq * beta_unsq * old_val

    # Step 3: output
    # o_inter = alpha * (state @ q)  ->  [B, H, DV]
    o_inter = alpha_unsq * torch.einsum("bhkv,bhk->bhv", state, q)
    # o_intra = (q . k) * v_new  ->  [B, H, DV]
    qk_dot = torch.einsum("bhk,bhk->bh", q, k).unsqueeze(-1)  # [B, H, 1]
    o_intra = qk_dot * v_new
    o = o_inter + o_intra

    # Step 4: new_state = alpha * state + outer(k, v_new)
    # outer(k, v_new): [B,H,DK,1] * [B,H,1,DV] -> [B,H,DK,DV]
    new_state = alpha_unsq.unsqueeze(-1) * state + k.unsqueeze(-1) * v_new.unsqueeze(-2)

    return o, new_state


class GatedDeltaNetDecodeTest(WorkloadBase):

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, ...]:
        B, H, DK, DV = self.batch, self.heads, self.dim_k, self.dim_v
        q = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=self.dtype) * 0.1
        g = -torch.rand(B, H, device="cuda", dtype=self.dtype)
        beta = torch.rand(B, H, device="cuda", dtype=self.dtype) * 0.5
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=self.dtype) * 0.1
        return q, k, v, g, beta, state

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        o, new_state = _gated_deltanet_decode_torch_ref(q, k, v, g, beta, state)
        return o.to(self.dtype), new_state.to(self.dtype)
