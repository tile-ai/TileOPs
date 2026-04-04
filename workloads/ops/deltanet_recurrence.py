from typing import Tuple

import torch

from workloads.base import WorkloadBase


def deltanet_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step delta rule (ungated).

    Args:
        q: [B, H, DK]
        k: [B, H, DK]
        v: [B, H, DV]
        beta: [B, H]    (writing strength)
        state: [B, H, DK, DV]

    Returns:
        o: [B, H, DV]
        new_state: [B, H, DK, DV]
    """
    q, k, v = q.float(), k.float(), v.float()
    beta = beta.float()
    state = state.float()

    # Step 1: old_val = state @ k  ->  [B, H, DV]
    old_val = torch.einsum("bhkv,bhk->bhv", state, k)

    # Step 2: v_new = beta * (v - old_val)  ->  [B, H, DV]
    beta_unsq = beta.unsqueeze(-1)   # [B, H, 1]
    v_new = beta_unsq * (v - old_val)

    # Step 3: output
    # o = state @ q + (q . k) * v_new  ->  [B, H, DV]
    o_inter = torch.einsum("bhkv,bhk->bhv", state, q)
    qk_dot = torch.einsum("bhk,bhk->bh", q, k).unsqueeze(-1)  # [B, H, 1]
    o_intra = qk_dot * v_new
    o = o_inter + o_intra

    # Step 4: new_state = state + outer(k, v_new)
    new_state = state + k.unsqueeze(-1) * v_new.unsqueeze(-2)

    return o, new_state

class DeltaNetDecodeTest(WorkloadBase):

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
        beta = torch.rand(B, H, device="cuda", dtype=self.dtype) * 0.5
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=self.dtype) * 0.1
        return q, k, v, beta, state
