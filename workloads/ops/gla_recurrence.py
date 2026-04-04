from typing import Tuple

import torch

from workloads.base import WorkloadBase


def _gla_decode_torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    state: torch.Tensor,
    scale: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step GLA recurrence.

    Args:
        q: [B, H, DK]
        k: [B, H, DK]
        v: [B, H, DV]
        gk: [B, H, DK]   (log-space per-key gate)
        state: [B, H, DK, DV]
        scale: query scale factor (default: DK^{-0.5})

    Returns:
        o: [B, H, DV]
        new_state: [B, H, DK, DV]
    """
    DK = q.shape[-1]
    if scale <= 0:
        scale = DK ** -0.5

    q, k, v = q.float(), k.float(), v.float()
    gk = gk.float()
    state = state.float()

    alpha = torch.exp(gk)  # [B, H, DK]

    # State update: new_state[dk, dv] = alpha[dk] * state[dk, dv] + k[dk] * v[dv]
    new_state = alpha.unsqueeze(-1) * state + k.unsqueeze(-1) * v.unsqueeze(-2)

    # Output: o = scale * q^T @ new_state
    o = scale * torch.einsum("bhk,bhkv->bhv", q, new_state)

    return o, new_state


class GLADecodeTest(WorkloadBase):

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        scale: float = -1.0,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.scale = scale

    def gen_inputs(self) -> Tuple[torch.Tensor, ...]:
        B, H, DK, DV = self.batch, self.heads, self.dim_k, self.dim_v
        q = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=self.dtype) * 0.1
        gk = -torch.rand(B, H, DK, device="cuda", dtype=self.dtype)
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=self.dtype) * 0.1
        return q, k, v, gk, state

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        o, new_state = _gla_decode_torch_ref(q, k, v, gk, state, self.scale)
        return o.to(self.dtype), new_state.to(self.dtype)
