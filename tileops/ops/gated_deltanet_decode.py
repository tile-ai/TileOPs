from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.linear_attn import GatedDeltaNetDecodeKernel

from .op import Op

__all__ = ["GatedDeltaNetDecodeOp"]


class GatedDeltaNetDecodeOp(Op):
    """Gated DeltaNet decode (single-step recurrence).

    Computes one step of the gated delta rule:
        S_t = S_{t-1} (alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T
        o_t = S_t q_t

    Layout: BHD (batch, head, dim).

    Args:
        batch: Batch size.
        heads: Number of attention heads.
        dim_k: Key/query dimension.
        dim_v: Value dimension.
        dtype: Data type for computation.
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)

        kernel_cls = self.kernel_map["GatedDeltaNetDecodeKernel"]
        kernel_dtype = Kernel.dtype_to_str(torch.float32)
        self.kernel = kernel_cls(
            batch, heads, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GatedDeltaNetDecodeKernel": GatedDeltaNetDecodeKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run gated deltanet single-step decode.

        Args:
            q: Query tensor [B, H, DK].
            k: Key tensor [B, H, DK].
            v: Value tensor [B, H, DV].
            g: Gate tensor [B, H] (log-space, alpha = exp(g)).
            beta: Beta tensor [B, H] (writing strength).
            state: Hidden state [B, H, DK, DV] (S_{t-1}).

        Returns:
            Tuple of (o, new_state):
                o: Output tensor [B, H, DV].
                new_state: Updated hidden state [B, H, DK, DV].
        """
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        g, beta = g.float(), beta.float()
        state = state.float()
        o, new_state = self.kernel(q, k, v, g, beta, state)
        return o.to(input_dtype), new_state.to(input_dtype)
