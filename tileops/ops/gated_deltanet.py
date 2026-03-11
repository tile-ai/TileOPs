from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.linear_attn import (
    GatedDeltaNetBwdKernel,
    GatedDeltaNetFwdKernel,
    PrepareWYReprKernel,
)

from .op import Op

__all__ = ["GatedDeltaNetFwdOp", "GatedDeltaNetBwdOp"]


class GatedDeltaNetFwdOp(Op):
    """Gated DeltaNet forward operator.

    Pipeline: prepare_wy_repr(k, g, beta) -> (Aw, Au) -> gated_deltanet_fwd(q, k, v, g, beta, Aw, Au) -> o.

    Layout: BHSD (batch, head, seq_len, dim).

    Args:
        batch: Batch size.
        heads: Number of attention heads.
        seq_len: Sequence length (must be divisible by chunk_size).
        dim_k: Key/query dimension.
        dim_v: Value dimension.
        chunk_size: Chunk size for chunked linear attention.
        dtype: Data type for computation.
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 32,
        dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype

        assert seq_len % chunk_size == 0, (
            f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
        )

        self.dispatch_kernel(kernel_map)

        wy_kernel_cls = self.kernel_map["PrepareWYReprKernel"]
        fwd_kernel_cls = self.kernel_map["GatedDeltaNetFwdKernel"]

        # Kernels always run in float32; inputs are cast in forward().
        kernel_dtype = Kernel.dtype_to_str(torch.float32)
        self.wy_kernel = wy_kernel_cls(
            batch, heads, seq_len, chunk_size, dim_k,
            dtype=kernel_dtype,
            tune=tune,
        )
        self.kernel = fwd_kernel_cls(
            batch, heads, seq_len, chunk_size, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "PrepareWYReprKernel": PrepareWYReprKernel,
            "GatedDeltaNetFwdKernel": GatedDeltaNetFwdKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Run gated deltanet forward.

        Args:
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            g: Gate tensor [B, H, S].
            beta: Beta tensor [B, H, S].

        Returns:
            Output tensor [B, H, S, DV].
        """
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        g, beta = g.float(), beta.float()
        Aw, Au = self.wy_kernel(k, g, beta)
        o = self.kernel(q, k, v, g, beta, Aw, Au)
        return o.to(input_dtype)


class GatedDeltaNetBwdOp(Op):
    """Gated DeltaNet backward operator.

    Pipeline: prepare_wy_repr -> fwd (to get Aw, Au) -> bwd kernel -> (dq, dk, dv, dg, dbeta).

    Args:
        batch: Batch size.
        heads: Number of attention heads.
        seq_len: Sequence length (must be divisible by chunk_size).
        dim_k: Key/query dimension.
        dim_v: Value dimension.
        chunk_size: Chunk size for chunked linear attention.
        dtype: Data type for computation.
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 32,
        dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype

        assert seq_len % chunk_size == 0, f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"

        self.dispatch_kernel(kernel_map)

        wy_kernel_cls = self.kernel_map["PrepareWYReprKernel"]
        bwd_kernel_cls = self.kernel_map["GatedDeltaNetBwdKernel"]

        # Kernels always run in float32; inputs are cast in forward().
        kernel_dtype = Kernel.dtype_to_str(torch.float32)
        self.wy_kernel = wy_kernel_cls(
            batch, heads, seq_len, chunk_size, dim_k,
            dtype=kernel_dtype,
            tune=tune,
        )
        self.kernel = bwd_kernel_cls(
            batch, heads, seq_len, chunk_size, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "PrepareWYReprKernel": PrepareWYReprKernel,
            "GatedDeltaNetBwdKernel": GatedDeltaNetBwdKernel,
        }

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run gated deltanet backward.

        Args:
            do: Gradient of output [B, H, S, DV].
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            g: Gate tensor [B, H, S].
            beta: Beta tensor [B, H, S].

        Returns:
            Tuple of (dq, dk, dv, dg, dbeta).
        """
        input_dtype = q.dtype
        do, q, k, v = do.float(), q.float(), k.float(), v.float()
        g, beta = g.float(), beta.float()
        Aw, Au = self.wy_kernel(k, g, beta)
        dq, dk, dv, dg, dbeta = self.kernel(do, q, k, v, g, beta, Aw, Au)
        return dq.to(input_dtype), dk.to(input_dtype), dv.to(input_dtype), dg.to(input_dtype), dbeta.to(input_dtype)
