from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.linear_attn import (
    GatedDeltaNetBwdKernel,
    GatedDeltaNetFwdKernel,
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

        if seq_len % chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
            )

        self.dispatch_kernel(kernel_map)

        fwd_kernel_cls = self.kernel_map["GatedDeltaNetFwdKernel"]

        kernel_dtype = Kernel.dtype_to_str(dtype)
        self.kernel = fwd_kernel_cls(
            batch, heads, seq_len, chunk_size, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
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
            Tuple of (o, S, Aw, Au).
        """
        o, S, Aw, Au = self.kernel(q, k, v, g, beta)
        return o, S, Aw, Au


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

        if seq_len % chunk_size != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})")

        self.dispatch_kernel(kernel_map)

        bwd_kernel_cls = self.kernel_map["GatedDeltaNetBwdKernel"]

        kernel_dtype = Kernel.dtype_to_str(dtype)
        self.kernel = bwd_kernel_cls(
            batch, heads, seq_len, chunk_size, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
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
        S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run gated deltanet backward.

        Args:
            do: Gradient of output [B, H, S, DV].
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            g: Gate tensor [B, H, S].
            beta: Beta tensor [B, H, S].
            S: Per-chunk boundary states from forward [B, H, NC+1, DK, DV].

        Returns:
            Tuple of (dq, dk, dv, dg, dbeta).
        """
        dq, dk, dv, dg, dbeta = self.kernel(do, q, k, v, g, beta, S)
        return dq, dk, dv, dg, dbeta
