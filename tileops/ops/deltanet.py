from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.deltanet import (
    DeltaNetBwdKernel,
    DeltaNetFwdKernel,
)
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["DeltaNetBwdOp", "DeltaNetFwdOp", "DeltaNetOp"]


class DeltaNetFwdOp(Op):
    """DeltaNet forward operator (ungated).

    Pipeline: prepare_wy_repr(k, beta) -> (Aw, Au) -> deltanet_fwd(q, k, v, beta, Aw, Au) -> o.

    Layout: BHSD (batch, head, seq_len, dim).

    .. note:: Layout convention difference with FLA

        TileOPs uses **BHSD** layout: ``q/k [B, H, S, DK]``, ``v [B, H, S, DV]``,
        ``beta [B, H, S]``.

        FLA (``fla.ops.delta_rule.chunk_delta_rule``) uses **BTHN**
        layout: ``q/k [B, T, H, K]``, ``v [B, T, H, V]``, ``beta [B, T, H]``.

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
        chunk_size: int = 64,
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

        fwd_kernel_cls = self.kernel_map["DeltaNetFwdKernel"]

        kernel_dtype = Kernel.dtype_to_str(dtype)
        self.kernel = fwd_kernel_cls(
            batch, heads, seq_len, chunk_size, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "DeltaNetFwdKernel": DeltaNetFwdKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Run deltanet forward.

        Args:
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            beta: Beta tensor [B, H, S].

        Returns:
            Tuple of (o, S, Aw, Au).
        """
        o, S, Aw, Au, w, u = self.kernel(q, k, v, beta)
        return o, S, Aw, Au, w, u


class DeltaNetBwdOp(Op):
    """DeltaNet backward operator (ungated).

    Pipeline: prepare_wy_repr -> fwd (to get Aw, Au) -> bwd kernel -> (dq, dk, dv, dbeta).

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
        chunk_size: int = 64,
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

        bwd_kernel_cls = self.kernel_map["DeltaNetBwdKernel"]

        kernel_dtype = Kernel.dtype_to_str(dtype)
        self.kernel = bwd_kernel_cls(
            batch, heads, seq_len, chunk_size, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "DeltaNetBwdKernel": DeltaNetBwdKernel,
        }

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
        Aw: torch.Tensor,
        Au: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run deltanet backward.

        Args:
            do: Gradient of output [B, H, S, DV].
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            beta: Beta tensor [B, H, S].
            S: Per-chunk boundary states from forward [B, H, NC+1, DK, DV].
            Aw: A_inv matrix from forward [B, H, S, BC].
            Au: A_inv matrix from forward [B, H, S, BC].
            w: WY w vectors from forward [B, H, S, DK].
            u: WY u vectors from forward [B, H, S, DV].

        Returns:
            Tuple of (dq, dk, dv, dbeta).
        """
        dq, dk, dv, dbeta = self.kernel(do, q, k, v, beta, S, Aw, Au, w, u)
        return dq, dk, dv, dbeta


class _DeltaNetFunction(torch.autograd.Function):
    """Autograd function wrapping TileOPs fwd + bwd kernels."""

    @staticmethod
    def forward(ctx, q, k, v, beta, fwd_kernel, bwd_kernel):
        o, S, Aw, Au, w, u = fwd_kernel(q, k, v, beta)
        ctx.save_for_backward(q, k, v, beta, S, Aw, Au, w, u)
        ctx.bwd_kernel = bwd_kernel
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, beta, S, Aw, Au, w, u = ctx.saved_tensors
        dq, dk, dv, dbeta = ctx.bwd_kernel(do, q, k, v, beta, S, Aw, Au, w, u)
        return dq, dk, dv, dbeta, None, None


class DeltaNetOp(Op):
    """Combined DeltaNet fwd+bwd operator with autograd support (ungated).

    Wraps ``DeltaNetFwdKernel`` and ``DeltaNetBwdKernel`` in a
    ``torch.autograd.Function`` so that ``output.backward(do)`` automatically
    invokes the TileOPs backward kernels.

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
        chunk_size: int = 64,
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

        kernel_dtype = Kernel.dtype_to_str(dtype)
        fwd_cls = self.kernel_map["DeltaNetFwdKernel"]
        bwd_cls = self.kernel_map["DeltaNetBwdKernel"]
        self.fwd_kernel = fwd_cls(
            batch, heads, seq_len, chunk_size, dim_k, dim_v,
            dtype=kernel_dtype, tune=tune,
        )
        self.bwd_kernel = bwd_cls(
            batch, heads, seq_len, chunk_size, dim_k, dim_v,
            dtype=kernel_dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "DeltaNetFwdKernel": DeltaNetFwdKernel,
            "DeltaNetBwdKernel": DeltaNetBwdKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Run deltanet forward with autograd backward support.

        Args:
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            beta: Beta tensor [B, H, S].

        Returns:
            Output tensor o [B, H, S, DV] (supports .backward()).
        """
        return _DeltaNetFunction.apply(
            q, k, v, beta, self.fwd_kernel, self.bwd_kernel,
        )
