from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.linear_attn import (
    GatedDeltaNetBwdKernel,
    GatedDeltaNetFwdKernel,
)

from .op import Op

__all__ = ["GatedDeltaNetFwdOp", "GatedDeltaNetBwdOp", "GatedDeltaNetOp"]


class GatedDeltaNetFwdOp(Op):
    """Gated DeltaNet forward operator.

    Pipeline: prepare_wy_repr(k, g, beta) -> (Aw, Au) -> gated_deltanet_fwd(q, k, v, g, beta, Aw, Au) -> o.

    Layout: BHSD (batch, head, seq_len, dim).

    .. note:: Layout convention difference with FLA

        TileOPs uses **BHSD** layout: ``q/k [B, H, S, DK]``, ``v [B, H, S, DV]``,
        ``g/beta [B, H, S]``.

        FLA (``fla.ops.gated_delta_rule.chunk_gated_delta_rule``) uses **BTHN**
        layout: ``q/k [B, T, H, K]``, ``v [B, T, H, V]``, ``g/beta [B, T, H]``.

        When comparing against FLA, tensors must be transposed::

            # TileOPs BHSD -> FLA BTHK
            q_fla = q.permute(0, 2, 1, 3)   # [B, H, S, DK] -> [B, S, H, DK]
            g_fla = g.permute(0, 2, 1)       # [B, H, S]     -> [B, S, H]

            # FLA BTHV -> TileOPs BHSD
            o_tileops = o_fla.permute(0, 2, 1, 3)  # [B, S, H, DV] -> [B, H, S, DV]

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

        assert seq_len % chunk_size == 0, (
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

        assert seq_len % chunk_size == 0, f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"

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


class _GatedDeltaNetFunction(torch.autograd.Function):
    """Autograd function wrapping TileOPs fwd + bwd kernels."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, fwd_kernel, bwd_kernel):
        o, S, Aw, Au = fwd_kernel(q, k, v, g, beta)
        ctx.save_for_backward(q, k, v, g, beta, S)
        ctx.bwd_kernel = bwd_kernel
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, S = ctx.saved_tensors
        dq, dk, dv, dg, dbeta = ctx.bwd_kernel(do, q, k, v, g, beta, S)
        return dq, dk, dv, dg, dbeta, None, None


class GatedDeltaNetOp(Op):
    """Combined Gated DeltaNet fwd+bwd operator with autograd support.

    Wraps ``GatedDeltaNetFwdKernel`` and ``GatedDeltaNetBwdKernel`` in a
    ``torch.autograd.Function`` so that ``output.backward(do)`` automatically
    invokes the TileOPs backward kernels.

    This makes end-to-end benchmarking against FLA straightforward::

        op = GatedDeltaNetOp(B, H, S, DK, DV, chunk_size, dtype)
        o = op(q, k, v, g, beta)   # forward
        o.backward(do)              # backward via TileOPs kernels

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

        assert seq_len % chunk_size == 0, (
            f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
        )

        self.dispatch_kernel(kernel_map)

        kernel_dtype = Kernel.dtype_to_str(dtype)
        fwd_cls = self.kernel_map["GatedDeltaNetFwdKernel"]
        bwd_cls = self.kernel_map["GatedDeltaNetBwdKernel"]
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
            "GatedDeltaNetFwdKernel": GatedDeltaNetFwdKernel,
            "GatedDeltaNetBwdKernel": GatedDeltaNetBwdKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Run gated deltanet forward with autograd backward support.

        Args:
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            g: Gate tensor [B, H, S].
            beta: Beta tensor [B, H, S].

        Returns:
            Output tensor o [B, H, S, DV] (supports .backward()).
        """
        return _GatedDeltaNetFunction.apply(
            q, k, v, g, beta, self.fwd_kernel, self.bwd_kernel,
        )
