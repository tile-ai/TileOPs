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
        chunk_size: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self.chunk_size = chunk_size
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "DeltaNetFwdKernel": DeltaNetFwdKernel,
        }

    def _get_kernel(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, heads, seq_len, self.chunk_size, dim_k, dim_v, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["DeltaNetFwdKernel"](
                batch,
                heads,
                seq_len,
                self.chunk_size,
                dim_k,
                dim_v,
                dtype=Kernel.dtype_to_str(dtype),
                tune=self.tune,
            )
        return self._kernel_cache[key]

    def _bind_from_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        if not all(tensor.is_cuda for tensor in (q, k, v, beta)):
            raise ValueError("q, k, v, and beta must be CUDA tensors")
        if q.ndim != 4:
            raise ValueError("q must have shape [batch, heads, seq_len, dim_k]")
        batch, heads, seq_len, dim_k = q.shape
        if k.shape != (batch, heads, seq_len, dim_k):
            raise ValueError("k must match q shape")
        if v.ndim != 4 or v.shape[:3] != (batch, heads, seq_len):
            raise ValueError("v must have shape [batch, heads, seq_len, dim_v]")
        if beta.shape != (batch, heads, seq_len):
            raise ValueError("beta must have shape [batch, heads, seq_len]")
        dtype = q.dtype
        for name, tensor in (("k", k), ("v", v), ("beta", beta)):
            if tensor.dtype != dtype:
                raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
        if seq_len % self.chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size ({self.chunk_size})"
            )

        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = v.shape[-1]
        self.dtype = dtype
        self.kernel = self._get_kernel(
            batch, heads, seq_len, dim_k, self.dim_v, dtype, q.device.index)

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
        self._bind_from_inputs(q, k, v, beta)
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
        chunk_size: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self.chunk_size = chunk_size
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "DeltaNetBwdKernel": DeltaNetBwdKernel,
        }

    def _get_kernel(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, heads, seq_len, self.chunk_size, dim_k, dim_v, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["DeltaNetBwdKernel"](
                batch,
                heads,
                seq_len,
                self.chunk_size,
                dim_k,
                dim_v,
                dtype=Kernel.dtype_to_str(dtype),
                tune=self.tune,
            )
        return self._kernel_cache[key]

    def _bind_from_inputs(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        if not all(tensor.is_cuda for tensor in (do, q, k, v, beta)):
            raise ValueError("do, q, k, v, and beta must be CUDA tensors")
        if q.ndim != 4:
            raise ValueError("q must have shape [batch, heads, seq_len, dim_k]")
        batch, heads, seq_len, dim_k = q.shape
        if k.shape != (batch, heads, seq_len, dim_k):
            raise ValueError("k must match q shape")
        if v.ndim != 4 or v.shape[:3] != (batch, heads, seq_len):
            raise ValueError("v must have shape [batch, heads, seq_len, dim_v]")
        dim_v = v.shape[-1]
        if do.shape != (batch, heads, seq_len, dim_v):
            raise ValueError("do must have shape [batch, heads, seq_len, dim_v]")
        if beta.shape != (batch, heads, seq_len):
            raise ValueError("beta must have shape [batch, heads, seq_len]")
        dtype = q.dtype
        for name, tensor in (("do", do), ("k", k), ("v", v), ("beta", beta)):
            if tensor.dtype != dtype:
                raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
        if seq_len % self.chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size ({self.chunk_size})"
            )

        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.kernel = self._get_kernel(
            batch, heads, seq_len, dim_k, dim_v, dtype, q.device.index)

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
        self._bind_from_inputs(do, q, k, v, beta)
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
        chunk_size: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self.chunk_size = chunk_size
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._fwd_kernel_cache: Dict[tuple, Kernel] = {}
        self._bwd_kernel_cache: Dict[tuple, Kernel] = {}
        self.fwd_kernel = None
        self.bwd_kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "DeltaNetFwdKernel": DeltaNetFwdKernel,
            "DeltaNetBwdKernel": DeltaNetBwdKernel,
        }

    def _bind_from_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        if not all(tensor.is_cuda for tensor in (q, k, v, beta)):
            raise ValueError("q, k, v, and beta must be CUDA tensors")
        batch, heads, seq_len, dim_k = q.shape
        if k.shape != (batch, heads, seq_len, dim_k):
            raise ValueError("k must match q shape")
        if v.ndim != 4 or v.shape[:3] != (batch, heads, seq_len):
            raise ValueError("v must have shape [batch, heads, seq_len, dim_v]")
        if beta.shape != (batch, heads, seq_len):
            raise ValueError("beta must have shape [batch, heads, seq_len]")
        dtype = q.dtype
        for name, tensor in (("k", k), ("v", v), ("beta", beta)):
            if tensor.dtype != dtype:
                raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
        if seq_len % self.chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size ({self.chunk_size})"
            )
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = v.shape[-1]
        self.dtype = dtype

        key = (
            batch,
            heads,
            seq_len,
            self.chunk_size,
            dim_k,
            self.dim_v,
            dtype,
            q.device.index,
            self.tune,
        )
        if key not in self._fwd_kernel_cache:
            kernel_dtype = Kernel.dtype_to_str(dtype)
            self._fwd_kernel_cache[key] = self.kernel_map["DeltaNetFwdKernel"](
                batch, heads, seq_len, self.chunk_size, dim_k, self.dim_v,
                dtype=kernel_dtype, tune=self.tune)
            self._bwd_kernel_cache[key] = self.kernel_map["DeltaNetBwdKernel"](
                batch, heads, seq_len, self.chunk_size, dim_k, self.dim_v,
                dtype=kernel_dtype, tune=self.tune)
        self.fwd_kernel = self._fwd_kernel_cache[key]
        self.bwd_kernel = self._bwd_kernel_cache[key]

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
        self._bind_from_inputs(q, k, v, beta)
        return _DeltaNetFunction.apply(
            q, k, v, beta, self.fwd_kernel, self.bwd_kernel,
        )
