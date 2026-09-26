from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.deltanet import (
    DeltaNetBwdKernel,
    DeltaNetFwdKernel,
)
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["DeltaNetAutogradFwdOp", "DeltaNetBwdOp", "DeltaNetFwdOp"]


class DeltaNetFwdOp(Op):
    """DeltaNet forward operator (ungated).

    Pipeline: prepare_wy_repr(k, beta) -> (Aw, Au) -> deltanet_fwd(q, k, v, beta, Aw, Au) -> o.

    Layout: BHSD (batch, head, seq_len, dim).

    !!! note "Layout convention difference with FLA"

        TileOPs uses **BHSD** layout: ``q/k [B, H, S, DK]``, ``v [B, H, S, DV]``,
        ``beta [B, H, S]``.

        FLA (``fla.ops.delta_rule.chunk_delta_rule``) uses **BTHN**
        layout: ``q/k [B, T, H, K]``, ``v [B, T, H, V]``, ``beta [B, T, H]``.

    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"DeltaNetFwdKernel": DeltaNetFwdKernel}

    def __init__(
        self,
        chunk_size: int = 64,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            chunk_size: Chunk size for chunked linear attention.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel overrides.
            tune: Whether to autotune kernels.
        """
        self.chunk_size = chunk_size
        self.tune = tune
        self.target = target
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, chunk length, dtype and device."""
        batch, heads, seq_len, dim_k, dim_v, dtype, _device = call
        return call, lambda: self.kernel_map[role](
            batch,
            heads,
            seq_len,
            self.chunk_size,
            dim_k,
            dim_v,
            dtype=Kernel.dtype_to_str(dtype),
            tune=self.tune,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Run deltanet forward.

        Args:
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            beta: Beta tensor [B, H, S].

        Returns:
            Tuple of (o, S, Aw, Au, w, u).
        """
        return self._call_boundary(q, k, v, beta)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        kernel = self.kernel_for(
            "DeltaNetFwdKernel", (q, k, v, beta), (*q.shape, v.shape[3], q.dtype, q.device.index)
        )
        return kernel(q, k, v, beta)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])


class DeltaNetBwdOp(Op):
    """DeltaNet backward operator (ungated).

    Pipeline: prepare_wy_repr -> fwd (to get Aw, Au) -> bwd kernel -> (dq, dk, dv, dbeta).

    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"DeltaNetBwdKernel": DeltaNetBwdKernel}

    def __init__(
        self,
        chunk_size: int = 64,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            chunk_size: Chunk size for chunked linear attention.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel overrides.
            tune: Whether to autotune kernels.
        """
        self.chunk_size = chunk_size
        self.tune = tune
        self.target = target
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, chunk length, dtype and device."""
        batch, heads, seq_len, dim_k, dim_v, dtype, _device = call
        return call, lambda: self.kernel_map[role](
            batch,
            heads,
            seq_len,
            self.chunk_size,
            dim_k,
            dim_v,
            dtype=Kernel.dtype_to_str(dtype),
            tune=self.tune,
        )

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
        return self._call_boundary(do, q, k, v, beta, S, Aw, Au, w, u)

    def _eager_forward(
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
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        inputs = (do, q, k, v, beta, S, Aw, Au, w, u)
        kernel = self.kernel_for(
            "DeltaNetBwdKernel", inputs, (*q.shape, v.shape[3], q.dtype, q.device.index)
        )
        return kernel(*inputs)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])


class _DeltaNetFunction(torch.autograd.Function):
    """Autograd function running DeltaNetFwdOp forward and DeltaNetBwdOp backward."""

    @staticmethod
    def forward(ctx, q, k, v, beta, fwd_op, bwd_op):
        """Run ``fwd_op`` and keep the chunk buffers ``bwd_op`` reads."""
        o, S, Aw, Au, w, u = fwd_op(q, k, v, beta)
        ctx.save_for_backward(q, k, v, beta, S, Aw, Au, w, u)
        ctx.bwd_op = bwd_op
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, beta, S, Aw, Au, w, u = ctx.saved_tensors
        dq, dk, dv, dbeta = ctx.bwd_op(do.contiguous(), q, k, v, beta, S, Aw, Au, w, u)
        return dq, dk, dv, dbeta, None, None


class DeltaNetAutogradFwdOp(Op):
    """Combined DeltaNet fwd+bwd operator with autograd support (ungated).

    Runs ``DeltaNetFwdOp`` inside a ``torch.autograd.Function`` whose backward
    runs ``DeltaNetBwdOp``, so ``output.backward(do)`` invokes the TileOPs
    backward kernels.

    Layout: BHSD (batch, head, seq_len, dim).

    """

    def __init__(
        self,
        chunk_size: int = 64,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            chunk_size: Chunk size for chunked linear attention.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel overrides, passed to both sub-ops.
            tune: Whether to autotune kernels.
        """
        self.chunk_size = chunk_size
        self.tune = tune
        self.target = target
        # This composite owns no kernel; the override reaches the sub-ops that do.
        self.dispatch_kernel(kernel_map)
        shared = {"target": target, "kernel_map": kernel_map, "tune": tune}
        self._fwd_op = DeltaNetFwdOp(chunk_size, **shared)
        self._bwd_op = DeltaNetBwdOp(chunk_size, **shared)

    def kernel_delegates(self) -> tuple[Op, ...]:
        return (self._fwd_op, self._bwd_op)

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
        return _DeltaNetFunction.apply(q, k, v, beta, self._fwd_op, self._bwd_op)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
