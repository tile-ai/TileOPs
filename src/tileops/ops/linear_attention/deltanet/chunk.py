from typing import ClassVar, Mapping, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.linear_attention import (
    DeltaNetChunkBwdInterface,
    DeltaNetChunkBwdKernel,
    DeltaNetChunkCall,
    DeltaNetChunkFwdInterface,
    DeltaNetChunkFwdKernel,
)
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["DeltaNetChunkBwdOp", "DeltaNetChunkFwdOp"]


class DeltaNetChunkFwdOp(Op):
    """DeltaNet forward operator (ungated).

    Pipeline: prepare_wy_repr(k, beta) -> (Aw, Au) -> deltanet_fwd(q, k, v, beta, Aw, Au) -> o.

    Layout: BHSD (batch, head, seq_len, dim).

    !!! note "Layout convention difference with FLA"

        TileOPs uses **BHSD** layout: ``q/k [B, H, S, DK]``, ``v [B, H, S, DV]``,
        ``beta [B, H, S]``.

        FLA (``fla.ops.delta_rule.chunk_delta_rule``) uses **BTHN**
        layout: ``q/k [B, T, H, K]``, ``v [B, T, H, V]``, ``beta [B, T, H]``.

    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "deltanet_chunk_fwd": DeltaNetChunkFwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "deltanet_chunk_fwd": DeltaNetChunkFwdInterface
    }

    def __init__(
        self,
        chunk_size: int = 64,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            chunk_size: Chunk size for chunked linear attention.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.chunk_size = chunk_size
        super().__init__(target=target)

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
        kernel = self.kernel_for("deltanet_chunk_fwd", self._call(q, v))
        return kernel(q, k, v, beta)

    def _call(self, q: torch.Tensor, v: torch.Tensor) -> DeltaNetChunkCall:
        """The facts of one call: the shapes read off the tensors and the op's chunk length."""
        batch, heads, seq_len, dim_k = q.shape
        return DeltaNetChunkCall(
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            chunk_size=self.chunk_size,
            dim_k=dim_k,
            dim_v=v.shape[3],
            dtype=q.dtype,
            device=q.device,
        )

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])


class DeltaNetChunkBwdOp(Op):
    """DeltaNet backward operator (ungated).

    Pipeline: prepare_wy_repr -> fwd (to get Aw, Au) -> bwd kernel -> (dq, dk, dv, dbeta).

    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "deltanet_chunk_bwd": DeltaNetChunkBwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "deltanet_chunk_bwd": DeltaNetChunkBwdInterface
    }

    def __init__(
        self,
        chunk_size: int = 64,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            chunk_size: Chunk size for chunked linear attention.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.chunk_size = chunk_size
        super().__init__(target=target)

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
        inputs = (do, q, k, v, beta, S, Aw, Au, w, u)
        kernel = self.kernel_for("deltanet_chunk_bwd", self._call(q, v))
        return kernel(*inputs)

    def _call(self, q: torch.Tensor, v: torch.Tensor) -> DeltaNetChunkCall:
        """The facts of one call: the shapes read off the tensors and the op's chunk length."""
        batch, heads, seq_len, dim_k = q.shape
        return DeltaNetChunkCall(
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            chunk_size=self.chunk_size,
            dim_k=dim_k,
            dim_v=v.shape[3],
            dtype=q.dtype,
            device=q.device,
        )

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
