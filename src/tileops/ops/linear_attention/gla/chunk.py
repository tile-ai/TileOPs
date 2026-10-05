from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.linear_attention import (
    GLABwdInterface,
    GLABwdKernel,
    GLAChunkCall,
    GLAFwdInterface,
    GLAFwdKernel,
)
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["GLAChunkBwdOp", "GLAChunkFwdOp"]


class GLAChunkFwdOp(Op):
    """GLA (Gated Linear Attention) forward operator.

    Chunked GLA forward: (q, k, v, g) -> (o, final_state).

    Layout: BTHD (batch, seq_len, heads, dim).

    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"gla_fwd": GLAFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gla_fwd": GLAFwdInterface}

    def __init__(
        self,
        chunk_size: int = 64,
        scale: float = -1.0,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            chunk_size: Chunk size for chunked linear attention.
            scale: Query scale factor; a non-positive value means ``dim_k ** -0.5``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel overrides.
            tune: Whether to autotune kernels.
        """
        self.chunk_size = chunk_size
        self.scale = scale
        self.tune = tune
        self.target = target
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run GLA forward.

        Args:
            q: Query tensor [B, T, H, K].
            k: Key tensor [B, T, H, K].
            v: Value tensor [B, T, H, V].
            g: Log-space forget gates [B, T, H, K].
            initial_state: Optional fp32 initial hidden state [B, H, K, V]; absent
                starts the recurrence from zeros.

        Returns:
            Tuple of (o, final_state).
        """
        return self._call_boundary(q, k, v, g, initial_state)

    def _call(self, q: torch.Tensor, v: torch.Tensor, has_initial_state: bool) -> GLAChunkCall:
        """The facts of one call: the shapes read off the tensors and the op's own params."""
        batch, seq_len, heads, dim_k = q.shape
        return GLAChunkCall(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim_k=dim_k,
            dim_v=v.shape[3],
            chunk_size=self.chunk_size,
            scale=self.scale,
            dtype=q.dtype,
            has_initial_state=has_initial_state,
            device=q.device,
        )

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        call = self._call(q, v, initial_state is not None)
        kernel = self.kernel_for("gla_fwd", call)
        return kernel(q, k, v, g, initial_state)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])


class GLAChunkBwdOp(Op):
    """GLA (Gated Linear Attention) backward operator.

    Computes gradients (dq, dk, dv, dg) given output gradient do.

    Uses h_out saved from the forward pass (no recomputation needed).

    Layout: BTHD (batch, seq_len, heads, dim).

    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"gla_bwd": GLABwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gla_bwd": GLABwdInterface}

    def __init__(
        self,
        chunk_size: int = 64,
        scale: float = -1.0,
        has_initial_state: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            chunk_size: Chunk size for chunked linear attention.
            scale: Query scale factor; a non-positive value means ``dim_k ** -0.5``.
            has_initial_state: Whether the forward this backward pairs with was given
                an initial state.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel overrides.
            tune: Whether to autotune kernels.
        """
        self.chunk_size = chunk_size
        self.scale = scale
        self.has_initial_state = has_initial_state
        self.tune = tune
        self.target = target
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run GLA backward.

        Args:
            q: Queries [B, T, H, K].
            k: Keys [B, T, H, K].
            v: Values [B, T, H, V].
            g: Log-space forget gates [B, T, H, K].
            h: Hidden states from forward [B, NT+1, H, K, V] (fp32).
            do: Output gradient [B, T, H, V].
            dht: Final-state gradient [B, H, K, V].

        Returns:
            Tuple of (dq, dk, dv, dg).
        """
        return self._call_boundary(q, k, v, g, h, do, dht)

    def _call(self, q: torch.Tensor, v: torch.Tensor, has_initial_state: bool) -> GLAChunkCall:
        """The facts of one call: the shapes read off the tensors and the op's own params."""
        batch, seq_len, heads, dim_k = q.shape
        return GLAChunkCall(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim_k=dim_k,
            dim_v=v.shape[3],
            chunk_size=self.chunk_size,
            scale=self.scale,
            dtype=q.dtype,
            has_initial_state=has_initial_state,
            device=q.device,
        )

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        inputs = (q, k, v, g, h, do, dht)
        kernel = self.kernel_for("gla_bwd", self._call(q, v, self.has_initial_state))
        return kernel(*inputs, self.has_initial_state)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
