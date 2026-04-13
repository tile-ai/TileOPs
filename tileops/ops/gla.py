from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.gla import GLABwdKernel, GLAFwdKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["GLAFwdOp", "GLABwdOp"]


class GLAFwdOp(Op):
    """GLA (Gated Linear Attention) forward operator.

    Chunked GLA forward: (q, k, v, g) -> (o, final_state).

    Layout: BTHD (batch, seq_len, heads, dim).

    Args:
        batch: Batch size.
        seq_len: Sequence length (must be divisible by chunk_size).
        heads: Number of attention heads.
        dim_k: Key/query dimension.
        dim_v: Value dimension.
        chunk_size: Chunk size for chunked linear attention.
        scale: Query scale factor (default: dim_k**-0.5).
        output_final_state: Whether to return the final hidden state.
        dtype: Data type for computation.
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: float = -1.0,
        output_final_state: bool = False,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.scale = scale
        self.output_final_state = output_final_state
        self.dtype = dtype

        assert seq_len % chunk_size == 0, (
            f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
        )

        self.dispatch_kernel(kernel_map)

        fwd_kernel_cls = self.kernel_map["GLAFwdKernel"]
        self.kernel = fwd_kernel_cls(
            batch, seq_len, heads, dim_k, dim_v, chunk_size,
            scale=scale,
            output_final_state=output_final_state,
            dtype=dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GLAFwdKernel": GLAFwdKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run GLA forward.

        Args:
            q: Query tensor [B, T, H, K].
            k: Key tensor [B, T, H, K].
            v: Value tensor [B, T, H, V].
            g: Log-space forget gates [B, T, H, K].
            initial_state: Optional initial hidden state [B, H, K, V].

        Returns:
            Tuple of (o, final_state). final_state is None if output_final_state=False.
        """
        return self.kernel(q, k, v, g, initial_state)


class GLABwdOp(Op):
    """GLA (Gated Linear Attention) backward operator.

    Computes gradients (dq, dk, dv, dg) given output gradient do.

    Uses h_out saved from the forward pass (no recomputation needed).

    Layout: BTHD (batch, seq_len, heads, dim).

    Args:
        batch: Batch size.
        seq_len: Sequence length (must be divisible by chunk_size).
        heads: Number of attention heads.
        dim_k: Key/query dimension.
        dim_v: Value dimension.
        chunk_size: Chunk size for chunked linear attention.
        scale: Query scale factor (default: dim_k**-0.5).
        dtype: Data type for computation.
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: float = -1.0,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.scale = scale
        self.dtype = dtype

        assert seq_len % chunk_size == 0, (
            f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
        )

        self.dispatch_kernel(kernel_map)

        bwd_kernel_cls = self.kernel_map["GLABwdKernel"]
        self.kernel = bwd_kernel_cls(
            batch, seq_len, heads, dim_k, dim_v, chunk_size,
            scale=scale,
            dtype=dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GLABwdKernel": GLABwdKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        has_initial_state: bool = False,
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
            has_initial_state: Whether initial_state was provided by the user.

        Returns:
            Tuple of (dq, dk, dv, dg).
        """
        return self.kernel(q, k, v, g, h, do, dht, has_initial_state)
