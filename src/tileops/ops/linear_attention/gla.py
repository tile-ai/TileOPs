from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.linear_attention.gla import GLABwdKernel, GLAFwdKernel

from .._validation import check_tensor_shape
from ..op_base import Op

__all__ = ["GLABwdOp", "GLAFwdOp"]


def _resolve_gla_bthd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
    do: Optional[torch.Tensor] = None,
) -> tuple[int, int, int, int, int, torch.dtype]:
    if not all(tensor.is_cuda for tensor in (q, k, v, g)):
        raise ValueError("q, k, v, and g must be CUDA tensors")
    if q.ndim != 4:
        raise ValueError("q must have shape [batch, seq_len, heads, dim_k]")
    batch, seq_len, heads, dim_k = q.shape
    if k.shape != (batch, seq_len, heads, dim_k):
        raise ValueError("k must match q shape")
    if v.ndim != 4 or v.shape[:3] != (batch, seq_len, heads):
        raise ValueError("v must have shape [batch, seq_len, heads, dim_v]")
    dim_v = v.shape[-1]
    if g.shape != (batch, seq_len, heads, dim_k):
        raise ValueError("g must match q shape")
    if do is not None and do.shape != (batch, seq_len, heads, dim_v):
        raise ValueError("do must have shape [batch, seq_len, heads, dim_v]")
    dtype = q.dtype
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})")
    return batch, seq_len, heads, dim_k, dim_v, dtype


class GLAFwdOp(Op):
    """GLA (Gated Linear Attention) forward operator.

    Chunked GLA forward: (q, k, v, g) -> (o, final_state).

    Layout: BTHD (batch, seq_len, heads, dim).

    Args:
        chunk_size: Chunk size for chunked linear attention.
        scale: Query scale factor (default: dim_k**-0.5).
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        scale: float = -1.0,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = None
        self.seq_len = None
        self.heads = None
        self.dim_k = None
        self.dim_v = None
        self.chunk_size = chunk_size
        self.scale = scale
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GLAFwdKernel": GLAFwdKernel,
        }

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (
            batch,
            seq_len,
            heads,
            dim_k,
            dim_v,
            self.chunk_size,
            self.scale,
            dtype,
            device_index,
            self.tune,
        )
        return self.get_or_build_kernel(
            "GLAFwdKernel",
            inputs,
            key=key,
            build=lambda: self.kernel_map["GLAFwdKernel"](
                batch,
                seq_len,
                heads,
                dim_k,
                dim_v,
                self.chunk_size,
                scale=self.scale,
                output_final_state=True,
                dtype=dtype,
                tune=self.tune,
            ),
        )

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
        initial_state_shape: tuple[int, ...] | None,
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: the output, and the state the recurrence ends on."""
        b, s, h, dk = q_shape
        dv = v_shape[3]
        return {"o": (b, s, h, dv), "final_state": (b, h, dk, dv)}

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
        batch, seq_len, heads, dim_k, dim_v, dtype = _resolve_gla_bthd(q, k, v, g, self.chunk_size)
        self._validate_dtypes(q, k, v, g, initial_state=initial_state)
        if initial_state is not None:
            check_tensor_shape("initial_state", initial_state, (batch, heads, dim_k, dim_v))
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.kernel = self._get_kernel(
            (q, k, v, g, initial_state), batch, seq_len, heads, dim_k, dim_v, dtype, q.device.index
        )
        return self.kernel(q, k, v, g, initial_state)


class GLABwdOp(Op):
    """GLA (Gated Linear Attention) backward operator.

    Computes gradients (dq, dk, dv, dg) given output gradient do.

    Uses h_out saved from the forward pass (no recomputation needed).

    Layout: BTHD (batch, seq_len, heads, dim).

    Args:
        chunk_size: Chunk size for chunked linear attention.
        scale: Query scale factor (default: dim_k**-0.5).
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        scale: float = -1.0,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = None
        self.seq_len = None
        self.heads = None
        self.dim_k = None
        self.dim_v = None
        self.chunk_size = chunk_size
        self.scale = scale
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GLABwdKernel": GLABwdKernel,
        }

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (
            batch,
            seq_len,
            heads,
            dim_k,
            dim_v,
            self.chunk_size,
            self.scale,
            dtype,
            device_index,
            self.tune,
        )
        return self.get_or_build_kernel(
            "GLABwdKernel",
            inputs,
            key=key,
            build=lambda: self.kernel_map["GLABwdKernel"](
                batch,
                seq_len,
                heads,
                dim_k,
                dim_v,
                self.chunk_size,
                scale=self.scale,
                dtype=dtype,
                tune=self.tune,
            ),
        )

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
        h_shape: tuple[int, ...],
        do_shape: tuple[int, ...],
        dht_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: each gradient has the shape of what it is for."""
        return {
            "dq": tuple(q_shape),
            "dk": tuple(k_shape),
            "dv": tuple(v_shape),
            "dg": tuple(g_shape),
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
        batch, seq_len, heads, dim_k, dim_v, dtype = _resolve_gla_bthd(
            q, k, v, g, self.chunk_size, do=do
        )
        self._validate_dtypes(q, k, v, g, h, do, dht)
        chunks = seq_len // self.chunk_size
        check_tensor_shape("h", h, (batch, chunks + 1, heads, dim_k, dim_v))
        check_tensor_shape("dht", dht, (batch, heads, dim_k, dim_v))
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.kernel = self._get_kernel(
            (q, k, v, g, h, do, dht), batch, seq_len, heads, dim_k, dim_v, dtype, q.device.index
        )
        return self.kernel(q, k, v, g, h, do, dht, has_initial_state)
