from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.gla import GLABwdKernel, GLAFwdKernel, GLAPrefillFwdKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["GLABwdOp", "GLAFwdOp", "GLAPrefillFwdOp"]


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
    for name, tensor in (("k", k), ("v", v), ("g", g)):
        if tensor.dtype != dtype:
            raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
    if do is not None and do.dtype != dtype:
        raise ValueError(f"do.dtype must be {dtype}, got {do.dtype}")
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
        output_final_state: Whether to return the final hidden state.
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        scale: float = -1.0,
        output_final_state: bool = False,
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
        self.output_final_state = output_final_state
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
            self.output_final_state,
            dtype,
            device_index,
            self.tune,
        )
        return self.get_or_build_kernel(
            "GLAFwdKernel",
            key=key,
            build=lambda: self.kernel_map["GLAFwdKernel"](
                batch,
                seq_len,
                heads,
                dim_k,
                dim_v,
                self.chunk_size,
                scale=self.scale,
                output_final_state=self.output_final_state,
                dtype=dtype,
                tune=self.tune,
            ),
        )

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
        batch, seq_len, heads, dim_k, dim_v, dtype = _resolve_gla_bthd(q, k, v, g, self.chunk_size)
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.kernel = self._get_kernel(batch, seq_len, heads, dim_k, dim_v, dtype, q.device.index)
        return self.kernel(q, k, v, g, initial_state)


class GLAPrefillFwdOp(Op):
    """Serving-oriented zero-state GLA prefill.

    Unlike :class:`GLAFwdOp`, this interface does not accept an initial state
    or retain training-forward artifacts for backward. It always returns the
    prompt output and the recurrent state consumed by decode.

    Layout: BTHD (batch, sequence, heads, dimension).
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
        self._active_sig: Optional[tuple] = None
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"GLAPrefillFwdKernel": GLAPrefillFwdKernel}

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        del k_shape, g_shape
        return {
            "o": tuple(q_shape[:-1]) + (v_shape[-1],),
            "final_state": (q_shape[0], q_shape[2], q_shape[-1], v_shape[-1]),
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
    ) -> None:
        dtype = q.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported dtype: {dtype}")
        for name, tensor in (("k", k), ("v", v), ("g", g)):
            if tensor.dtype != dtype:
                raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")

    def _get_kernel(
        self,
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
            "GLAPrefillFwdKernel",
            key=key,
            build=lambda: self.kernel_map["GLAPrefillFwdKernel"](
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

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import gla_prefill_fwd_roofline

        return gla_prefill_fwd_roofline(self)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        sig = (
            q.shape,
            k.shape,
            v.shape,
            g.shape,
            q.dtype,
            k.dtype,
            v.dtype,
            g.dtype,
            q.device,
            k.device,
            v.device,
            g.device,
            self.chunk_size,
            self.scale,
            self.tune,
        )
        if sig != self._active_sig:
            self._validate_dtypes(q, k, v, g)
            batch, seq_len, heads, dim_k, dim_v, dtype = _resolve_gla_bthd(
                q, k, v, g, self.chunk_size
            )
            self.batch = batch
            self.seq_len = seq_len
            self.heads = heads
            self.dim_k = dim_k
            self.dim_v = dim_v
            self.dtype = dtype
            self.kernel = self._get_kernel(
                batch, seq_len, heads, dim_k, dim_v, dtype, q.device.index
            )
            self._active_sig = sig
        return self.kernel(q, k, v, g)


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
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.kernel = self._get_kernel(batch, seq_len, heads, dim_k, dim_v, dtype, q.device.index)
        return self.kernel(q, k, v, g, h, do, dht, has_initial_state)
