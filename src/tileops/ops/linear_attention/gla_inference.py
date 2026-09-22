"""Inference-facing GLA contract and dense-prefill dispatch."""

import math
from typing import Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.gla.dense_prefill_subchunk import (
    GLADensePrefillSubchunkKernel,
)
from tileops.perf.formulas import gla_fwd_roofline
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["GLAInferenceFwdOp"]


class GLAInferenceFwdOp(Op):
    """Gated linear attention for inference, with caller-owned FP32 state.

    Q, K, V and the log-space, per-key gate G use BTHD layout. One call may
    describe equal-length prefill, packed-varlen prefill, or single-token
    decode. The caller may omit ``initial_state`` to start from zero; every
    call returns ``(o, final_state)``. Only Hopper dense prefill is currently
    implemented in tree. The old training-forward and decode Ops are intact.
    """

    def __init__(
        self,
        scale: Optional[float] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        target: Target = None,
    ) -> None:
        """Fix the query scale and optional backend target for this instance."""
        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            raise ValueError("scale must be a positive finite value")
        self.scale = scale
        self.target = target
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gla_dense_prefill": GLADensePrefillSubchunkKernel}

    def entry_for(self, role: str, call: tuple) -> Entry:
        del role
        batch, seq_len, heads, dim_k, dim_v, dtype, device, scale, varlen = call
        unsupported = []
        if varlen:
            unsupported.append("packed varlen")
        if seq_len < 64 or seq_len % 64:
            unsupported.append("T not divisible by 64")
        if dim_k != dim_v or dim_k not in (64, 128):
            unsupported.append("K/V dimensions other than matching 64 or 128")
        if dtype not in (torch.float16, torch.bfloat16):
            unsupported.append("dtype other than float16 or bfloat16")
        if device.type != "cuda":
            unsupported.append("non-CUDA device")
        if unsupported:
            raise ValueError(
                "the in-tree GLA dense-prefill kernel does not yet support "
                + ", ".join(unsupported)
            )
        return call, lambda: self.kernel_map["gla_dense_prefill"](
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim_k=dim_k,
            dim_v=dim_v,
            scale=scale,
            dtype=dtype,
            device_index=device.index,
        )

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
        initial_state_shape: Optional[tuple[int, ...]] = None,
        cu_seqlens_shape: Optional[tuple[int, ...]] = None,
        cu_seqlens_cpu_shape: Optional[tuple[int, ...]] = None,
    ) -> dict[str, tuple[int, ...]]:
        del k_shape, g_shape, initial_state_shape, cu_seqlens_cpu_shape
        batch, seq_len, heads, dim_k = q_shape
        state_batch = cu_seqlens_shape[0] - 1 if cu_seqlens_shape is not None else batch
        return {
            "o": (batch, seq_len, heads, v_shape[-1]),
            "final_state": (state_batch, heads, dim_k, v_shape[-1]),
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("q must have float16, bfloat16, or float32 dtype")
        for name, tensor in (("k", k), ("v", v), ("g", g)):
            if tensor.dtype != q.dtype:
                raise ValueError(f"{name} must have the same dtype as q")
        if initial_state is not None and initial_state.dtype != torch.float32:
            raise ValueError("initial_state must have float32 dtype")
        for name, tensor in (("cu_seqlens", cu_seqlens), ("cu_seqlens_cpu", cu_seqlens_cpu)):
            if tensor is not None and tensor.dtype != torch.int64:
                raise ValueError(f"{name} must have int64 dtype")

    def _validate_forward_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        cu_seqlens_cpu: Optional[torch.Tensor],
    ) -> None:
        if q.ndim != 4 or k.shape != q.shape or g.shape != q.shape:
            raise ValueError("q, k, and g must have the same [B, T, H, K] shape")
        if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
            raise ValueError("v must have shape [B, T, H, V]")
        batch, seq_len, heads, dim_k = q.shape
        if min(batch, seq_len, heads, dim_k, v.shape[-1]) <= 0:
            raise ValueError("GLA tensor dimensions must be positive")

        state_batch = batch
        if cu_seqlens is not None:
            if batch != 1 or cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
                raise ValueError("packed cu_seqlens requires q batch 1 and shape [N + 1]")
            state_batch = cu_seqlens.shape[0] - 1
        if cu_seqlens_cpu is not None:
            if cu_seqlens is None or cu_seqlens_cpu.shape != cu_seqlens.shape:
                raise ValueError("cu_seqlens_cpu requires matching cu_seqlens")
            if cu_seqlens_cpu.device.type != "cpu":
                raise ValueError("cu_seqlens_cpu must be on CPU")
        if initial_state is not None and initial_state.shape != (
            state_batch,
            heads,
            dim_k,
            v.shape[-1],
        ):
            raise ValueError("initial_state must have shape [N, H, K, V]")

        self._validate_dtypes(q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu)
        for name, tensor in (
            ("k", k),
            ("v", v),
            ("g", g),
            ("initial_state", initial_state),
            ("cu_seqlens", cu_seqlens),
        ):
            if tensor is not None and tensor.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")

    def eval_roofline(self) -> tuple[int, int]:
        if not hasattr(self, "q_shape"):
            raise RuntimeError("eval_roofline() requires one completed forward call")
        return gla_fwd_roofline(
            q_shape=self.q_shape,
            v_shape=self.v_shape,
            dtype=self.dtype,
            initial_state_shape=self.state_shape,
        )

    def compute_roof(self) -> str:
        if not hasattr(self, "dtype"):
            raise RuntimeError("compute_roof() requires one completed forward call")
        return tensor_core_roof(self.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run GLA prefill or decode and return output plus final FP32 state."""
        self._validate_forward_inputs(q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu)
        inputs = tuple(
            tensor.contiguous() if tensor is not None else None
            for tensor in (q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu)
        )
        self.q_shape = tuple(q.shape)
        self.v_shape = tuple(v.shape)
        self.dtype = q.dtype
        self.state_shape = tuple(initial_state.shape) if initial_state is not None else None
        batch, seq_len, heads, dim_k = q.shape
        call = (
            batch,
            seq_len,
            heads,
            dim_k,
            v.shape[-1],
            q.dtype,
            q.device,
            self.scale if self.scale is not None else dim_k**-0.5,
            cu_seqlens is not None,
        )
        kernel = self.kernel_for("gla_dense_prefill", inputs, call)
        return kernel(*inputs)
