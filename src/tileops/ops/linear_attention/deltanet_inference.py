"""Inference-facing DeltaNet forward contract and dense-prefill dispatch."""

import math
from typing import Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.deltanet.dense_prefill import (
    DeltaNetDensePrefillFwdKernel,
)
from tileops.perf.formulas import deltanet_inference_roofline
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["DeltaNetInferenceFwdOp"]


class DeltaNetInferenceFwdOp(Op):
    """Inference forward for the ungated delta rule.

    The input layout is ``[B, T, H, D]``. One call covers equal-length
    prefill, packed-varlen prefill, and single-token decode. The recurrent
    state is FP32 and belongs to the caller: ``initial_state`` is optional,
    while ``(o, final_state)`` is always returned. The in-tree implementation
    currently supports Hopper dense prefill; packed varlen and decode remain
    part of the public contract for external targets and future kernels.

    ``beta`` contains the already-transformed update strength. This Op does
    not apply a sigmoid or another beta transform.
    """

    def __init__(
        self,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        target: Target = None,
    ) -> None:
        """Configure the attention scale, Q/K normalization, and target."""
        if scale is not None and not math.isfinite(scale):
            raise ValueError(f"scale must be finite, got {scale}")
        self.scale = scale
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self.target = target
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"deltanet_dense_prefill": DeltaNetDensePrefillFwdKernel}

    def entry_for(self, role: str, call: tuple) -> Entry:
        del role
        (
            batch,
            seq_len,
            heads,
            dim_k,
            dim_v,
            dtype,
            device_index,
            scale,
            l2norm,
            _has_initial_state,
            varlen,
        ) = call
        unsupported = []
        if l2norm:
            unsupported.append("Q/K L2 normalization")
        if varlen:
            unsupported.append("packed varlen")
        if seq_len < 64 or seq_len % 64:
            unsupported.append("T not divisible by 64")
        if dim_k != dim_v or dim_k not in (64, 128):
            unsupported.append("K/V dimensions other than matching 64 or 128")
        if dtype not in (torch.float16, torch.bfloat16):
            unsupported.append("dtype other than float16 or bfloat16")
        if unsupported:
            raise ValueError(
                "the in-tree DeltaNet dense-prefill kernel does not yet support "
                + ", ".join(unsupported)
            )
        return call, lambda: self.kernel_map["deltanet_dense_prefill"](
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            dim=dim_k,
            scale=scale,
            dtype=dtype,
            device_index=device_index,
        )

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
        initial_state_shape: Optional[tuple[int, ...]] = None,
        cu_seqlens_shape: Optional[tuple[int, ...]] = None,
        cu_seqlens_cpu_shape: Optional[tuple[int, ...]] = None,
    ) -> dict[str, tuple[int, ...]]:
        del k_shape, beta_shape, initial_state_shape, cu_seqlens_cpu_shape
        batch, seq_len, heads, dim_k = q_shape
        dim_v = v_shape[-1]
        state_batch = cu_seqlens_shape[0] - 1 if cu_seqlens_shape is not None else batch
        return {
            "o": (batch, seq_len, heads, dim_v),
            "final_state": (state_batch, heads, dim_k, dim_v),
        }

    def _validate_forward_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        cu_seqlens_cpu: Optional[torch.Tensor],
    ) -> None:
        if q.ndim != 4 or k.shape != q.shape:
            raise ValueError("q and k must have the same [B, T, H, K] shape")
        if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
            raise ValueError("v must have shape [B, T, H, V]")
        batch, seq_len, heads, dim_k = q.shape
        if min(batch, seq_len, heads, dim_k, v.shape[-1]) <= 0:
            raise ValueError("DeltaNet tensor dimensions must be positive")
        if beta.shape != (batch, seq_len, heads):
            raise ValueError("beta must have shape [B, T, H]")

        state_batch = batch
        if cu_seqlens is not None:
            if batch != 1:
                raise ValueError("packed varlen inputs require B == 1")
            if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
                raise ValueError("cu_seqlens must have shape [N + 1]")
            state_batch = cu_seqlens.shape[0] - 1
        if cu_seqlens_cpu is not None:
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_cpu requires cu_seqlens")
            if cu_seqlens_cpu.shape != cu_seqlens.shape:
                raise ValueError("cu_seqlens_cpu must have the same shape as cu_seqlens")
            if cu_seqlens_cpu.device.type != "cpu":
                raise ValueError("cu_seqlens_cpu must be on CPU")
        if initial_state is not None and initial_state.shape != (
            state_batch,
            heads,
            dim_k,
            v.shape[-1],
        ):
            raise ValueError("initial_state must have shape [N, H, K, V]")

        self._validate_dtypes(q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu)
        for name, tensor in (
            ("k", k),
            ("v", v),
            ("beta", beta),
            ("initial_state", initial_state),
            ("cu_seqlens", cu_seqlens),
        ):
            if tensor is not None and tensor.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("q must have float16 or bfloat16 dtype")
        for name, tensor in (("k", k), ("v", v), ("beta", beta)):
            if tensor.dtype != q.dtype:
                raise ValueError(f"{name} must have the same dtype as q")
        if initial_state is not None and initial_state.dtype != torch.float32:
            raise ValueError("initial_state must have float32 dtype")
        for name, tensor in (("cu_seqlens", cu_seqlens), ("cu_seqlens_cpu", cu_seqlens_cpu)):
            if tensor is not None and tensor.dtype != torch.int64:
                raise ValueError(f"{name} must have int64 dtype")

    def eval_roofline(self) -> tuple[int, int]:
        if not hasattr(self, "q_shape"):
            raise RuntimeError("eval_roofline() requires one completed forward call")
        return deltanet_inference_roofline(
            q_shape=self.q_shape,
            v_shape=self.v_shape,
            dtype=self.dtype,
            initial_state=self.has_initial_state,
            cu_seqlens_shape=self.cu_seqlens_shape,
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
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run prefill or decode and return ``(o, final_state)``."""
        self._validate_forward_inputs(q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu)
        inputs = tuple(
            tensor.contiguous() if tensor is not None else None
            for tensor in (q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu)
        )
        self.q_shape = tuple(q.shape)
        self.v_shape = tuple(v.shape)
        self.dtype = q.dtype
        self.has_initial_state = initial_state is not None
        self.cu_seqlens_shape = tuple(cu_seqlens.shape) if cu_seqlens is not None else None
        batch, seq_len, heads, dim_k = q.shape
        call = (
            batch,
            seq_len,
            heads,
            dim_k,
            v.shape[-1],
            q.dtype,
            q.device.index,
            self.scale if self.scale is not None else dim_k**-0.5,
            self.use_qk_l2norm_in_kernel,
            initial_state is not None,
            cu_seqlens is not None,
        )
        kernel = self.kernel_for("deltanet_dense_prefill", inputs, call)
        return kernel(*inputs)
