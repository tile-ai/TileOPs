"""Inference-facing GLA contract with dense-prefill and decode dispatch."""

import math
from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.gla.dense_decode import GLADenseDecodeFwdKernel
from tileops.kernels.linear_attention.gla.dense_prefill_partitioned import (
    GLADensePrefillPartitionedKernel,
)
from tileops.kernels.linear_attention.gla.dense_prefill_subchunk import (
    GLADensePrefillSubchunkKernel,
)
from tileops.perf.profile import tensor_core_roof
from tileops.utils import is_h200

from ..op_base import Op

__all__ = ["GLAInferenceFwdOp"]


class GLAInferenceFwdOp(Op):
    """Gated linear attention for inference, with caller-owned FP32 state.

    Q, K, V and the log-space, per-key gate G use FP16/BF16 BTHD layout. One call may
    describe equal-length prefill, packed-varlen prefill, or single-token
    decode. The caller may omit ``initial_state`` to start from zero; every
    call returns ``(o, final_state)``. Hopper dense prefill and single-token
    decode are implemented in tree; packed varlen is not.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gla_dense_decode": GLADenseDecodeFwdKernel,
        "gla_dense_prefill_partitioned": GLADensePrefillPartitionedKernel,
        "gla_dense_prefill_subchunk": GLADensePrefillSubchunkKernel,
    }

    def __init__(
        self,
        scale: Optional[float] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Fix the query scale and optional backend target for this instance.

        Args:
            scale: Positive query scale, or ``None`` for ``K**-0.5``.
            target: Backend target, or ``None`` to resolve from the input
                device.
            kernel_map: Optional in-tree kernel overrides.
            tune: Autotune a kernel when it is first built.
        """
        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            raise ValueError("scale must be a positive finite value")
        self.scale = scale
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        batch, seq_len, heads, dim_k, dim_v, dtype, device, scale, varlen = call
        unsupported = []
        if varlen:
            unsupported.append("packed varlen")
        if seq_len != 1 and (seq_len < 64 or seq_len % 64):
            unsupported.append("T not divisible by 64")
        if dim_k != dim_v or dim_k not in (64, 128):
            unsupported.append("K/V dimensions other than matching 64 or 128")
        if dtype not in (torch.float16, torch.bfloat16):
            unsupported.append("dtype other than float16 or bfloat16")
        if device.type != "cuda":
            unsupported.append("non-CUDA device")
        if unsupported:
            raise ValueError(
                "the in-tree GLA dense kernel does not yet support " + ", ".join(unsupported)
            )
        if role == "gla_dense_decode":
            return call, lambda: self.kernel_map["gla_dense_decode"](
                batch=batch,
                heads=heads,
                dim_k=dim_k,
                dim_v=dim_v,
                scale=scale,
                dtype=dtype,
                device_index=device.index,
            )
        # A 16-chunk partition creates enough independent CTAs only for long
        # calls. Shorter calls keep the existing serial-state specialization.
        partition_ctas = batch * heads * (seq_len // 1024)
        kernel_key = (
            "gla_dense_prefill_partitioned"
            if (
                seq_len >= 16384
                and dim_k == 64
                and dim_v == 64
                and seq_len % 1024 == 0
                and partition_ctas >= 128
                and is_h200(device.index)
            )
            else "gla_dense_prefill_subchunk"
        )
        return call, lambda: self.kernel_map[kernel_key](
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim_k=dim_k,
            dim_v=dim_v,
            scale=scale,
            dtype=dtype,
            device_index=device.index,
        )

    def compute_roof(self) -> str:
        """Prefill contracts chunks on tensor cores; a decode step is a matvec on CUDA cores."""
        q_shape, q_dtype = self.last_call.tensors["q"]
        return super().compute_roof() if q_shape[1] == 1 else tensor_core_roof(q_dtype)

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
        inputs = tuple(
            tensor.contiguous() if tensor is not None else None
            for tensor in (q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu)
        )
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
        role = "gla_dense_decode" if seq_len == 1 else "gla_dense_prefill"
        kernel = self.kernel_for(role, inputs, call)
        return kernel(*inputs)
