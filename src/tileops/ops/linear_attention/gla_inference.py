"""Inference-facing GLA contract and dense-prefill dispatch."""

import math
from typing import Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
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
        return {
            "gla_dense_prefill_partitioned": GLADensePrefillPartitionedKernel,
            "gla_dense_prefill_subchunk": GLADensePrefillSubchunkKernel,
        }

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
        """The state contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

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
        kernel = self.kernel_for("gla_dense_prefill", inputs, call)
        return kernel(*inputs)
