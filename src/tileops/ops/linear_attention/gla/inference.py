"""Inference-facing GLA contract with dense-prefill and decode dispatch."""

import math
from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.linear_attention.gla.call_spec import (
    GLAInferenceCallSpec,
    GLAInferenceFwdInterface,
)
from tileops.kernels.linear_attention.gla.dense_decode import GLADenseDecodeFwdKernel
from tileops.kernels.linear_attention.gla.dense_prefill_partitioned import (
    GLADensePrefillPartitionedKernel,
)
from tileops.kernels.linear_attention.gla.dense_prefill_subchunk import (
    GLADensePrefillSubchunkKernel,
)
from tileops.kernels.linear_attention.gla.varlen_prefill import GLAVarlenPrefillFwdKernel
from tileops.kernels.linear_attention.gla.varlen_prefill_partitioned import (
    GLAVarlenPrefillPartitionedFwdKernel,
)
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["GLAInferenceFwdOp"]


class GLAInferenceFwdOp(Op):
    """Gated Linear Attention (GLA) for inference, with caller-owned FP32 state.

    Q, K, V and the log-space, per-key gate G use FP16/BF16 BTHD layout. One call is
    equal-length prefill, packed-varlen prefill, or single-token decode. An absent
    ``initial_state`` starts from zero; every call returns ``(o, final_state)``. The
    in-tree kernels serve equal-length prefill, packed-varlen prefill and decode.
    """

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gla_dense_decode": GLADenseDecodeFwdKernel,
        "gla_dense_prefill_partitioned": GLADensePrefillPartitionedKernel,
        "gla_dense_prefill_subchunk": GLADensePrefillSubchunkKernel,
        "gla_varlen_prefill": GLAVarlenPrefillFwdKernel,
        "gla_varlen_prefill_partitioned": GLAVarlenPrefillPartitionedFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "gla_inference": GLAInferenceFwdInterface
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
        return self._call_boundary(q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator."""
        inputs = tuple(
            tensor.contiguous() if tensor is not None else None
            for tensor in (q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu)
        )
        batch, seq_len, heads, dim_k = q.shape
        call = GLAInferenceCallSpec(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim_k=dim_k,
            dim_v=v.shape[-1],
            dtype=q.dtype,
            scale=self.scale if self.scale is not None else dim_k**-0.5,
            varlen=cu_seqlens is not None,
            has_initial_state=initial_state is not None,
            num_sequences=batch if cu_seqlens is None else cu_seqlens.shape[0] - 1,
            device=q.device,
        )
        return self.kernel_for("gla_inference", call)(*inputs)
