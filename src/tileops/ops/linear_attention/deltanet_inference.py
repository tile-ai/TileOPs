"""Inference-facing DeltaNet forward contract and dense-prefill dispatch."""

from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.linear_attention import (
    DeltaNetDensePrefillFwdKernel,
    DeltaNetInferenceCall,
    DeltaNetInferenceFwdInterface,
)
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["DeltaNetInferenceFwdOp"]


class DeltaNetInferenceFwdOp(Op):
    """Inference forward for the ungated delta rule.

    The input layout is ``[B, T, H, D]``. One call covers equal-length
    prefill, packed-varlen prefill, and single-token decode. The recurrent
    state is FP32 and belongs to the caller: ``initial_state`` is optional,
    while ``(o, final_state)`` is always returned. The in-tree implementation
    currently supports SM90 dense prefill; packed varlen and decode remain
    part of the public contract for external targets and future kernels.

    ``beta`` contains the already-transformed update strength. This Op does
    not apply a sigmoid or another beta transform.
    """

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "deltanet_dense_prefill": DeltaNetDensePrefillFwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "deltanet_inference": DeltaNetInferenceFwdInterface
    }

    def __init__(
        self,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Configure the attention scale, Q/K normalization, and target.

        Args:
            scale: Query scale, or ``None`` for ``K**-0.5``.
            use_qk_l2norm_in_kernel: Normalize Q and K internally.
            target: Backend target, or ``None`` to resolve from the input
                device.
            kernel_map: Optional in-tree kernel overrides.
            tune: Autotune a kernel when it is first built.
        """
        self.scale = scale
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def compute_roof(self) -> str:
        """The state contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

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
        return self._call_boundary(q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator."""
        inputs = tuple(
            tensor.contiguous() if tensor is not None else None
            for tensor in (q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu)
        )
        batch, seq_len, heads, dim_k = q.shape
        call = DeltaNetInferenceCall(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim_k=dim_k,
            dim_v=v.shape[-1],
            dtype=q.dtype,
            scale=self.scale if self.scale is not None else dim_k**-0.5,
            l2norm=self.use_qk_l2norm_in_kernel,
            varlen=cu_seqlens is not None,
            device=q.device,
        )
        return self.kernel_for("deltanet_inference", call)(*inputs)
