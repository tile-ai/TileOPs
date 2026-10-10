"""Inference-facing DeltaNet forward contract, and its dense prefill and decode dispatch."""

from typing import ClassVar, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.linear_attention import (
    DeltaNetCall,
    DeltaNetDenseDecodeFwdKernel,
    DeltaNetDensePrefillFwdKernel,
    DeltaNetFwdInterface,
)
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["DeltaNetFwdOp"]


class DeltaNetFwdOp(Op):
    """Inference forward for the ungated delta rule.

    The input layout is ``[B, T, H, D]``. One call covers equal-length
    prefill, packed-varlen prefill, and single-token decode. The recurrent
    state is FP32 and belongs to the caller: ``initial_state`` is optional,
    while ``(o, final_state)`` is always returned.

    The in-tree implementations serve SM90. Prefill runs over a 64- or
    128-wide square state in float16 or bfloat16, equal-length or packed, with
    a sequence that is not a whole number of 64-token chunks, and with or
    without ``use_qk_l2norm_in_kernel``. Decode runs one token over a 128-wide
    square state with Q and K already normalized. A key width that is neither
    64 nor 128, or one that differs from the value width, has no in-tree
    kernel and needs an external target implementation.

    ``beta`` contains the already-transformed update strength. This Op does
    not apply a sigmoid or another beta transform.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "deltanet_dense_decode": DeltaNetDenseDecodeFwdKernel,
        "deltanet_dense_prefill": DeltaNetDensePrefillFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"deltanet": DeltaNetFwdInterface}

    def __init__(
        self,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        *,
        target: Target = None,
    ) -> None:
        """Configure the attention scale, Q/K normalization, and target.

        Args:
            scale: Query scale, or ``None`` for ``K**-0.5``.
            use_qk_l2norm_in_kernel: Normalize Q and K internally.
            target: Backend target, or ``None`` to resolve from the input
                device.
        """
        self.scale = scale
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        super().__init__(target=target)

    def roof_key(self) -> str:
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
        inputs = tuple(
            tensor.contiguous() if tensor is not None else None
            for tensor in (q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu)
        )
        batch, seq_len, heads, dim_k = q.shape
        call = DeltaNetCall(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim_k=dim_k,
            dim_v=v.shape[-1],
            dtype=q.dtype,
            scale=self.scale if self.scale is not None else dim_k**-0.5,
            l2norm=self.use_qk_l2norm_in_kernel,
            varlen=cu_seqlens is not None,
            has_initial_state=initial_state is not None,
            num_sequences=batch if cu_seqlens is None else cu_seqlens.numel() - 1,
            device=q.device,
        )
        return self.kernel_for("deltanet", call)(*inputs)
