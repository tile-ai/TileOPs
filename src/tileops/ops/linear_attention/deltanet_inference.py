"""Inference-facing DeltaNet forward contract and dense-prefill dispatch."""

from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.deltanet.dense_prefill import (
    DeltaNetDensePrefillFwdKernel,
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
