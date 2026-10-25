"""Inference-facing GLA contract with dense-prefill and decode dispatch."""

from typing import ClassVar, Mapping, Optional, Tuple

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.linear_attention.gla.call_spec import (
    GLACall,
    GLAFwdInterface,
)
from tileops._kernels.linear_attention.gla.dense_decode import GLADenseDecodeFwdKernel
from tileops._kernels.linear_attention.gla.dense_prefill_partitioned import (
    GLADensePrefillPartitionedKernel,
)
from tileops._kernels.linear_attention.gla.dense_prefill_subchunk import (
    GLADensePrefillSubchunkKernel,
)
from tileops._kernels.linear_attention.gla.varlen_prefill import GLAVarlenPrefillFwdKernel
from tileops._kernels.linear_attention.gla.varlen_prefill_partitioned import (
    GLAVarlenPrefillPartitionedFwdKernel,
)
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["GLAFwdOp"]


class GLAFwdOp(Op):
    """Gated Linear Attention (GLA) for inference, with caller-owned FP32 state.

    Q, K, V and the log-space, per-key gate G use FP16/BF16 BTHD layout. One call is
    equal-length prefill, packed-varlen prefill, or single-token decode. An absent
    ``initial_state`` starts from zero; every call returns ``(o, final_state)``. The
    in-tree kernels serve equal-length prefill, packed-varlen prefill and decode.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gla_dense_decode": GLADenseDecodeFwdKernel,
        "gla_dense_prefill_partitioned": GLADensePrefillPartitionedKernel,
        "gla_dense_prefill_subchunk": GLADensePrefillSubchunkKernel,
        "gla_varlen_prefill": GLAVarlenPrefillFwdKernel,
        "gla_varlen_prefill_partitioned": GLAVarlenPrefillPartitionedFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gla": GLAFwdInterface}

    def __init__(
        self,
        scale: Optional[float] = None,
        *,
        validate_inputs: bool = False,
        target: Target = None,
    ) -> None:
        """Fix the query scale and optional backend target for this instance.

        Args:
            scale: Positive query scale, or ``None`` for ``K**-0.5`` (K the key head dim).
            validate_inputs: Check on the CPU that ``cu_seqlens`` (and
                ``cu_seqlens_cpu``) start at 0, never decrease and end at ``S``.
                It synchronizes the device; leave it off under CUDA Graph capture.
            target: Backend target, or ``None`` to resolve from the input
                device.
        """
        self.scale = scale
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def roof_key(self) -> str:
        """Prefill contracts chunks on tensor cores; a decode step is a matvec on CUDA cores."""
        q_shape, q_dtype = self.last_call.tensors["q"]
        return super().roof_key() if q_shape[1] == 1 else tensor_core_roof(q_dtype)

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
        """Run GLA prefill or decode and return output plus final FP32 state.

        Args:
            q: Queries, $[B \\times S \\times H \\times K]$, ``float16`` or ``bfloat16``.
            k: Keys, $[B \\times S \\times H \\times K]$, same dtype as ``q``.
            v: Values, $[B \\times S \\times H \\times V]$, same dtype as ``q``.
            g: Log-space per-key gate, $[B \\times S \\times H \\times K]$, same dtype as ``q``.
            initial_state: ``float32`` starting state, $[N \\times H \\times K \\times V]$,
                or ``None`` to start from zero.
            cu_seqlens: ``int64`` offsets of the packed sequences, $[N + 1]$, or ``None`` for
                equal-length input. Requires $B = 1$.
            cu_seqlens_cpu: The same offsets on the CPU, or ``None``. Requires ``cu_seqlens``.

        Returns:
            ``(o, final_state)``. ``o`` is the output, $[B \\times S \\times H \\times V]$, in
            the dtype of ``q``. ``final_state`` is the ``float32`` state after the last
            token, $[N \\times H \\times K \\times V]$. $N = B$ unless ``cu_seqlens`` is given.
        """
        inputs = tuple(
            tensor.contiguous() if tensor is not None else None
            for tensor in (q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu)
        )
        batch, seq_len, heads, dim_k = q.shape
        return self.kernel_for(
            "gla",
            lambda: GLACall(
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
            ),
        )(*inputs)
