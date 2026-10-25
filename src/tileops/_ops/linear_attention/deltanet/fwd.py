"""Inference-facing DeltaNet forward contract, and its dense prefill and decode dispatch."""

from typing import ClassVar, Mapping, Optional, Tuple

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.linear_attention import (
    DeltaNetCall,
    DeltaNetChunkPrefillFwdKernel,
    DeltaNetDenseDecodeFwdKernel,
    DeltaNetDensePrefillFwdKernel,
    DeltaNetFwdInterface,
)
from tileops._ops._offsets import packed_sequence_check
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["DeltaNetFwdOp"]


class DeltaNetFwdOp(Op):
    """Inference forward for the ungated delta rule.

    The input layout is ``[B, T, H, D]``. One call covers equal-length
    prefill, packed-varlen prefill, and single-token decode. The recurrent
    state is FP32 and belongs to the caller: ``initial_state`` is optional,
    while ``(o, final_state)`` is always returned.

    The in-tree implementations serve a 64- or 128-wide square state in
    float16 or bfloat16, with an even or single head count, and with or
    without ``use_qk_l2norm_in_kernel``. Prefill runs on SM89 and SM90,
    equal-length or packed, with a sequence that is not a whole number of
    64-token chunks. On SM89, shared memory holds the offsets of at most 703
    packed sequences over a 128-wide state. Decode runs one token per
    sequence on SM80, SM89 and SM90, and does not take ``cu_seqlens``. A key
    width that is neither 64 nor 128, or one that differs from the value
    width, has no in-tree kernel and needs an external target implementation.

    ``beta`` contains the already-transformed update strength. This Op does
    not apply a sigmoid or another beta transform.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "deltanet_dense_decode": DeltaNetDenseDecodeFwdKernel,
        "deltanet_dense_prefill": DeltaNetDensePrefillFwdKernel,
        "deltanet_chunk_prefill": DeltaNetChunkPrefillFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"deltanet": DeltaNetFwdInterface}

    def __init__(
        self,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        *,
        validate_inputs: bool = False,
        target: Target = None,
    ) -> None:
        """Configure the attention scale, Q/K normalization, and target.

        Args:
            scale: Query scale, or ``None`` for ``K**-0.5`` (K the key head dim).
            use_qk_l2norm_in_kernel: Normalize Q and K internally.
            validate_inputs: Check on the CPU that ``cu_seqlens`` (and
                ``cu_seqlens_cpu``) start at 0, never decrease and end at ``S``.
                It synchronizes the device; leave it off under CUDA Graph capture.
            target: Backend target, or ``None`` to resolve from the input
                device.
        """
        self.scale = scale
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def roof_key(self) -> str:
        """The state contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

    _check_input_values = packed_sequence_check(5)

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
        """Run prefill or decode and return ``(o, final_state)``.

        Args:
            q: Queries, $[B \\times S \\times H \\times K]$, ``float16`` or ``bfloat16``.
            k: Keys, $[B \\times S \\times H \\times K]$, same dtype as ``q``.
            v: Values, $[B \\times S \\times H \\times V]$, same dtype as ``q``.
            beta: Update strength, already transformed, $[B \\times S \\times H]$, same dtype
                as ``q``.
            initial_state: ``float32`` starting state, $[N \\times H \\times K \\times V]$,
                or ``None``.
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
            for tensor in (q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu)
        )
        batch, seq_len, heads, dim_k = q.shape
        return self.kernel_for(
            "deltanet",
            lambda: DeltaNetCall(
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
            ),
        )(*inputs)
