from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.quantization.fp8_quant import (
    FP8QuantCall,
    FP8QuantFwdInterface,
    FP8QuantKernel,
)
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["FP8QuantPerTokenGroupFwdOp"]


class FP8QuantPerTokenGroupFwdOp(Op):
    """Quantize each group of columns to FP8, into ``out_q`` in place when it is passed.

    vLLM's ``per_token_group_quant_fp8``. A group is ``group_size`` adjacent columns of the
    last axis.

    ``scale`` is the group's absolute maximum, floored at ``eps``, over 448, to within one
    float32 ulp. ``q`` is the group scaled by the reciprocal of that scale, so for a finite
    group every element is within one ``float8_e4m3fn`` code of ``clamp(x / scale, -448,
    448)`` rather than equal to it. With ``out_q`` the codes are written there and it is
    returned as ``q``.

    A group holding an infinity or a NaN has no defined result: whether the group maximum and
    the clamp carry the non-finite value depends on the kernel serving the call.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"fp8_quant_kernel": FP8QuantKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"fp8_quant": FP8QuantFwdInterface}

    def __init__(
        self,
        group_size: int,
        eps: float = 1e-10,
        dtype: torch.dtype | None = None,
        column_major_scales: bool = False,
        tma_aligned_scales: bool = False,
        use_ue8m0: bool = False,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and the input dtype are taken from the first call.

        Args:
            group_size: Columns per group; divides the last axis.
            eps: Floor of each group's absolute maximum.
            dtype: The codes' dtype, ``torch.float8_e4m3fn``, or None for it.
            column_major_scales: Lay the scales out column-major.
            tma_aligned_scales: Pad the column-major scales to TMA alignment.
            use_ue8m0: Round each scale up to a power of two.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.group_size = group_size
        self.eps = eps
        self.dtype = dtype
        self.column_major_scales = column_major_scales
        self.tma_aligned_scales = tma_aligned_scales
        self.use_ue8m0 = use_ue8m0
        super().__init__(target=target)

    def forward(
        self, x: torch.Tensor, out_q: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize ``x`` group by group.

        Args:
            x: $[*S \\times K]$, ``float16``, ``bfloat16`` or ``float32``.
            out_q: $[*S \\times K]$ ``float8_e4m3fn`` buffer the codes are written to, or
                None.

        Returns:
            ``q`` $[*S \\times K]$ in ``float8_e4m3fn`` (``out_q`` where given) and
            ``scale`` $[*S \\times K / group\\_size]$ in ``float32``.
        """
        x = x.contiguous()
        rows = x.numel() // self.group_size
        kernel = self.kernel_for(
            "fp8_quant",
            lambda: FP8QuantCall(
                rows=rows,
                group_size=self.group_size,
                eps=self.eps,
                dtype=x.dtype,
                device=x.device,
                column_major_scales=self.column_major_scales,
                tma_aligned_scales=self.tma_aligned_scales,
                use_ue8m0=self.use_ue8m0,
            ),
        )
        scale, q = kernel(x.reshape(rows, self.group_size))
        q = q.view(x.shape)
        scale = scale.view(*x.shape[:-1], x.shape[-1] // self.group_size)
        if out_q is None:
            return q, scale
        out_q.copy_(q)
        return out_q, scale
