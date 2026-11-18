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
    """Quantize each group of columns to FP8, or fake-quantize ``x`` in place.

    vLLM's ``per_token_group_quant_fp8``. A group is ``group_size`` adjacent columns of the
    last axis.

    Without ``use_ue8m0``, ``scale`` is the group's absolute maximum, floored at ``eps``, over
    448, to within one float32 ulp, and ``q`` is the group scaled by the reciprocal of that
    scale, so for a finite group every element is within one ``float8_e4m3fn`` code of
    ``clamp(x / scale, -448, 448)`` rather than equal to it. With ``out_q`` the codes are
    written there and it is returned as ``q``.

    A group holding an infinity or a NaN has no defined result: whether the group maximum and
    the clamp carry the non-finite value depends on the kernel serving the call. Neither has a
    group whose floored maximum is below ``448 / 3.4e38``, about ``1.3e-36``, which only an
    ``eps`` below that bound admits (an all-zero group under ``eps=0``, for one): the
    reciprocal of its scale overflows float32.

    ``use_ue8m0=None`` gives float32 scales, as vLLM resolves None where DeepGEMM's UE8M0
    scales are not in use; ``tma_aligned_scales`` takes effect only with
    ``column_major_scales``, as in vLLM.

    With ``use_ue8m0`` the scale is
    $2^{\\lceil \\log_2(\\max(amax, eps) \\cdot r) \\rceil}$, ``r`` the float32 nearest
    ``1 / 448``, and ``q`` is ``clamp(x / scale, -448, 448)`` rounded to nearest even
    exactly: dividing by a power of two loses nothing. With ``eps=1e-4`` this is
    DeepSeek-V4-Flash ``inference/kernel.py`` ``act_quant`` with ``scale_fmt="ue8m0"``, bit
    for bit. ``scale_dtype`` ``float8_e8m0fnu`` stores these power-of-two scales.

    With ``inplace``, ``x`` is overwritten with ``q * scale`` in its own dtype, where it lies
    (a column slice of wider rows is written without a copy), and returned in the third slot,
    ``q`` and ``scale`` being ``None``; otherwise the third slot is ``None``.
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
        use_ue8m0: bool | None = None,
        scale_dtype: torch.dtype = torch.float32,
        inplace: bool = False,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and the input dtype are taken from the first call.

        Args:
            group_size: Columns per group; divides the last axis.
            eps: Floor of each group's absolute maximum.
            dtype: The codes' dtype, ``torch.float8_e4m3fn``, or None for it.
            column_major_scales: Lay the scales out column-major.
            tma_aligned_scales: Pad the column-major scales to TMA alignment; read only with
                ``column_major_scales``.
            use_ue8m0: Round each scale up to a power of two; None for float32 scales.
            scale_dtype: ``torch.float32``, or ``torch.float8_e8m0fnu`` with ``use_ue8m0``.
            inplace: Overwrite ``x`` with its fake-quantized values instead of returning
                ``q`` and ``scale``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.group_size = group_size
        self.eps = eps
        self.dtype = dtype
        self.column_major_scales = column_major_scales
        self.tma_aligned_scales = tma_aligned_scales
        self.use_ue8m0 = use_ue8m0
        self.scale_dtype = scale_dtype
        self.inplace = inplace
        super().__init__(target=target)

    def forward(
        self, x: torch.Tensor, out_q: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Quantize ``x`` group by group.

        Args:
            x: $[*S \\times K]$, ``float16``, ``bfloat16`` or ``float32``; written in place
                under ``inplace``, a strided view included.
            out_q: $[*S \\times K]$ contiguous ``float8_e4m3fn`` buffer the codes are written
                to, or None.

        Returns:
            ``(q, scale, None)``: ``q`` $[*S \\times K]$ in ``float8_e4m3fn`` (``out_q`` where
            given) and ``scale`` $[*S \\times K / group\\_size]$ in ``scale_dtype``; with
            ``inplace``, ``(None, None, x)``.
        """
        if not self.inplace:
            # In place, x is written where it lies, a strided view included.
            x = x.contiguous()
        kernel = self.kernel_for(
            "fp8_quant",
            lambda: FP8QuantCall(
                k=x.shape[-1],
                group_size=self.group_size,
                eps=self.eps,
                dtype=x.dtype,
                device=x.device,
                column_major_scales=self.column_major_scales,
                tma_aligned_scales=self.tma_aligned_scales,
                use_ue8m0=self.use_ue8m0,
                scale_dtype=self.scale_dtype,
                inplace=self.inplace,
            ),
        )
        q, scale = kernel(x, out_q)
        if self.inplace:
            return None, None, x
        return q, scale, None
