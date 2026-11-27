"""Block-scaled FP8 M-grouped GEMM over masked expert slabs or contiguous expert rows, with
``deep_gemm.m_grouped_fp8_gemm_nt_masked`` and ``_nt_contiguous`` semantics."""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.moe import (
    MGroupedGEMMFP8Call,
    MGroupedGEMMFP8FwdInterface,
    MoEGroupedGEMMFP8Kernel,
    MoEGroupedGEMMFP8MMAKernel,
)
from tileops._ops.moe.contracts import ContiguousLayoutSpec, MaskedLayoutSpec, layout_call_fields
from tileops._ops.moe.staged import MoELayoutRowsRoofline
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["MoEGroupedGEMMFP8FwdOp"]


class MoEGroupedGEMMFP8FwdOp(MoELayoutRowsRoofline, Op):
    """Block-scaled FP8 GEMM per expert over its rows: ``out[rows of g] = a[rows of g] @ b[g]^T``.

    ``b`` is ``[E, N, K]`` and ``a`` holds the expert rows, both ``float8_e4m3fn``. ``a_scale``
    holds one float32 scale per row and 128 columns of ``a``, ``b_scale`` one per 128x128
    block of ``b`` (``[E, N / 128, K / 128]``); ``K`` and ``N`` are multiples of 128. The
    layout places the rows and fixes ``layout_metadata`` as ``MoEGroupedGEMMFwdOp``'s does;
    ``a_scale`` is ``a``'s shape with ``K / 128`` columns and ``out`` with ``N`` columns. Rows
    an aligned layout pads with, and a masked slab's rows past its valid count, hold
    unspecified values in ``out``.

    Each 128-wide K block is contracted from the raw FP8 values, then multiplied by its
    two scales and added to a float32 accumulator. The result is written in bfloat16. The
    metadata is not checked, since checking would synchronise: a masked count past ``max_m``
    is read as ``max_m`` and a negative one as 0, and nothing outside ``out`` is written.

    Accuracy: on the valid rows the output matches an fp32 reference that dequantizes both
    operands under their scales to ``atol = 2e-2 * sqrt(max(1, K / 1024))`` and
    ``rtol = 2e-2``.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "grouped_gemm_fp8": MoEGroupedGEMMFP8Kernel,
        "grouped_gemm_fp8_mma": MoEGroupedGEMMFP8MMAKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "grouped_gemm_fp8": MGroupedGEMMFP8FwdInterface
    }

    def __init__(
        self,
        layout: ContiguousLayoutSpec | MaskedLayoutSpec,
        *,
        target: Target = None,
    ) -> None:
        """Fix the expert layout.

        Args:
            layout: The masked slab capacity ``max_m`` each expert's rows are padded to, or
                the packing and metadata of contiguous expert rows.
            target: Which backend serves this instance; detected from the tensors when
                ``None``.
        """
        self.layout = layout
        super().__init__(target=target)

    def roof_key(self) -> str:
        """FLOPs are FP8 matmul contractions; priced on tensor cores."""
        return tensor_core_roof(torch.float8_e4m3fn)

    def forward(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        layout_metadata: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one block-scaled GEMM per expert over its rows.

        Args:
            a: ``float8_e4m3fn`` activations, ``[M, K]``, or ``[E, max_m, K]`` when masked.
            a_scale: ``float32`` per-row scales of ``a``, ``a``'s shape with ``K / 128``
                columns.
            b: ``float8_e4m3fn`` per-expert weights, ``[E, N, K]``.
            b_scale: ``float32`` per-block scales of ``b``, ``[E, N / 128, K / 128]``.
            layout_metadata: ``int32`` metadata of the layout (see the class docstring).
            out: Optional preallocated contiguous ``bfloat16`` output, ``a``'s shape with
                ``N`` columns, written in place and returned; ``None`` allocates one. Must not
                overlap any input, or a ``ValueError`` is raised before writing.

        Returns:
            ``a``'s shape with ``N`` columns in ``bfloat16``; the rows no expert owns are
            undefined.
        """
        a_scale, b_scale = a_scale.contiguous(), b_scale.contiguous()
        num_experts, n, k = b.shape
        kernel = self.kernel_for(
            "grouped_gemm_fp8",
            lambda: MGroupedGEMMFP8Call(
                device=a.device,
                **layout_call_fields(self.layout),
                ab_dtype=a.dtype,
                num_groups=num_experts,
                m=a.numel() // k,
                n=n,
                k=k,
                a_scale_shape=tuple(a_scale.shape),
                b_scale_shape=tuple(b_scale.shape),
            ),
        )
        return kernel(a, a_scale, b, b_scale, layout_metadata, out=out)
