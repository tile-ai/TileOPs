"""MXFP4 M-grouped GEMM over masked expert slabs or contiguous expert rows.

Provides:
  - MoEGroupedGEMMMXFP4FwdOp: each expert's FP8 rows, under their per-128 scales, times its
    packed e2m1 weights under their per-32 scales, in bf16; DeepSeek-V4-Flash
    ``inference/kernel.py`` ``fp4_gemm`` per expert
"""

import math
from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.moe import (
    MGroupedGEMMMXFP4Call,
    MGroupedGEMMMXFP4FwdInterface,
    MoEGroupedGEMMMXFP4Kernel,
    MoEGroupedGEMMMXFP4SharedDecodeKernel,
)
from tileops._ops.moe.contracts import ContiguousLayoutSpec, MaskedLayoutSpec
from tileops._ops.moe.staged import MoELayoutRowsRoofline
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["MoEGroupedGEMMMXFP4FwdOp"]


class MoEGroupedGEMMMXFP4FwdOp(MoELayoutRowsRoofline, Op):
    """MXFP4 GEMM per expert over its rows: ``out[rows of g] = a[rows of g] @ b[g]^T``.

    ``a`` holds the expert rows in ``float8_e4m3fn`` with one scale per row and 128 columns
    (``a_scale``, ``a``'s shape with ``K / 128`` columns). ``b`` is ``[E, N, K / 2]`` ``uint8``,
    two e2m1 (FP4) codes per byte along K, the even k in the low nibble, with one scale per 32
    of K (``b_scale``, ``[E, N, K / 32]``). Both scales are ``float8_e8m0fnu`` (Microscaling,
    MX, power-of-two scales) or both ``float32``; ``K`` is a multiple of 128. The layout places
    the rows, as ``MoEGroupedGEMMFwdOp``'s does:

    | layout                     | ``a``             | ``layout_metadata``           | ``out``           |
    | -------------------------- | ----------------- | ----------------------------- | ----------------- |
    | contiguous · physical_psum | ``[M, K]``        | ``[E]`` segment end rows      | ``[M, N]``        |
    | contiguous · per_row       | ``[M, K]``        | ``[M]`` expert id per row     | ``[M, N]``        |
    | masked                     | ``[E, max_m, K]`` | ``[E]`` valid rows per expert | ``[E, max_m, N]`` |

    Rows an aligned layout pads with, and a masked slab's rows past its valid count, hold
    unspecified values in ``out``. The metadata is not checked, since checking would
    synchronise: a masked count past ``max_m`` is read as ``max_m`` and a negative one as 0,
    and nothing outside ``out`` is written.

    The weights are decoded to fp8 (e4m3, which holds every e2m1 value) and contracted with
    ``a``'s fp8 rows on the tensor core, one 128-wide block of K at a time, each block added
    into a float32 accumulator under its scales. Under e8m0 scales a block's four weight
    sub-blocks enter one contraction scaled, exactly, relative to the largest of their four
    scales, and the block is added under that scale and the row's ``a_scale``; under float32
    scales each 32-wide sub-block is contracted alone and added under its two scales, as
    DeepSeek-V4-Flash's ``fp4_gemm`` does. A weight tile whose blocks' e8m0 scales span more
    than five binades, or holding a weight scale outside ``[2 ** -63, 2 ** 63]``, is
    contracted sub-block by sub-block, each product's scales applied by mantissa and
    exponent apart, so no scale product overflows or flushes before it is added. Other tiles
    apply the row's ``a_scale`` last, where a subnormal one (e8m0 code 0, ``2 ** -127``) is
    flushed to zero: a row under that scale loses results below ``K * 2 ** -52``. Aligned
    contiguous rows under e8m0 scales, the alignment a multiple of 128 and ``K`` a multiple of
    2048 whose code tables fit in shared memory (``K`` up to 10240 on SM90), instead decode each
    weight tile once into shared memory: there a tile folds when each weight row's scales lie in
    ``[2 ** -63, 2 ** 63]`` and span at most eleven binades over ``K`` and each ``a_scale`` of
    its rows, padding rows included, is code 0 (flushed as above) or lies in
    ``[2 ** -63, 2 ** 63]`` (a padding row left holding other codes sends its tile to the
    slower sub-block contraction); its sub-blocks enter the contraction scaled relative to a
    reference six binades under their row's largest scale (exact while the e2m1 values stay
    e4m3 normals), each block is added under the row's ``a_scale``, and each column is scaled
    by its reference once at the end. Other tiles there are contracted sub-block by sub-block
    as above. The float32 accumulator flushes terms and sums
    below ``2 ** -126`` to zero. The result is written in bfloat16.

    Accuracy: on the valid rows the output matches an fp32 reference that dequantizes both
    operands under their scales to ``atol = 2e-2 * sqrt(max(1, K / 1024))`` and
    ``rtol = 2e-2``, the FP8 GEMM bound, where the products do not cancel far below their
    magnitude. The scales apply to each block's contraction, not to the operands, so the
    products round apart from the reference's: where large products cancel, the error is
    relative to their magnitude and may pass that bound, as in ``MoEGroupedGEMMFP8FwdOp``. Where
    a dequantized value itself passes float32's range (an e8m0 scale of 2 ** 127 on a value of
    2 or more), the reference is not finite (infinite, or NaN where that value meets a zero)
    and the output is the sum of the scaled block products, each of which may overflow,
    rounded to bfloat16.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "grouped_gemm_mxfp4": MoEGroupedGEMMMXFP4Kernel,
        "grouped_gemm_mxfp4_shared_decode": MoEGroupedGEMMMXFP4SharedDecodeKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "grouped_gemm_mxfp4": MGroupedGEMMMXFP4FwdInterface
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
        """FLOPs are contractions on fp8 operands; priced on tensor cores."""
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
        """Run one MXFP4 GEMM per expert over its rows.

        Args:
            a: ``float8_e4m3fn`` activations, ``[M, K]``, or ``[E, max_m, K]`` when masked.
            a_scale: ``float8_e8m0fnu`` or ``float32`` scales of ``a``, ``a``'s shape with
                ``K / 128`` columns.
            b: ``uint8`` per-expert weights, ``[E, N, K / 2]``, two e2m1 codes per byte.
            b_scale: Scales of ``b`` in ``a_scale``'s dtype, ``[E, N, K / 32]``.
            layout_metadata: ``int32`` metadata of the layout (see the class docstring).
            out: Optional preallocated contiguous ``bfloat16`` output, ``a``'s shape with
                ``N`` columns, written in place and returned; ``None`` allocates one. Must not
                overlap any input, or a ``ValueError`` is raised before writing.

        Returns:
            ``a``'s shape with ``N`` columns in ``bfloat16``; the rows no expert owns are
            undefined.
        """
        num_experts, n, _ = b.shape
        k = a.shape[-1]
        layout = self.layout
        contiguous = layout.kind == "contiguous"
        kernel = self.kernel_for(
            "grouped_gemm_mxfp4",
            lambda: MGroupedGEMMMXFP4Call(
                device=a.device,
                kind=layout.kind,
                packing=layout.packing.value if contiguous else None,
                metadata_kind=layout.metadata_kind.value if contiguous else None,
                alignment=layout.alignment if contiguous else 1,
                max_m=layout.max_m,
                scale_dtype=a_scale.dtype,
                num_groups=num_experts,
                m=math.prod(a.shape[:-1]),
                n=n,
                k=k,
            ),
        )
        return kernel(a, a_scale, b, b_scale, layout_metadata, out=out)
