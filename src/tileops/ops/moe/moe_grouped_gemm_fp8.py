"""Block-scaled FP8 M-grouped GEMM over masked expert slabs.

Provides:
  - MoEGroupedGemmFP8FwdOp: out[g, :m_g] = (a[g, :m_g] * a_scale) @ (b[g] * b_scale)^T in bf16,
    deep_gemm.m_grouped_fp8_gemm_nt_masked semantics
"""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.moe import (
    MGroupedGemmFP8Call,
    MGroupedGemmFP8FwdInterface,
    MoEGroupedGemmFP8Kernel,
)
from tileops.ops.moe.contracts import MaskedLayoutSpec
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["MoEGroupedGemmFP8FwdOp"]


class MoEGroupedGemmFP8FwdOp(Op):
    """Block-scaled FP8 GEMM per expert over a masked slab: ``out[g, :m_g] = a[g, :m_g] @ b[g]^T``.

    ``a`` is ``[E, max_m, K]`` and ``b`` ``[E, N, K]``, both ``float8_e4m3fn``. ``a_scale``
    holds one float32 scale per row and 128 columns of ``a`` (``[E, max_m, K / 128]``),
    ``b_scale`` one per 128x128 block of ``b`` (``[E, N / 128, K / 128]``); ``K`` and ``N``
    are multiples of 128. ``layout_metadata[g]`` is expert ``g``'s valid row count ``m_g``.

    Each 128-wide K block is contracted from the raw FP8 values, then multiplied by its
    two scales and added to a float32 accumulator, so the scales are applied once per
    128 K, as ``deep_gemm.m_grouped_fp8_gemm_nt_masked`` does. The result is written in
    bfloat16. Rows at or past ``layout_metadata[g]`` hold undefined values. The counts are
    not checked, since checking would synchronise: a count past ``max_m`` is read as
    ``max_m`` and a negative one as 0, and nothing outside ``out`` is written.

    Accuracy: on the valid rows the output matches an fp32 reference that dequantizes both
    operands under their scales to ``atol = 2e-2 * sqrt(max(1, K / 1024))`` and
    ``rtol = 2e-2`` (``workloads.gemm.fp8_matmul_verification``), a bound that grows with
    ``K`` as the fp8 accumulation noise does. The scales are applied in DeepGEMM's order;
    on the manifest's E = 8 rows the output is bit-identical to
    ``deep_gemm.m_grouped_fp8_gemm_nt_masked``.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "grouped_gemm_fp8": MoEGroupedGemmFP8Kernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "grouped_gemm_fp8": MGroupedGemmFP8FwdInterface
    }

    def roofline_inputs(self) -> "dict[str, int]":
        """The valid rows this call's layout metadata marks, which its flops follow."""
        from tileops.perf.formulas import moe_layout_rows

        return {"valid_rows": moe_layout_rows(self.last_call)}

    def eval_roofline_read_bytes(self) -> int:
        """``bytes`` less the write of ``output``, which a masked slab makes on its valid rows
        only, where the base class would take the whole output off."""
        from tileops.perf.formulas import moe_slab_bytes

        return int(self.eval_roofline()[1]) - moe_slab_bytes(self.last_call, "output")

    def __init__(
        self,
        layout: MaskedLayoutSpec,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Fix the masked expert layout.

        Args:
            layout: The masked slab capacity ``max_m`` each expert's rows are padded to.
            target: Which backend serves this instance; detected from the tensors when
                ``None``.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune the kernel.
        """
        self.layout = layout
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def compute_roof(self) -> str:
        """FLOPs are FP8 matmul contractions; priced on tensor cores."""
        return tensor_core_roof(torch.float8_e4m3fn)

    def forward(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one block-scaled GEMM per expert over its valid rows.

        Args:
            a: ``float8_e4m3fn`` activations, ``[E, max_m, K]``.
            a_scale: ``float32`` per-row scales of ``a``, ``[E, max_m, K / 128]``.
            b: ``float8_e4m3fn`` per-expert weights, ``[E, N, K]``.
            b_scale: ``float32`` per-block scales of ``b``, ``[E, N / 128, K / 128]``.
            layout_metadata: ``int32`` valid row count of each expert, ``[E]``.
            out: Optional preallocated contiguous ``bfloat16`` output, ``[E, max_m, N]``,
                written in place and returned; ``None`` allocates one.

        Returns:
            ``[E, max_m, N]`` in ``bfloat16``; rows at or past ``layout_metadata[g]`` are
            undefined.
        """
        return self._call_boundary(a, a_scale, b, b_scale, layout_metadata, out)

    def _eager_forward(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a_scale, b_scale = a_scale.contiguous(), b_scale.contiguous()
        num_experts, n, k = b.shape
        call = MGroupedGemmFP8Call(
            device=a.device,
            kind=self.layout.kind,
            max_m=self.layout.max_m,
            ab_dtype=a.dtype,
            num_groups=num_experts,
            n=n,
            k=k,
            a_scale_shape=tuple(a_scale.shape),
            b_scale_shape=tuple(b_scale.shape),
        )
        kernel = self.kernel_for("grouped_gemm_fp8", call)
        return kernel(a, a_scale, b, b_scale, layout_metadata, out=out)
