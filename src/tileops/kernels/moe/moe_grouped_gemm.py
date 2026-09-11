"""MoE adapter for the shared persistent grouped GEMM template."""

from typing import Optional

import torch

from tileops.kernels.grouped_gemm.heuristics import ACTIVATIONS, GemmType
from tileops.kernels.grouped_gemm.template import GroupedGemmTemplate
from tileops.kernels.kernel_base import Kernel

__all__ = ["MoeGroupedGemmKernel"]


class MoeGroupedGemmKernel(Kernel):
    """Adapt staged MoE grouped-GEMM calls to the shared GEMM template."""

    supported_archs: list[int] = [90]

    _TYPES: dict[tuple[str, Optional[str], Optional[str]], GemmType] = {
        ("contiguous", "tight", "physical_psum"): GemmType.M_GROUPED_TIGHT_PSUM,
        ("contiguous", "aligned", "physical_psum"): GemmType.M_GROUPED_ALIGNED_PSUM,
        ("contiguous", "aligned", "per_row"): GemmType.M_GROUPED_ALIGNED_PER_ROW,
        ("masked", None, None): GemmType.M_GROUPED_MASKED,
    }
    _ALIGNED_TILE_HEIGHTS = (64, 128, 256)

    @classmethod
    def applies(cls, call) -> bool:
        n_step = 8 if call.activation is None else 16
        return (
            (call.kind, call.packing, call.metadata_kind) in cls._TYPES
            and call.ab_dtype in (torch.bfloat16, torch.float16)
            and call.cd_dtype in (call.ab_dtype, torch.float32)
            and (call.packing != "aligned" or call.alignment in cls._ALIGNED_TILE_HEIGHTS)
            and (call.activation is None or call.activation in ACTIVATIONS)
            and call.k % 8 == 0
            and call.n % n_step == 0
        )

    def __init__(self, call) -> None:
        super().__init__()
        self.call = call
        self.inner = GroupedGemmTemplate(
            self._TYPES[(call.kind, call.packing, call.metadata_kind)],
            num_groups=call.num_groups,
            m_alignment=call.alignment if call.packing == "aligned" else 128,
            cd_dtype=None if call.cd_dtype is call.ab_dtype else call.cd_dtype,
            activation="none" if call.activation is None else call.activation,
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run each expert's grouped product, including a fused activation when requested."""
        return self.inner(a, b, grouped_layout=layout_metadata, out=out)
