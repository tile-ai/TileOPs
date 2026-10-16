"""Three-consumer dense prefill for 64-dimensional attention heads."""

from typing import ClassVar, Optional

from tileops.kernels.attention.call_spec import AttentionCall
from tileops.kernels.attention.gqa.dense import GQADenseWSKernel

__all__ = ["GQADenseThreeConsumerKernel"]


class GQADenseThreeConsumerKernel(GQADenseWSKernel):
    """Process 192 query rows with three consumer warpgroups on SM90."""

    preferred_over = frozenset({"gqa_dense"})
    _block_m: ClassVar[int] = 192

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """Serve the measured D=64 region of contiguous prefill."""
        if call.dim != 64:
            return "requires head dimension 64"
        return super().refusal(call)
