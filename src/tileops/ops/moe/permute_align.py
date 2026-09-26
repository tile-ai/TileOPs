"""MoE permute-align op: routes tokens to experts and pads to tile boundary."""

from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.moe import MoePermuteAlignKernel

from ..op_base import Op

__all__ = ["MoePermuteAlignFwdOp"]


class MoePermuteAlignFwdOp(Op):
    """Route tokens to experts and pad each expert's token count to block_size.

    Takes ``topk_ids`` and produces the three index arrays required by MoE
    grouped GEMM: sorted token indices, per-block expert ids, and the total
    padded token count.

    Example:
        ```python linenums="1"
        op = MoePermuteAlignFwdOp(num_experts=8, block_size=16)
        sorted_ids, expert_ids, num_post_pad = op(topk_ids)
        ```
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "permute_align_kernel": MoePermuteAlignKernel
    }

    def __init__(
        self,
        num_experts: int,
        block_size: int = 64,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. The routed extents are taken from each call.

        Args:
            num_experts: Number of experts.
            block_size: GEMM tile size (M dimension); default 64.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune the kernel.
        """
        self.num_experts = num_experts
        self.block_size = block_size
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: int) -> Entry:
        """One implementation, built per routed count ``T * K``."""
        return call, lambda: self.kernel_map[role](
            call, self.num_experts, self.block_size, tune=self.tune
        )

    def forward(self, topk_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run permute-align.

        Args:
            topk_ids: [T, K] int32 expert indices (0-indexed).

        Returns:
            sorted_token_ids: [T * K + (num_experts + 1) * (block_size - 1)] int32
            expert_ids:       [ceil_div(that, block_size)] int32
            num_tokens_post_pad: [1] int32
        """
        return self._call_boundary(topk_ids)

    def _eager_forward(
        self, topk_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Launch inside the operator, where dynamo does not follow the kernel call."""
        kernel = self.kernel_for("permute_align_kernel", (topk_ids,), topk_ids.numel())
        return kernel(topk_ids)
