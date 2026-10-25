"""MoE permute-align op: routes tokens to experts and pads to tile boundary."""

from typing import ClassVar, Mapping, Tuple

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.moe import (
    MoEPermuteAlignKernel,
    PermuteAlignCall,
    PermuteAlignFwdInterface,
)
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["MoEPermuteAlignFwdOp"]


class MoEPermuteAlignFwdOp(Op):
    """Route tokens to experts and pad each expert's token count to block_size.

    Takes ``topk_ids`` and produces the three index arrays required by MoE
    grouped GEMM: sorted token indices, per-block expert ids, and the total
    padded token count.

    With ``numel = T * K`` routes and ``count[e]`` of them on expert ``e``, the result is:

    - ``num_tokens_post_pad``: the sum over experts of ``count[e]`` rounded up to
      ``block_size``.
    - ``sorted_token_ids[:num_tokens_post_pad]``: one range per expert, in expert order, of
      that rounded-up length. The first ``count[e]`` slots of the range hold the indices of
      the routes on ``e``, each once, in an order that is unspecified and may differ
      between calls on the same input; the rest of the range holds ``numel``.
    - ``expert_ids[:num_tokens_post_pad // block_size]``: the owner of each ``block_size``
      block of those ranges.
    - Past those lengths, every entry of ``sorted_token_ids`` is ``numel`` and every entry
      of ``expert_ids`` lies outside ``[0, num_experts)``.

    Example:
        ```python linenums="1"
        op = MoEPermuteAlignFwdOp(num_experts=8, block_size=16)
        sorted_ids, expert_ids, num_post_pad = op(topk_ids)
        ```
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"permute_align": MoEPermuteAlignKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "permute_align": PermuteAlignFwdInterface
    }

    def __init__(
        self,
        num_experts: int,
        block_size: int = 64,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. The routed extents are taken from each call.

        Args:
            num_experts: Number of experts.
            block_size: GEMM tile size (M dimension); default 64.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.num_experts = num_experts
        self.block_size = block_size
        super().__init__(target=target)

    def forward(self, topk_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run permute-align.

        Args:
            topk_ids: [T, K] int32 expert indices (0-indexed).

        Returns:
            sorted_token_ids: [T * K + (num_experts + 1) * (block_size - 1)] int32
            expert_ids:       [ceil_div(that, block_size)] int32
            num_tokens_post_pad: [1] int32
        """
        return self.kernel_for(
            "permute_align",
            lambda: PermuteAlignCall(
                num_routes=topk_ids.numel(),
                num_experts=self.num_experts,
                block_size=self.block_size,
                device=topk_ids.device,
            ),
        )(topk_ids)
