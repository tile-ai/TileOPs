"""MoE permute-align op: routes tokens to experts and pads to tile boundary."""

from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.moe import MoePermuteAlignKernel

from .._compile_boundary_codegen import OperatorSpec
from ..op_base import Op

__all__ = ["MoePermuteAlignFwdOp"]


class MoePermuteAlignFwdOp(Op):
    """Route tokens to experts and pad each expert's token count to block_size.

    Takes ``topk_ids`` and produces the three index arrays required by MoE
    grouped GEMM: sorted token indices, per-block expert ids, and the total
    padded token count.

    Example:
        ```python linenums="1"
        op = MoePermuteAlignFwdOp(total_tokens=4, top_k=8, num_experts=8, block_size=16)
        sorted_ids, expert_ids, num_post_pad = op(topk_ids)
        ```
    """

    compile_boundary: ClassVar[tuple[OperatorSpec, ...]] = (OperatorSpec(),)

    def __init__(
        self,
        total_tokens: int,
        top_k: int,
        num_experts: int,
        block_size: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            total_tokens: Number of input tokens T.
            top_k: Number of experts selected per token K.
            num_experts: Number of experts.
            block_size: GEMM tile size (M dimension); default 64.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune the kernel.
        """
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.block_size = block_size
        self.numel = total_tokens * top_k

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"permute_align_kernel": MoePermuteAlignKernel}

    def entry_for(self, role: str, call: object) -> Entry:
        """One implementation, built from the extents the caller committed to."""
        return call, lambda: self.kernel_map[role](
            self.numel, self.num_experts, self.block_size, tune=self.tune
        )

    def _padded_extents(self, numel: int) -> Tuple[int, int]:
        """``(max_padded, num_blocks)`` — the two padded extents the manifest states.

        Both appear verbatim in the manifest ``roofline.bytes`` expression, which is
        where the output extents of this op are written down.
        """
        max_padded = numel + (self.num_experts + 1) * (self.block_size - 1)
        return max_padded, (max_padded + self.block_size - 1) // self.block_size

    def _infer_output_shapes(
        self,
        topk_ids_shape: Tuple[int, ...],
    ) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``signature.outputs`` at the extents ``roofline.bytes`` declares.

        ``numel`` comes off ``topk_ids``: the manifest states
        ``topk_ids.shape == (total_tokens, top_k)``.
        """
        max_padded, num_blocks = self._padded_extents(topk_ids_shape[0] * topk_ids_shape[1])
        return {
            "sorted_token_ids": (max_padded,),
            "expert_ids": (num_blocks,),
            "num_tokens_post_pad": (1,),
        }

    def forward(self, topk_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run permute-align.

        Args:
            topk_ids: [total_tokens, top_k] int32 expert indices (0-indexed).

        Returns:
            sorted_token_ids: [max_num_tokens_padded] int32
            expert_ids:       [num_blocks] int32
            num_tokens_post_pad: [1] int32
        """
        return self._wrapped(topk_ids, self._instance_key)

    def _eager_forward(
        self, topk_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Launch inside the operator, where dynamo does not follow the kernel call."""
        kernel = self.kernel_for("permute_align_kernel", (topk_ids,), topk_ids.device)
        return kernel(topk_ids)
