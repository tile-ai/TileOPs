from typing import Dict, Optional
import torch

from top.ops.op import Op
from top.kernels.kernel import Kernel
from top.kernels.fuse_moe.reduce import ReduceKernel

__all__ = ["MoeReduceOp"]


class MoeReduceOp(Op):
    """MoE Reduce Op."""

    def __init__(self,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        self.config = {} if config is None else dict(config)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["ReduceKernel"](
            num_seq=1,
            hidden_size=1,
            num_topk=1,
            config=self.config if self.config else None,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ReduceKernel": ReduceKernel}

    def forward(self, x: torch.Tensor, topk_pos: torch.Tensor, topk_scale: torch.Tensor, shared_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of MoeReduceOp.

        Args:
            x: Expert output tokens [total_tokens, hidden_size]
            topk_pos: Position mapping from count_and_gather [num_seq, num_topk]
            topk_scale: Scaling factors for each token [num_seq, num_topk]
            shared_output: Optional shared output tensor [num_seq, hidden_size]

        Returns:
            output: Reduced output tensor [num_seq, hidden_size]
        """
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D tensor, got {x.ndim}D")

        if topk_pos.ndim != 2:
            raise ValueError(f"Expected topk_pos to be 2D tensor, got {topk_pos.ndim}D")

        if topk_scale.ndim != 2:
            raise ValueError(f"Expected topk_scale to be 2D tensor, got {topk_scale.ndim}D")

        if topk_pos.shape != topk_scale.shape:
            raise ValueError(
                f"Mismatched shape between topk_pos and topk_scale: {topk_pos.shape} vs {topk_scale.shape}"
            )

        if shared_output is not None:
            if shared_output.ndim != 2:
                raise ValueError(f"Expected shared_output to be 2D tensor, got {shared_output.ndim}D")

            if shared_output.shape[0] != topk_pos.shape[0]:
                raise ValueError(
                    f"Mismatched batch size: shared_output has {shared_output.shape[0]}, topk_pos has {topk_pos.shape[0]}"
                )

            if shared_output.shape[1] != x.shape[1]:
                raise ValueError(
                    f"Mismatched hidden size: shared_output has {shared_output.shape[1]}, x has {x.shape[1]}"
                )

        num_seq, num_topk = topk_pos.shape
        self.kernel.num_seq = num_seq
        self.kernel.hidden_size = x.shape[1]
        self.kernel.num_topk = num_topk
        return self.kernel(x, topk_pos, topk_scale, shared_output)
