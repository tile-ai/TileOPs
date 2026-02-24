from typing import Dict, Optional, Tuple
import torch

from top.ops.op import Op
from top.kernels.fuse_moe.count_and_gather import CountAndGatherKernel
from top.kernels.kernel import Kernel

__all__ = ["CountAndGatherOp"]


class CountAndGatherOp(Op):
    """Count and gather Op for MoE."""

    def __init__(self,
                 num_expert: int,
                 rank_ep: int = 0,
                 tile_m: int = 16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        self.num_expert = num_expert
        self.rank_ep = rank_ep
        self.config = {"tile_m": tile_m}
        if config is not None:
            self.config.update(config)

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["CountAndGatherKernel"](
            num_seq=1,
            hidden_size=1,
            num_topk=1,
            num_expert=self.num_expert,
            config=self.config,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"CountAndGatherKernel": CountAndGatherKernel}

    def forward(self, x: torch.Tensor, topk_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of CountAndGatherOp.

        Args:
            x: Input token features [num_seq, hidden_size]
            topk_ids: Expert assignment for each token [num_seq, num_topk]

        Returns:
            gate_up_input: Gathered input for gate and up projection
            topk_pos: Position mapping for each token
            seqlens: Number of tokens per expert
            cu_seqlens: Cumulative sequence lengths
            tiles: Number of tiles per expert
        """
        self.kernel.num_seq = x.shape[0]
        self.kernel.hidden_size = x.shape[1]
        self.kernel.num_topk = topk_ids.shape[1]
        return self.kernel(x, topk_ids, self.rank_ep)
