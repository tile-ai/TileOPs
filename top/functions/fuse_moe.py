from typing import Optional, Tuple
import torch

from top.kernels.fuse_moe.count_and_gather import CountAndGatherKernel
from top.kernels.fuse_moe.moe_reduce import MoeReduceKernel


def count_and_gather(
    x: torch.Tensor,  # [num_seq, hidden_size]
    topk_ids: torch.Tensor,  # [num_seq, num_topk]
    num_expert: int,
    rank_ep: int = 0,
    tile_m: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Count and gather tokens by expert.

    Args:
        x: Input token features
        topk_ids: Expert assignment for each token
        num_expert: Number of experts
        rank_ep: Expert parallel rank
        tile_m: Tile size for M dimension

    Returns:
        gate_up_input: Gathered input for gate and up projection
        topk_pos: Position mapping for each token
        seqlens: Number of tokens per expert
        cu_seqlens: Cumulative sequence lengths
        tiles: Number of tiles per expert
    """
    # Input validation
    if x.ndim != 2:
        raise ValueError(f"Expected x to be 2D tensor, got {x.ndim}D")

    if topk_ids.ndim != 2:
        raise ValueError(f"Expected topk_ids to be 2D tensor, got {topk_ids.ndim}D")

    if x.shape[0] != topk_ids.shape[0]:
        raise ValueError(
            f"Mismatched batch size: x has {x.shape[0]}, topk_ids has {topk_ids.shape[0]}")

    num_seq, hidden_size = x.shape
    num_topk = topk_ids.shape[1]
    kernel = CountAndGatherKernel(
        num_seq=num_seq,
        hidden_size=hidden_size,
        num_topk=num_topk,
        num_expert=num_expert,
        config={"tile_m": tile_m},
    )
    return kernel.count_and_gather(x, topk_ids, rank_ep)


def reduce(
        x: torch.Tensor,  # [total_tokens, hidden_size]
        topk_pos: torch.Tensor,  # [num_seq, num_topk]
        topk_scale: torch.Tensor,  # [num_seq, num_topk]
        shared_output: Optional[torch.Tensor] = None,  # [num_seq, hidden_size]
) -> torch.Tensor:
    """Reduce (scatter-add) tokens back to original positions.

    Args:
        x: Expert output tokens
        topk_pos: Position mapping from count_and_gather
        topk_scale: Scaling factors for each token
        shared_output: Optional shared output tensor

    Returns:
        output: Reduced output tensor [num_seq, hidden_size]
    """
    # Input validation
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
    _, hidden_size = x.shape
    kernel = MoeReduceKernel(
        num_seq=num_seq,
        hidden_size=hidden_size,
        num_topk=num_topk,
    )
    return kernel.forward(x, topk_pos, topk_scale, shared_output)


def fuse_moe_pertensor_fp8(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scale: torch.Tensor,
    num_expert: int,
    rank_ep: int = 0,
) -> torch.Tensor:
    """Fused MoE with per-tensor FP8 quantization.

    Args:
        x: Input token features
        gate_up_weight: Gate and up projection weights
        down_weight: Down projection weights
        topk_ids: Expert assignment for each token
        topk_scale: Scaling factors for each token
        num_expert: Number of experts
        rank_ep: Expert parallel rank

    Returns:
        output: Fused MoE output
    """
    # TODO: Implement fuse_moe_pertensor_fp8 function
    # This will be implemented in a separate step
    raise NotImplementedError("fuse_moe_pertensor_fp8 function not implemented yet")
