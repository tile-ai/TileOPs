"""SharedFusedMoE — FusedMoE with shared expert support.

Combines routed experts (via FusedMoe) with shared experts (SharedExpertMLPKernel).

Usage (single GPU, tp_size=1):
    op = SharedFusedMoE(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F,
        shared_ffn_size=F_s,
    )
    shared_out, routed_out = op(
        hidden, gating, w_gate_up, w_down,
        shared_w_gate_up=shared_w_gate_up,  # [2*F_s, H]
        shared_w_down=shared_w_down,         # [H, F_s]
    )

Usage (TP, tp_size>1):
    op = SharedFusedMoE(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F,
        shared_ffn_size=F_s,
        tp_size=tp_size, tp_rank=tp_rank,
    )
    # Pass complete weights; op shards them internally per tp_rank.
    # shared_out is a partial result — caller must all-reduce across TP ranks.
    shared_out_partial, routed_out = op(
        hidden, gating, w_gate_up, w_down,
        shared_w_gate_up=shared_w_gate_up,  # [2*F_s, H]  complete
        shared_w_down=shared_w_down,         # [H, F_s]   complete
    )
    # dist.all_reduce(shared_out_partial)  ← caller's responsibility
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.moe import SharedExpertMLPKernel
from tileops.ops.moe.fused_moe import FusedMoe

__all__ = ["SharedFusedMoE"]


class SharedFusedMoE(FusedMoe):
    """FusedMoE with shared expert support, optionally TP-aware.

    Extends FusedMoe to compute both shared and routed expert outputs.
    The shared expert is computed via SharedExpertMLPKernel (TileLang).

    TP support (shared expert only):
        When tp_size > 1, the op shards the shared expert weights internally:
          - shared_w_gate_up [2*F_s, H] is split along dim=0 (ColumnParallel)
          - shared_w_down    [H, F_s]   is split along dim=1 (RowParallel)
        The returned shared_out is a partial sum; the caller must all-reduce
        across TP ranks. The routed expert path is not affected.

    Args:
        shared_ffn_size: FFN intermediate size for the shared expert (full size,
            before TP sharding). If None, no shared expert is computed.
        tp_size: Tensor parallel world size. Default 1 (no TP).
        tp_rank: This rank's index in the TP group. Default 0.
        Other args: same as FusedMoe.

    Returns:
        (shared_output, routed_output): tuple of [T, H] tensors.
            shared_output is None when shared_ffn_size is None.
            shared_output is a partial sum when tp_size > 1.
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        with_correction_bias: bool = False,
        routed_scaling_factor: float = 1.0,
        layout: str = "nopad",
        dtype: torch.dtype = torch.bfloat16,
        expert_map: Optional[torch.Tensor] = None,
        shared_ffn_size: Optional[int] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            scoring_func=scoring_func,
            renormalize=renormalize,
            with_correction_bias=with_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            layout=layout,
            dtype=dtype,
            expert_map=expert_map,
        )

        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if not (0 <= tp_rank < tp_size):
            raise ValueError(f"tp_rank must be in [0, tp_size), got tp_rank={tp_rank}, tp_size={tp_size}")
        if shared_ffn_size is not None and shared_ffn_size % tp_size != 0:
            raise ValueError(
                f"shared_ffn_size ({shared_ffn_size}) must be divisible by tp_size ({tp_size})"
            )

        self.shared_ffn_size = shared_ffn_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # Kernel operates on the local shard size
        self._shared_mlp_kernel = (
            SharedExpertMLPKernel(
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                ffn_size=shared_ffn_size // tp_size,
                dtype=dtype,
            )
            if shared_ffn_size is not None
            else None
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        correction_bias: Optional[torch.Tensor] = None,
        shared_w_gate_up: Optional[torch.Tensor] = None,
        shared_w_down: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Run shared + routed MoE FFN.

        Args:
            hidden_states: [T, H] input hidden states.
            gating_output: [T, E] gating logits.
            w_gate_up: [E, 2F, H] routed expert gate+up weights.
            w_down: [E, H, F] routed expert down weights.
            correction_bias: Optional [E] bias for Kimi-style routing.
            shared_w_gate_up: [2*F_s, H] shared expert gate+up weights (full).
                Required when shared_ffn_size is not None.
                When tp_size > 1, sharded along dim=0 internally.
            shared_w_down: [H, F_s] shared expert down weight (full).
                Required when shared_ffn_size is not None.
                When tp_size > 1, sharded along dim=1 internally.

        Returns:
            (shared_output, routed_output): tuple of [T, H] tensors.
                shared_output is None when shared_ffn_size is None.
                shared_output is a partial sum when tp_size > 1;
                caller must all-reduce across TP ranks.
        """
        if self._shared_mlp_kernel is not None:
            if shared_w_gate_up is None or shared_w_down is None:
                raise ValueError(
                    "shared_w_gate_up and shared_w_down must be provided "
                    "when shared_ffn_size is set"
                )
            # TP sharding: ColumnParallel on gate_up (dim=0), RowParallel on down (dim=1)
            if self.tp_size > 1:
                F_s = self.shared_ffn_size
                shard_size = F_s // self.tp_size
                gate_up_shard = shared_w_gate_up.narrow(0, self.tp_rank * shard_size * 2, shard_size * 2).contiguous()
                down_shard = shared_w_down.narrow(1, self.tp_rank * shard_size, shard_size).contiguous()
            else:
                gate_up_shard = shared_w_gate_up
                down_shard = shared_w_down

            shared_out = self._shared_mlp_kernel(hidden_states, gate_up_shard, down_shard)
        else:
            shared_out = None

        routed_out = super().forward(
            hidden_states, gating_output, w_gate_up, w_down, correction_bias
        )

        return shared_out, routed_out
