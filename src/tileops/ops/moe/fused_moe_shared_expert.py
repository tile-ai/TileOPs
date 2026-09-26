"""FusedMoeSharedExpertFwdOp — FusedMoE with shared expert support.

Combines routed experts (via FusedMoe) with shared experts (SharedExpertMLPKernel).

Usage (single GPU, tp_size=1):
    op = FusedMoeSharedExpertFwdOp(top_k=K)
    shared_out, routed_out = op(
        hidden, gating, w_gate_up, w_down,
        shared_w_gate_up=shared_w_gate_up,  # [2*F_s, H]
        shared_w_down=shared_w_down,         # [H, F_s]
    )

Usage (TP, tp_size>1):
    op = FusedMoeSharedExpertFwdOp(top_k=K, tp_size=tp_size, tp_rank=tp_rank)
    # Pass complete weights; op shards them internally per tp_rank.
    # shared_out is a partial result — caller must all-reduce across TP ranks.
    shared_out_partial, routed_out = op(
        hidden, gating, w_gate_up, w_down,
        shared_w_gate_up=shared_w_gate_up,  # [2*F_s, H]  complete
        shared_w_down=shared_w_down,         # [H, F_s]   complete
    )
    # dist.all_reduce(shared_out_partial, group=tp_group)  ← caller's responsibility
    # Must use the TP process group, not the default group (important in EP/DP setups).
"""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.moe import SharedExpertMLPKernel
from tileops.ops.moe.abc import FusedMoEExpertsModular, FusedMoEPrepareAndFinalize
from tileops.ops.moe.fused_moe import FusedMoe

__all__ = ["FusedMoeSharedExpertFwdOp"]


class FusedMoeSharedExpertFwdOp(FusedMoe):
    """FusedMoE with shared expert support, optionally TP-aware.

    Extends FusedMoe to compute both shared and routed expert outputs. Passing the
    shared expert's weights enables it; the shared expert is computed via
    SharedExpertMLPKernel (TileLang), which applies ``silu_and_mul``.

    TP support (shared expert only):
        When tp_size > 1, the op shards the shared expert weights internally:
          - shared_w_gate_up [2*F_s, H] is split along dim=0 (ColumnParallel)
          - shared_w_down    [H, F_s]   is split along dim=1 (RowParallel)
        The returned shared_out is a partial sum; the caller must all-reduce
        across TP ranks. The routed expert path is not affected.

    Returns:
        (shared_output, routed_output): tuple of [T, H] tensors.
            shared_output is None when the shared weights are not passed.
            shared_output is a partial sum when tp_size > 1.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "shared_expert_mlp": SharedExpertMLPKernel
    }

    def __init__(
        self,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        routed_scaling_factor: float = 1.0,
        tp_size: int = 1,
        tp_rank: int = 0,
        *,
        activation: str = "silu_and_mul",
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None,
        experts: Optional[FusedMoEExpertsModular] = None,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            top_k: K -- experts selected per token.
            scoring_func: "softmax" (Qwen3) or "sigmoid" (Kimi K2 / DeepSeek-V3).
            renormalize: Renormalize top-k weights to sum to 1.
            routed_scaling_factor: Multiplier on the routed expert output.
            tp_size: Tensor parallel world size. Default 1 (no TP).
            tp_rank: This rank's index in the TP group. Default 0.
            activation: Gated activation of the routed experts; the shared expert
                supports ``silu_and_mul`` only.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Kernel overrides, for this op's kernel and its sub-ops.
            tune: Whether the kernels tune themselves when built.
            prepare_finalize: Override the PrepareAndFinalize implementation.
            experts: Override the Experts implementation.
        """
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.activation = activation
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._build_pipeline(prepare_finalize, experts)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per token count, width, shard size and dtype."""
        tokens, hidden, shard_ffn, dtype = call
        return call, lambda: self.kernel_map[role](
            num_tokens=tokens, hidden_size=hidden, ffn_size=shard_ffn, dtype=dtype, tune=self.tune
        )

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
            shared_w_gate_up: [2*F_s, H] shared expert gate+up weights (full); passing
                it enables the shared expert. When tp_size > 1, sharded along dim=0.
            shared_w_down: [H, F_s] shared expert down weight (full), passed with
                ``shared_w_gate_up``. When tp_size > 1, sharded along dim=1.

        Returns:
            (shared_output, routed_output): tuple of [T, H] tensors.
                shared_output is None when the shared weights are not passed, and a
                partial sum when tp_size > 1; the caller all-reduces across TP ranks.
        """
        shared_out = None
        if shared_w_gate_up is not None:
            ffn = shared_w_down.shape[1]
            shard = ffn // self.tp_size
            if self.tp_size > 1:
                # ColumnParallel: rank r computes neurons [r*s, (r+1)*s), so it needs
                # gate[r*s:(r+1)*s] and up[r*s:(r+1)*s] concatenated into [2*s, H].
                lo, hi = self.tp_rank * shard, (self.tp_rank + 1) * shard
                gate_up = torch.cat(
                    [shared_w_gate_up[lo:hi], shared_w_gate_up[ffn + lo : ffn + hi]], dim=0
                ).contiguous()
                down = shared_w_down.narrow(1, lo, shard).contiguous()
            else:
                gate_up, down = shared_w_gate_up, shared_w_down
            tensors = (hidden_states, gate_up, down)
            kernel = self.kernel_for(
                "shared_expert_mlp",
                tensors,
                (hidden_states.shape[0], hidden_states.shape[1], shard, hidden_states.dtype),
            )
            shared_out = kernel(*tensors)
        routed_out = self._routed(hidden_states, gating_output, w_gate_up, w_down, correction_bias)
        return shared_out, routed_out
