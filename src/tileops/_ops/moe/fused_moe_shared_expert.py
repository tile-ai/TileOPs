"""FusedMoESharedExpertFwdOp: routed experts (via FusedMoE) and the shared expert
(SharedExpertMLPFwdOp)."""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._ops.moe.fused_moe import FusedMoE
from tileops._ops.moe.routed_expert import FusedMoEExpertsFwdOp
from tileops._ops.moe.shared_expert_mlp import SharedExpertMLPFwdOp
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["FusedMoESharedExpertFwdOp"]


class FusedMoESharedExpertFwdOp(FusedMoE):
    """Routed MoE FFN plus an optional shared expert, with tensor parallelism (TP) for the
    shared expert.

    The routed half routes and computes as ``FusedMoEFwdOp`` does;
    ``routed_scaling_factor`` scales the routed output only, once. Passing the shared
    expert's weights enables the shared expert, a ``silu_and_mul`` MLP.

    When ``tp_size > 1`` the op takes the full shared weights and uses this rank's shard:
    ``shared_w_gate_up`` $[2F_s \\times H]$ split along dim 0 (column parallel, gate and
    up halves sharded alike) and ``shared_w_down`` $[H \\times F_s]$ split along dim 1
    (row parallel). The shared output is then a partial sum that the caller all-reduces
    across TP ranks. The routed experts are not sharded.
    """

    delegate_types: ClassVar[Mapping[str, type[Op]]] = {
        **FusedMoE.delegate_types,
        "shared_expert": SharedExpertMLPFwdOp,
    }

    def __init__(
        self,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        routed_scaling_factor: float = 1.0,
        num_expert_group: int = 1,
        topk_group: int = 1,
        tp_size: int = 1,
        tp_rank: int = 0,
        *,
        activation: str = "silu_and_mul",
        target: Target = None,
        experts: FusedMoEExpertsFwdOp | None = None,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            top_k: K -- experts selected per token.
            scoring_func: "softmax" (Qwen3) or "sigmoid" (Kimi K2 / DeepSeek-V3).
            renormalize: Renormalize top-k weights to sum to 1.
            routed_scaling_factor: Multiplier on the routed expert output.
            num_expert_group: Groups of consecutive experts the routing is restricted
                by; must divide E. 1 selects over all experts. DeepSeek-V3: 8.
            topk_group: Groups kept per token, holding at least ``top_k`` experts.
                DeepSeek-V3: 4.
            tp_size: Tensor parallel world size. Default 1 (no TP).
            tp_rank: This rank's index in the TP group. Default 0.
            activation: Gated activation of the routed experts; the shared expert
                supports ``silu_and_mul`` only.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            experts: A configured ``FusedMoEExpertsFwdOp`` to run the experts; ``None``
                builds one from ``activation`` and ``routed_scaling_factor``.
        """
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        super().__init__(
            top_k,
            scoring_func,
            renormalize,
            routed_scaling_factor,
            num_expert_group,
            topk_group,
            activation=activation,
            target=target,
            experts=experts,
        )
        self._shared_expert = self.delegate_for("shared_expert", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        shared_w_gate_up: torch.Tensor | None = None,
        shared_w_down: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run shared + routed MoE FFN.

        Args:
            hidden_states: [T, H] input hidden states.
            gating_output: [T, E] gating logits.
            w_gate_up: [E, 2F, H] routed expert gate+up weights.
            w_down: [E, H, F] routed expert down weights.
            correction_bias: Optional [E] bias for Kimi K2 / DeepSeek-V3 routing.
            shared_w_gate_up: [2*F_s, H] shared expert gate+up weights (full, unsharded);
                passing it enables the shared expert.
            shared_w_down: [H, F_s] shared expert down weight (full, unsharded), passed
                with ``shared_w_gate_up``.

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
                )
                down = shared_w_down.narrow(1, lo, shard).contiguous()
            else:
                gate_up, down = shared_w_gate_up, shared_w_down
            shared_out = self._shared_expert(hidden_states, gate_up, down)
        routed_out = self._routed(hidden_states, gating_output, w_gate_up, w_down, correction_bias)
        return shared_out, routed_out
