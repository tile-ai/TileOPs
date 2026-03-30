"""SharedFusedMoE — FusedMoE with shared expert support.

Combines routed experts (via FusedMoe) with shared experts (external MLP).
Follows vLLM's SharedFusedMoE design.

Usage:
    # Create shared expert MLP (user-provided)
    shared_mlp = nn.Sequential(
        nn.Linear(H, F),
        nn.SiLU(),
        nn.Linear(F, H),
    )

    # Create SharedFusedMoE
    op = SharedFusedMoE(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F,
        shared_experts_fn=shared_mlp,
        ...
    )

    # Forward returns (shared_output, routed_output)
    shared_out, routed_out = op(hidden, gating, w_gate_up, w_down)

    # Model layer combines them
    final_out = routed_out * routed_scaling_factor + shared_out
"""

from typing import Callable, Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.ops.moe.fused_moe import FusedMoe

__all__ = ["SharedFusedMoE"]


class SharedFusedMoE(FusedMoe):
    """FusedMoE with shared expert support.

    Extends FusedMoe to compute both shared and routed expert outputs.
    Follows vLLM's SharedFusedMoE design.

    Args:
        shared_experts_fn: Callable that computes shared expert output.
            Signature: fn(hidden_states: Tensor[T, H]) -> Tensor[T, H]
            If None, no shared expert computation (returns None for shared_out).
        Other args: same as FusedMoe.

    Returns:
        (shared_output, routed_output): tuple of [T, H] tensors.
            shared_output is None if shared_experts_fn is None.
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
        shared_experts_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
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
        self.shared_experts_fn = shared_experts_fn

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
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Run shared + routed MoE FFN.

        Args:
            Same as FusedMoe.forward.

        Returns:
            (shared_output, routed_output): tuple of [T, H] tensors.
                shared_output is None if shared_experts_fn is None.
        """
        # Compute shared expert output
        shared_out = (
            self.shared_experts_fn(hidden_states)
            if self.shared_experts_fn is not None
            else None
        )

        # Compute routed expert output
        routed_out = super().forward(
            hidden_states, gating_output, w_gate_up, w_down, correction_bias
        )

        return shared_out, routed_out
