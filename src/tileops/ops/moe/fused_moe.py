"""Routed Mixture-of-Experts (MoE) FFN operators.

``FusedMoeFwdOp`` is routing + expert FFN. Passing ``correction_bias`` adds the
per-expert bias during top-k selection (Kimi K2 style); withholding it selects
straight from the gating scores (Qwen3 / DeepSeek-V3 style).

The shared core (`FusedMoe`) wires `FusedTopKOp` (routing),
`FusedMoEPrepareAndFinalize` (quantization / EP dispatch), and an
`FusedMoEExpertsModular` implementation (permute + GEMM + unpermute). Shared
expert handling belongs to `SharedFusedMoE`.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.ops.moe.abc import (
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalize,
)
from tileops.ops.moe.fused_topk import FusedTopKOp
from tileops.ops.moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEP
from tileops.ops.moe.routed_expert import FusedMoEExpertsNopadPersistent3WGFwdOp
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["FusedMoe", "FusedMoeFwdOp"]


class FusedMoe(Op):
    """Shared composite implementation for routed MoE FFN ops.

    The concrete manifest identity (`FusedMoeFwdOp`) subclasses this; the
    routing-and-expert pipeline below is shared with `SharedFusedMoE`.

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
        routed_scaling_factor: float = 1.0,
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None,
        experts: Optional[FusedMoEExpertsModular] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        activation: str = "silu_and_mul",
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            num_tokens: T -- number of input tokens.
            num_experts: E -- total number of experts (global count).
            top_k: K -- experts selected per token.
            hidden_size: H -- model hidden dimension.
            ffn_size: F -- per-expert intermediate dimension.
            scoring_func: "softmax" (Qwen3) or "sigmoid" (Kimi K2 / DeepSeek-V3).
            renormalize: Renormalize top-k weights to sum to 1.
            routed_scaling_factor: Multiplier on expert output (Kimi K2: 2.827).
            prepare_finalize: Override the PrepareAndFinalize implementation.
            experts: Override the Experts implementation.
            kernel_map: Override the dispatched kernel map.
        """
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor

        self.dispatch_kernel(kernel_map)

        self._fused_topk = FusedTopKOp(
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
            kernel_map=kernel_map,
        )

        self._prepare: FusedMoEPrepareAndFinalize = (
            prepare_finalize if prepare_finalize is not None else MoEPrepareAndFinalizeNoDPEP()
        )

        if prepare_finalize is not None and experts is None:
            raise ValueError(
                "prepare_finalize may change the dispatched token count (T'); "
                "you must also supply a matching experts= instance sized for T'."
            )

        if experts is not None:
            # All in-tree FusedMoEExperts*FwdOp set self.activation in __init__.
            # We require it to be present rather than falling back silently —
            # a missing attribute on a third-party experts implementation would
            # otherwise let a non-matching `activation` argument be silently
            # accepted, producing a wrong-activation pipeline.
            if not hasattr(experts, "activation"):
                raise ValueError(
                    f"injected experts instance ({type(experts).__name__}) "
                    "is missing the required `.activation` attribute. "
                    "Set it in __init__ to the activation string this experts "
                    "implementation uses (e.g. 'silu_and_mul')."
                )
            experts_activation = experts.activation
            # Reject only conflicting non-default values. Passing the default
            # ("silu_and_mul") alongside experts= is silently accepted because
            # it cannot be distinguished from the bare experts= call. Passing
            # an explicit value that matches the injected experts' activation
            # is also accepted.
            if activation != "silu_and_mul" and activation != experts_activation:
                raise ValueError(
                    "activation conflicts with the injected experts instance: "
                    f"got activation={activation!r}, "
                    f"experts.activation={experts_activation!r} "
                    f"(experts={type(experts).__name__}). "
                    "Either omit activation or pass the same value."
                )
            self.activation = experts_activation
            self._experts: FusedMoEExpertsModular = experts
        else:
            self.activation = activation
            self._experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                routed_scaling_factor=routed_scaling_factor,
                kernel_map=kernel_map,
                activation=activation,
            )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {}

    def _infer_output_shapes(
        self,
        hidden_states_shape: tuple[int, ...],
        gating_output_shape: tuple[int, ...],
        w_gate_up_shape: tuple[int, ...],
        w_down_shape: tuple[int, ...],
        correction_bias_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: routing returns one row per token, of the input width."""
        return {"output": tuple(hidden_states_shape)}

    def forward(
        self,
        hidden_states: torch.Tensor,  # [T, H]
        gating_output: torch.Tensor,  # [T, E]
        w_gate_up: torch.Tensor,  # [E, 2*F, H]
        w_down: torch.Tensor,  # [E, H, F]
        correction_bias: Optional[torch.Tensor] = None,  # [E] float32
    ) -> torch.Tensor:  # [T, H]
        """Run the op on ``hidden_states``, ``gating_output``, ``w_gate_up``, ``w_down`` and ``correction_bias``."""
        topk_weights, topk_ids = self._fused_topk(gating_output, correction_bias)
        # The roofline counts the bias bytes of the call that ran, so this is
        # set once routing succeeded. Keep the shape, not the tensor: the op
        # need not hold the caller's memory.
        self.correction_bias_shape = (
            None if correction_bias is None else tuple(correction_bias.shape)
        )
        # The roofline prices hidden states and weights at the call's dtype.
        self.dtype = hidden_states.dtype

        r = self._prepare.prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            self.num_experts,
        )

        T_prime = r.hidden_q.shape[0]
        ws1_shape, ws2_shape = self._experts.workspace_shapes(
            T_prime,
            self.ffn_size,
            self.hidden_size,
            self.top_k,
            self.num_experts,
        )
        ws1 = hidden_states.new_empty(ws1_shape)
        ws2 = hidden_states.new_empty(ws2_shape)

        output = hidden_states.new_empty(hidden_states.shape)
        expert_out_shape = self._experts.output_shape(T_prime, self.hidden_size)
        expert_out = (
            output
            if expert_out_shape == tuple(hidden_states.shape)
            else hidden_states.new_empty(expert_out_shape)
        )
        self._experts.forward(
            expert_out,
            r.hidden_q,
            w_gate_up,
            w_down,
            r.topk_weights,
            r.topk_ids,
            workspace1=ws1,
            workspace2=ws2,
        )

        self._prepare.finalize(
            output,
            expert_out,
            r.topk_weights,
            r.topk_ids,
            self._experts.make_weighted_reduce(),
        )
        return output

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class FusedMoeFwdOp(FusedMoe):
    """Routed MoE FFN.

    Covers Qwen3 (softmax) and DeepSeek-V3 (sigmoid) style configurations where
    top-k comes straight from the gating scores, and Kimi K2 style ones where a
    per-expert ``correction_bias`` is passed: top-k is then selected from
    ``sigmoid(score) + correction_bias`` while the final weights use the
    original (unbiased) scores, renormalized.
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
        routed_scaling_factor: float = 1.0,
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None,
        experts: Optional[FusedMoEExpertsModular] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        activation: str = "silu_and_mul",
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            num_tokens: Manifest ``params.num_tokens``, ``int``.
            num_experts: Manifest ``params.num_experts``, ``int``.
            top_k: Manifest ``params.top_k``, ``int``.
            hidden_size: Manifest ``params.hidden_size``, ``int``.
            ffn_size: Manifest ``params.ffn_size``, ``int``.
            scoring_func: Manifest ``params.scoring_func``, ``str``, default ``'softmax'``.
            renormalize: Manifest ``params.renormalize``, ``bool``, default ``False``.
            routed_scaling_factor: Manifest ``params.routed_scaling_factor``, ``float``, default ``1.0``.
            kernel_map: Optional kernel override dict.
            activation: Manifest ``params.activation``, ``str``, default ``'silu_and_mul'``.
        """
        super().__init__(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            prepare_finalize=prepare_finalize,
            experts=experts,
            kernel_map=kernel_map,
            activation=activation,
        )
