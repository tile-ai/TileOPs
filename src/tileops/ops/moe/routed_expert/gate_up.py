"""The gate/up stage of the MoE expert pipeline, activation included."""

from typing import Dict, Optional

import torch

from tileops.kernels.grouped_gemm_call import GroupedGemmCall
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import (
    MoeGroupedGemmPersistent3WGFusedActKernel,
    MoeGroupedGemmSeparateActKernel,
)

from ...op_base import Op

__all__ = ["MoeGateUpFwdOp"]

#: The implementations of this role. The fused one states the shapes where carrying
#: the activation in its epilogue is worth the narrower schedule; the two-launch one
#: is the general implementation behind it.
_GATE_UP_KEYS = ("moe_grouped_gemm_fused_act_kernel", "moe_grouped_gemm_act_kernel")


class MoeGateUpFwdOp(Op):
    """Gate/up GEMM and its gated activation.

    Takes the tight permuted rows and the stacked gate||up weights, and returns
    activated rows of width ``ffn``.

    Args:
        numel: T * top_k tight row count.
        num_experts: Number of local experts E.
        ffn: FFN width; ``b`` holds 2*ffn rows (gate||up).
        k: Hidden size K.
        activation: 'silu_and_mul' or 'gelu_and_mul'.
        kernel_map: Optional kernel override dict.
        tune: Whether to autotune.

    Example:
        >>> op = MoeGateUpFwdOp(numel=4096, num_experts=128, ffn=2048, k=7168)
        >>> act = op(a, b, true_sizes, true_offsets)  # [4096, 2048]
    """

    def __init__(
        self,
        numel: int,
        num_experts: int,
        ffn: int,
        k: int,
        activation: str = "silu_and_mul",
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.numel = numel
        self.num_experts = num_experts
        self.ffn = ffn
        self.k = k
        self.activation = activation
        self.tune = tune

        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, inputs: tuple, dtype: torch.dtype) -> Kernel:
        call = GroupedGemmCall(
            numel=self.numel, num_experts=self.num_experts,
            n=self.ffn, k=self.k, dtype=dtype,
        )
        name = self.select_kernel_key(_GATE_UP_KEYS, call)
        return self.get_or_build_kernel(
            name, inputs,
            key=(name, dtype),
            build=lambda: self.kernel_map[name](
                self.numel, self.num_experts, self.ffn, self.k,
                dtype=dtype, activation=self.activation, tune=self.tune,
            ),
        )

    def _infer_output_shapes(
        self,
        a_shape: tuple,
        b_shape: tuple,
        true_sizes_shape: tuple,
        true_offsets_shape: tuple,
    ) -> Dict[str, tuple]:
        # b is [num_experts, 2 * ffn, K]; the gated activation halves that width.
        return {"c": (a_shape[0], b_shape[1] // 2)}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "moe_grouped_gemm_fused_act_kernel": MoeGroupedGemmPersistent3WGFusedActKernel,
            "moe_grouped_gemm_act_kernel": MoeGroupedGemmSeparateActKernel,
        }

    def forward(
        self,
        a: torch.Tensor,             # [numel, K]
        b: torch.Tensor,             # [num_experts, 2*ffn, K]
        true_sizes: torch.Tensor,    # [E] int32
        true_offsets: torch.Tensor,  # [E] int32
    ) -> torch.Tensor:
        """Run the gate/up GEMM and apply the gated activation.

        Args:
            a: [numel, K] tight permuted activations.
            b: [num_experts, 2*ffn, K] gate||up expert weights.
            true_sizes: [E] int32 token count per expert.
            true_offsets: [E] int32 tight start offset per expert in a.

        Returns:
            [numel, ffn] activated output.
        """
        self._validate_dtypes(a, b, true_sizes, true_offsets)
        self.dtype = a.dtype
        inputs = (a, b, true_sizes, true_offsets)
        return self._get_kernel(inputs, a.dtype)(*inputs)
