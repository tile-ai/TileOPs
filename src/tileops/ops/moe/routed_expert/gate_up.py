"""The gate/up stage of the MoE expert pipeline, activation included."""

from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.kernels.grouped_gemm import GroupedGemmCall, GroupedGemmPersistent3WGKernel
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import (
    MoeGroupedGemmNopadKernel,
    MoeGroupedGemmPersistent3WGFusedActKernel,
    MoeGroupedGemmSeparateActKernel,
)

from ...compile_boundary import get_instance
from ...op_base import Op

__all__ = ["MoeGateUpFwdOp"]

#: The implementations of this role; each states its own region.
_GATE_UP_KEYS = ("moe_grouped_gemm_fused_act_kernel", "moe_grouped_gemm_act_kernel")

#: The grouped GEMM the separate-activation implementation composes with.
_GEMM_KEYS = ("moe_grouped_gemm_kernel", "moe_grouped_gemm_persistent_kernel")


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

    #: The operator this op registers; a test asserts the graph holds nothing else.
    compile_op_names: ClassVar[Tuple[str, ...]] = ("top::moe_gate_up_fwd",)

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

    def _call(self, n: int, dtype: torch.dtype, activation: str) -> GroupedGemmCall:
        return GroupedGemmCall(
            numel=self.numel, num_experts=self.num_experts,
            n=n, k=self.k, dtype=dtype, activation=activation,
        )

    def _get_kernel(self, inputs: tuple, dtype: torch.dtype) -> Kernel:
        name = self.select_kernel_key(
            _GATE_UP_KEYS, self._call(self.ffn, dtype, self.activation))
        gemm_key, extra = None, {}
        if name == "moe_grouped_gemm_act_kernel":
            # The GEMM it composes with is a role of its own, selected here because
            # this is where the map holding those candidates lives. That role applies
            # no activation, so its call describes none.
            gemm_key = self.select_kernel_key(
                _GEMM_KEYS, self._call(2 * self.ffn, dtype, ""))
            extra["gemm_cls"] = self.kernel_map[gemm_key]
        return self.get_or_build_kernel(
            name, inputs,
            key=(name, gemm_key, dtype),
            build=lambda: self.kernel_map[name](
                self.numel, self.num_experts, self.ffn, self.k,
                dtype=dtype, activation=self.activation, tune=self.tune, **extra,
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
            "moe_grouped_gemm_kernel": MoeGroupedGemmNopadKernel,
            "moe_grouped_gemm_persistent_kernel": GroupedGemmPersistent3WGKernel,
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
        return _moe_gate_up_fwd(a, b, true_sizes, true_offsets, self._instance_key)

    def _eager_forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        true_sizes: torch.Tensor,
        true_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Validate, normalize, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo
        cannot follow.
        """
        self._validate_dtypes(a, b, true_sizes, true_offsets)
        for name, t in (("b", b), ("true_sizes", true_sizes), ("true_offsets", true_offsets)):
            if t.device != a.device:
                raise ValueError(
                    f"{name} must be on {a.device}, got {t.device}")
        self.dtype = a.dtype
        # The op hands over what the manifest declares; how a kernel wants it laid
        # out is its own business.
        inputs = tuple(t.contiguous() for t in (a, b, true_sizes, true_offsets))
        return self._get_kernel(inputs, a.dtype)(*inputs)


@torch.library.custom_op("top::moe_gate_up_fwd", mutates_args=())
def _moe_gate_up_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    true_sizes: torch.Tensor,
    true_offsets: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(a, b, true_sizes, true_offsets)


@_moe_gate_up_fwd.register_fake
def _moe_gate_up_fwd_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    true_sizes: torch.Tensor,
    true_offsets: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(
        tuple(a.shape), tuple(b.shape), tuple(true_sizes.shape),
        tuple(true_offsets.shape))
    # ``new_empty``, not ``empty_like``: ``_eager_forward`` normalizes contiguity, so a
    # non-contiguous public input's strides must not survive into the fake. Dtype is the
    # manifest's ``same_as(a)``.
    return a.new_empty(shapes["c"])
