"""MoE grouped GEMM op (no-pad variant): NT GEMM with precomputed tile scheduling."""

from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.kernels.grouped_gemm import GroupedGemmCall, GroupedGemmPersistent3WGKernel
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadKernel

from ...compile_boundary import get_instance
from ...op_base import Op

__all__ = ["MoeGroupedGemmNopadFwdOp"]

#: The implementations of this role; each states its own region.
_GEMM_KEYS = ("moe_grouped_gemm_kernel", "moe_grouped_gemm_persistent_kernel")


class MoeGroupedGemmNopadFwdOp(Op):
    """NT grouped GEMM for MoE without block_m-aligned padding.

    Uses a GPU tile scheduler to map each CTA to its (expert, row_offset) in O(1),
    eliminating the O(E) per-CTA expert scan in standard grouped GEMM.

    Accepts tight A[T*K, K] inputs (no padding between experts) from
    MoePermuteNoPadOp, producing tight C[T*K, N] outputs.

    Args:
        numel: T * top_k, total (token, expert) pairs = tight row count.
        num_experts: Total number of experts E.
        n: Output feature dimension N (e.g. 2*ffn_size or hidden_size).
        k: Input feature dimension K (hidden_size or ffn_size).
        kernel_map: Optional kernel override dict.
        tune: Whether to autotune.

    Example:
        >>> op = MoeGroupedGemmNopadFwdOp(numel=16384, num_experts=256, n=4096, k=2048,
        ...)
        >>> C = op(A, B, true_sizes, true_offsets)  # [numel, N]
    """

    #: The operator this op registers; a test asserts the graph holds nothing else.
    compile_op_names: ClassVar[Tuple[str, ...]] = ("top::moe_grouped_gemm_nopad_fwd",)

    def __init__(
        self,
        numel: int,
        num_experts: int,
        n: int,
        k: int,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.numel = numel
        self.num_experts = num_experts
        self.n = n
        self.k = k
        self.tune = tune

        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, inputs: tuple, dtype: torch.dtype) -> Kernel:
        call = GroupedGemmCall(
            numel=self.numel, num_experts=self.num_experts,
            n=self.n, k=self.k, dtype=dtype,
        )
        name = self.select_kernel_key(_GEMM_KEYS, call)
        return self.get_or_build_kernel(
            name, inputs,
            key=(name, dtype),
            build=lambda: self.kernel_map[name](
                self.numel, self.num_experts, self.n, self.k,
                dtype=dtype, tune=self.tune,
            ),
        )

    def _infer_output_shapes(
        self,
        a_shape: tuple,
        b_shape: tuple,
        true_sizes_shape: tuple,
        true_offsets_shape: tuple,
    ) -> Dict[str, tuple]:
        # b is [num_experts, N, K]; the tight output keeps a's row count.
        return {"c": (a_shape[0], b_shape[1])}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "moe_grouped_gemm_kernel": MoeGroupedGemmNopadKernel,
            "moe_grouped_gemm_persistent_kernel": GroupedGemmPersistent3WGKernel,
        }

    def forward(
        self,
        a: torch.Tensor,           # [numel, K]
        b: torch.Tensor,           # [num_experts, N, K]
        true_sizes: torch.Tensor,  # [E] int32
        true_offsets: torch.Tensor,  # [E] int32
    ) -> torch.Tensor:
        """Run tile-scheduled NT GEMM.

        Args:
            a: [numel, K] tight permuted activations.
            b: [num_experts, N, K] expert weights (NT: B^T applied).
            true_sizes: [E] int32 true token count per expert.
            true_offsets: [E] int32 tight start offset per expert in a.

        Returns:
            C: [numel, N] GEMM output.
        """
        return _moe_grouped_gemm_nopad_fwd(
            a, b, true_sizes, true_offsets, self._instance_key)

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


@torch.library.custom_op("top::moe_grouped_gemm_nopad_fwd", mutates_args=())
def _moe_grouped_gemm_nopad_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    true_sizes: torch.Tensor,
    true_offsets: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(a, b, true_sizes, true_offsets)


@_moe_grouped_gemm_nopad_fwd.register_fake
def _moe_grouped_gemm_nopad_fwd_fake(
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
