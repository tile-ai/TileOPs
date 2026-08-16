"""Gate/up grouped GEMM with the activation as a separate launch."""

import torch

from tileops.kernels.elementwise import GeluAndMulFwdKernel, SiluAndMulFwdKernel
from tileops.kernels.grouped_gemm import GroupedGemmPersistent3WGKernel
from tileops.kernels.grouped_gemm_call import GroupedGemmCall
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadKernel

__all__ = ["MoeGroupedGemmSeparateActKernel"]

_ACTIVATIONS = {
    "silu_and_mul": SiluAndMulFwdKernel,
    "gelu_and_mul": GeluAndMulFwdKernel,
}


class MoeGroupedGemmSeparateActKernel(Kernel):
    """Gate/up GEMM of width ``2 * ffn``, then the gated activation.

    Serves the same call as the fused-epilogue kernel and produces the same
    ``[numel, ffn]`` output, in two launches with the wide intermediate in global
    memory. It is the general implementation of that role: the fused one states the
    shapes where its epilogue is worth the narrower schedule, and this one runs
    everywhere else.

    The grouped GEMM inside is the persistent kernel wherever its tiling divides the
    shape, and the tile-scheduler kernel otherwise — the same regions those two
    classes state for the down GEMM.
    """

    general = True
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(self, numel, num_experts, N, K, dtype=torch.bfloat16,
                 activation="silu_and_mul", config=None, tune=False):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}")
        self.numel = numel
        self.num_experts = num_experts
        self.N = N            # ffn (output width), NOT 2*ffn
        self.K = K
        self.dtype = dtype
        self.activation = activation
        self.init_config(config, tune)

        gemm_call = GroupedGemmCall(
            numel=numel, num_experts=num_experts, n=2 * N, k=K, dtype=dtype)
        gemm_cls = (GroupedGemmPersistent3WGKernel
                    if GroupedGemmPersistent3WGKernel.refusal(gemm_call) is None
                    else MoeGroupedGemmNopadKernel)
        self._gemm = gemm_cls(numel, num_experts, 2 * N, K, dtype=dtype, tune=tune)
        self._act = _ACTIVATIONS[activation](numel, N, dtype, tune=tune)

    @property
    def default_config(self) -> dict:
        # Both launches carry their own schedule.
        return {}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, A, B, true_sizes, true_offsets):
        """Run the wide grouped GEMM and activate it.

        Args:
            A: [numel, K] tight permuted activations.
            B: [num_experts, 2*ffn, K] gate||up expert weights.
            true_sizes: [E] int32 token count per expert.
            true_offsets: [E] int32 tight start offset per expert in A.

        Returns:
            [numel, ffn] activated output.
        """
        return self._act(self._gemm(A, B, true_sizes, true_offsets))
