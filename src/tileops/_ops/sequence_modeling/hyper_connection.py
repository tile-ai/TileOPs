"""Manifold hyper-connection operators (L2 Op layer).

Provides:
  - HyperConnectionPreFwdOp: the pre step of a block over ``n`` residual streams
"""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.sequence_modeling.hyper_connection import (
    HyperConnectionPreCall,
    HyperConnectionPreFwdInterface,
    HyperConnectionPreKernel,
)
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["HyperConnectionPreFwdOp"]


class HyperConnectionPreFwdOp(Op):
    """The pre step of a manifold hyper-connection, as DeepSeek-V4-Flash ``Block.hc_pre`` runs it.

    Per token, ``x`` is the ``n`` streams of ``residual`` flattened to ``n * C`` values in
    float32, and

    $$
    m = (x\\, fn^T) \\cdot \\mathrm{rsqrt}(\\mathrm{mean}(x^2) + \\epsilon_{rms})
    $$

    splits into ``n`` pre, ``n`` post and ``n * n`` comb logits, each times its ``hc_scale``
    entry plus ``hc_base``. ``pre = sigmoid + hc_pre_eps`` and ``post_mix = sigmoid *
    hc_post_mult_value``. ``comb_mix`` is a row softmax plus ``hc_sinkhorn_eps``, then a
    column normalization and ``sinkhorn_repeat - 1`` row-then-column normalizations, each
    dividing by the sum plus ``hc_sinkhorn_eps``. ``layer_input = sum_i pre_i * residual_i``,
    rounded once to bfloat16.

    The projection is as accurate as a float32 matrix product: ``fn`` takes part as three
    bfloat16 terms whose sum is its float32 value. A residual whose stream axis has stride 0,
    one stream broadcast to all ``n`` as the first block's is, is read as that one stream.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "hyper_connection_pre": HyperConnectionPreKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "hyper_connection_pre": HyperConnectionPreFwdInterface
    }

    def __init__(
        self,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes are taken from each call.

        Args:
            rms_eps: Added to the mean square before the root.
            hc_pre_eps: Added to each pre weight.
            hc_sinkhorn_eps: Added to the softmax and to each normalization's divisor.
            hc_post_mult_value: Multiplies each post weight.
            sinkhorn_repeat: Sinkhorn iterations, the first one's row step being the softmax.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.rms_eps = rms_eps
        self.hc_pre_eps = hc_pre_eps
        self.hc_sinkhorn_eps = hc_sinkhorn_eps
        self.hc_post_mult_value = hc_post_mult_value
        self.sinkhorn_repeat = sinkhorn_repeat
        super().__init__(target=target)

    def forward(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix the streams of each token down to its block input.

        Args:
            residual: ``(*S, n, C)`` bfloat16 streams.
            fn: ``(n * n + 2 * n, n * C)`` float32 projection.
            hc_scale: ``(3,)`` float32 scales of the pre, post and comb logits.
            hc_base: ``(n * n + 2 * n,)`` float32 biases of the logits.

        Returns:
            ``post_mix`` ``(*S, n, 1)`` float32, ``comb_mix`` ``(*S, n, n)`` float32 and
            ``layer_input`` ``(*S, C)`` bfloat16.
        """
        fn = fn.contiguous()
        hc_scale = hc_scale.contiguous()
        hc_base = hc_base.contiguous()
        kernel = self.kernel_for(
            "hyper_connection_pre",
            lambda: HyperConnectionPreCall(
                device=residual.device,
                n=residual.shape[-2],
                c=residual.shape[-1],
                broadcast=residual.shape[-2] > 1 and residual.stride(-2) == 0,
                rms_eps=self.rms_eps,
                hc_pre_eps=self.hc_pre_eps,
                hc_sinkhorn_eps=self.hc_sinkhorn_eps,
                hc_post_mult_value=self.hc_post_mult_value,
                sinkhorn_repeat=self.sinkhorn_repeat,
            ),
        )
        return kernel(residual, fn, hc_scale, hc_base)
