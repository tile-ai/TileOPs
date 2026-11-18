"""Manifold hyper-connection operators (L2 Op layer).

Provides:
  - HyperConnectionPostFwdOp: output[j] = post_layer_mix[j] * x + sum_i comb_res_mix[i, j] *
    residual[i], per token
  - HyperConnectionPreFwdOp: the pre step of a block over ``n`` residual streams
"""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.sequence_modeling.hyper_connection import (
    HyperConnectionPostCall,
    HyperConnectionPostFwdInterface,
    HyperConnectionPostKernel,
    HyperConnectionPreCall,
    HyperConnectionPreFwdInterface,
    HyperConnectionPreKernel,
)
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["HyperConnectionPostFwdOp", "HyperConnectionPreFwdOp"]


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


class HyperConnectionPostFwdOp(Op):
    """Mix a block's output back into the ``n`` residual streams of a manifold hyper-connection.

    Per token, output stream ``j`` is
    $post\\_layer\\_mix_j \\cdot x + \\sum_i comb\\_res\\_mix_{i, j} \\cdot residual_i$, summed in
    float32 and rounded once to ``bfloat16``. This is DeepSeek-V4-Flash ``inference/model.py``
    ``Block.hc_post`` after each attention and MoE block, with ``post_layer_mix`` and
    ``comb_res_mix`` the ``post`` and ``comb`` the pre step produced; ``comb_res_mix`` is read
    by column, ``comb_res_mix[i, j]`` weighting input stream ``i`` in output stream ``j``.

    The mixes are ``float32`` or ``bfloat16``; GLM-5.3-Flash rounds them to ``bfloat16``
    before mixing, and passing them so gives its mix computed in float32. Elements are
    summed in another order than a reference summing the streams first, so an output can
    differ from it by one ``bfloat16`` rounding.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "hyper_connection_post": HyperConnectionPostKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "hyper_connection_post": HyperConnectionPostFwdInterface
    }

    def __init__(self, *, target: Target = None):
        """Build the op. Shapes are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        """Mix ``x`` into the residual streams.

        Args:
            x: The block output, $[*S \\times C]$ ``bfloat16``.
            residual: The ``n`` streams the block read, $[*S \\times n \\times C]$
                ``bfloat16``.
            post_layer_mix: Each output stream's weight of ``x``, $[*S \\times n \\times 1]$,
                ``float32`` or ``bfloat16``.
            comb_res_mix: Input stream ``i``'s weight in output stream ``j`` at ``[i, j]``,
                $[*S \\times n \\times n]$, in ``post_layer_mix``'s dtype.

        Returns:
            The ``n`` new streams, $[*S \\times n \\times C]$ ``bfloat16``.

        Example:
            ```python linenums="1"
            import torch
            from tileops.sequence_modeling import HyperConnectionPostFwdOp

            x = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
            residual = torch.randn(4096, 4, 4096, dtype=torch.bfloat16, device="cuda")
            post = torch.rand(4096, 4, 1, device="cuda") * 2
            comb = torch.softmax(torch.randn(4096, 4, 4, device="cuda"), dim=-1)
            output = HyperConnectionPostFwdOp()(x, residual, post, comb)
            ```
        """
        x = x.contiguous()
        residual = residual.contiguous()
        post_layer_mix = post_layer_mix.contiguous()
        comb_res_mix = comb_res_mix.contiguous()
        kernel = self.kernel_for(
            "hyper_connection_post",
            lambda: HyperConnectionPostCall(
                n=residual.shape[-2],
                c=residual.shape[-1],
                mix_dtype=post_layer_mix.dtype,
                device=residual.device,
            ),
        )
        return kernel(x, residual, post_layer_mix, comb_res_mix)
