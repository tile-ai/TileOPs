"""The min-p logit filter op."""

from typing import ClassVar, Mapping

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.sampling import MinPMaskFwdInterface, MinPMaskFwdKernel, SamplingCall
from tileops.ops.op_base import Op

__all__ = ["MinPMaskFwdOp"]


class MinPMaskFwdOp(Op):
    """Min-p logit filter: masks every token less probable than ``min_p[b]`` times row ``b``'s top.

    A token's probability is below ``min_p[b] * max(prob[b])`` exactly when its logit is
    below ``max_logit + log(min_p[b])``; those logits become ``-inf`` and the rest pass
    through bit for bit. ``0 <= min_p <= 1`` is the caller's obligation. The output has the
    shape and dtype of ``logits``.

    ``min_p = 0`` masks nothing, the value vLLM gives a request that disables min-p, and
    ``min_p = 1`` keeps only the logits equal to the row max. A row holding a NaN has a NaN
    threshold, so every comparison against it is false and the row passes through whole, as
    ``torch.amax`` and ``<`` leave it.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"min_p_mask_fwd": MinPMaskFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "min_p_mask_fwd": MinPMaskFwdInterface
    }

    def __init__(
        self,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtypes are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def forward(self, logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        """Mask the tokens of each row of ``logits`` below its ``min_p`` threshold.

        Args:
            logits: ``[B, V]`` logits, float16, bfloat16 or float32.
            min_p: ``[B]`` float32 fraction of the row's top probability a token must reach.

        Returns:
            ``[B, V]`` logits of ``logits``' dtype, ``-inf`` where masked.
        """
        logits = logits.contiguous()
        min_p = min_p.contiguous()
        batch, vocab = logits.shape
        call = SamplingCall(device=logits.device, batch=batch, vocab=vocab, dtype=logits.dtype)
        kernel = self.kernel_for("min_p_mask_fwd", call)
        return kernel(logits, min_p)
