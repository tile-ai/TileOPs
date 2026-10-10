"""The top-p (nucleus) logit filter op."""

from typing import ClassVar, Mapping

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.sampling import SamplingCall, TopPMaskFwdInterface, TopPMaskFwdKernel
from tileops.ops.op_base import Op

__all__ = ["TopPMaskFwdOp"]


class TopPMaskFwdOp(Op):
    """Top-p logit filter: keeps the most probable tokens of row ``b`` until they hold ``p[b]``.

    With ``prob = softmax(logits[b])``, a token survives while the total probability of the
    tokens strictly more probable than it is below ``p[b]``, so tokens tied at the boundary
    are kept together, as FlashInfer's ``top_p_renorm_probs`` keeps them. Masked logits
    become ``-inf``; the rest pass through bit for bit. ``0 <= p <= 1`` is the caller's
    obligation. The output has the shape and dtype of ``logits``.

    ``p = 0`` masks the row whole. ``p = 1`` keeps every token whose probability survives
    the row's float32 sum, and masks the ones the sum rounds away with the zero-probability
    ones. A row whose maximum is not finite, a NaN or an infinity among its logits, has an
    all-NaN softmax, so nothing in it is masked and it passes through whole.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"top_p_mask_fwd": TopPMaskFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "top_p_mask_fwd": TopPMaskFwdInterface
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

    def forward(self, logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Mask each row of ``logits`` to its top-``p`` nucleus.

        Args:
            logits: ``[B, V]`` logits, float16, bfloat16 or float32.
            p: ``[B]`` float32 probability mass each row keeps.

        Returns:
            ``[B, V]`` logits of ``logits``' dtype, ``-inf`` where masked.
        """
        logits = logits.contiguous()
        p = p.contiguous()
        batch, vocab = logits.shape
        call = SamplingCall(device=logits.device, batch=batch, vocab=vocab, dtype=logits.dtype)
        kernel = self.kernel_for("top_p_mask_fwd", call)
        return kernel(logits, p)
