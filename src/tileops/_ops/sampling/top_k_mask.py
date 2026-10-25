"""The top-k logit filter op."""

from typing import ClassVar, Mapping

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.sampling import SamplingCall, TopKMaskFwdInterface, TopKMaskFwdKernel
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["TopKMaskFwdOp"]


class TopKMaskFwdOp(Op):
    """Top-k logit filter: keeps the ``top_k[b]`` largest logits of row ``b``, the rest ``-inf``.

    It returns the filtered logits for sampling, where ``TopKSelectFwdOp`` and
    ``FusedTopKFwdOp`` return the indices they select.

    Every logit equal to the ``top_k[b]``-th largest is kept, so a tie at the threshold
    keeps more than ``top_k[b]``, and a row with ``top_k[b] >= V`` is returned unchanged, as
    FlashInfer's ``top_k_mask_logits`` does. The output has the shape and dtype of
    ``logits``, and each kept logit is passed through bit for bit.

    NaN ranks above every number, as ``torch.sort`` places it, and is masked wherever the
    row is filtered, as ``logits >= kth`` is false at a NaN; a row with ``top_k[b] >= V``
    keeps its NaNs.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"top_k_mask_fwd": TopKMaskFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "top_k_mask_fwd": TopKMaskFwdInterface
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

    def forward(self, logits: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
        """Mask each row of ``logits`` to its ``top_k`` largest.

        Args:
            logits: ``[B, V]`` logits, float16, bfloat16 or float32.
            top_k: ``[B]`` int32 number of logits each row keeps, at least 1.

        Returns:
            ``[B, V]`` logits of ``logits``' dtype, ``-inf`` where masked.
        """
        logits = logits.contiguous()
        top_k = top_k.contiguous()
        batch, vocab = logits.shape
        kernel = self.kernel_for(
            "top_k_mask_fwd",
            lambda: SamplingCall(
                device=logits.device, batch=batch, vocab=vocab, dtype=logits.dtype
            ),
        )
        return kernel(logits, top_k)
