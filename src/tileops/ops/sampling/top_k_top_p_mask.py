"""The top-k then top-p logit filter op."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.sampling import SamplingCall, TopKTopPMaskFwdInterface, TopKTopPMaskFwdKernel
from tileops.ops.op_base import Op

__all__ = ["TopKTopPMaskFwdOp"]


class TopKTopPMaskFwdOp(Op):
    """Top-k then top-p logit filter: ``TopPMaskFwdOp(TopKMaskFwdOp(logits, k), p)``.

    The top-k filter of ``TopKMaskFwdOp`` runs first; top-p's probabilities are then the
    softmax over the tokens it kept, with ties at either boundary kept together.
    ``0 < p < 1`` is the caller's obligation. The output has the shape and dtype of
    ``logits``, and each kept logit is passed through bit for bit, except a NaN, which comes
    back as a quiet NaN rather than its own payload.

    NaN ranks above every number, as ``torch.sort`` places it, so it is masked wherever
    ``k`` filters the row; a row with ``k[b] >= V`` keeps its NaNs, and a row whose largest
    surviving logit is NaN or ``+inf`` has no finite probability for top-p to compare, so
    top-p removes nothing from it.

    The top-p bound is settled from float32 weights summed in bin order, where the reference
    sums them in sorted order, so the two can disagree only on a token whose exclusive
    probability sits within a few rounding steps of ``p``.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "top_k_top_p_mask_fwd": TopKTopPMaskFwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "top_k_top_p_mask_fwd": TopKTopPMaskFwdInterface
    }

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtypes are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(self, logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Mask each row of ``logits`` to its top ``k``, then to the top-``p`` nucleus of those.

        Args:
            logits: ``[B, V]`` logits, float16, bfloat16 or float32.
            k: ``[B]`` int32 number of logits each row keeps, at least 1.
            p: ``[B]`` float32 probability mass each row keeps among its top ``k``.

        Returns:
            ``[B, V]`` logits of ``logits``' dtype, ``-inf`` where masked.
        """
        return self._call_boundary(logits, k, p)

    def _eager_forward(
        self, logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        logits = logits.contiguous()
        k = k.contiguous()
        p = p.contiguous()
        batch, vocab = logits.shape
        call = SamplingCall(device=logits.device, batch=batch, vocab=vocab, dtype=logits.dtype)
        kernel = self.kernel_for("top_k_top_p_mask_fwd", call)
        return kernel(logits, k, p)
