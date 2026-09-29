"""The top-k then top-p logit filter op."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sampling.call_spec import SamplingCall
from tileops.ops.op_base import Op

__all__ = ["TopKTopPMaskFwdOp"]


class TopKTopPMaskFwdOp(Op):
    """Top-k then top-p logit filter: ``TopPMaskFwdOp(TopKMaskFwdOp(logits, k), p)``.

    The top-k filter of ``TopKMaskFwdOp`` runs first; top-p's probabilities are then the
    softmax over the tokens it kept, with ties at either boundary kept together.
    ``0 < p < 1`` is the caller's obligation. The output has the shape and dtype of
    ``logits``.

    No in-tree kernel implements this op yet, so a call raises ``OpNotAvailableError``
    unless a target serves it.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {}

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
        logits = logits.contiguous()
        k = k.contiguous()
        p = p.contiguous()
        batch, vocab = logits.shape
        call = SamplingCall(
            device=logits.device, batch=batch, vocab=vocab, dtype=logits.dtype, tune=self.tune
        )
        kernel = self.kernel_for("top_k_top_p_mask", (logits, k, p), call)
        return kernel(logits, k, p)
