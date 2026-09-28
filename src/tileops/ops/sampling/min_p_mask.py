"""The min-p logit filter op."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sampling.call_spec import SamplingCall

from ..op_base import Op

__all__ = ["MinPMaskFwdOp"]


class MinPMaskFwdOp(Op):
    """Min-p logit filter: masks every token less probable than ``min_p[b]`` times row ``b``'s top.

    A token's probability is below ``min_p[b] * max(prob[b])`` exactly when its logit is
    below ``max_logit + log(min_p[b])``; those logits become ``-inf`` and the rest pass
    unchanged. ``0 < min_p <= 1`` is the caller's obligation. The output has the shape and
    dtype of ``logits``.

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
        call = SamplingCall(
            device=logits.device, batch=batch, vocab=vocab, dtype=logits.dtype, tune=self.tune
        )
        kernel = self.kernel_for("min_p_mask", (logits, min_p), call)
        return kernel(logits, min_p)
