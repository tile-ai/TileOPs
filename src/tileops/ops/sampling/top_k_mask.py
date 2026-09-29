"""The top-k logit filter op."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sampling.call_spec import SamplingCall
from tileops.ops.op_base import Op

__all__ = ["TopKMaskFwdOp"]


class TopKMaskFwdOp(Op):
    """Top-k logit filter: keeps the ``k[b]`` largest logits of row ``b``, the rest ``-inf``.

    Every logit equal to the ``k[b]``-th largest is kept, so a tie at the threshold keeps
    more than ``k[b]``, and a row with ``k[b] >= V`` is returned unchanged, as FlashInfer's
    ``top_k_mask_logits`` does. The output has the shape and dtype of ``logits``.

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

    def forward(self, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Mask each row of ``logits`` to its ``k`` largest.

        Args:
            logits: ``[B, V]`` logits, float16, bfloat16 or float32.
            k: ``[B]`` int32 number of logits each row keeps, at least 1.

        Returns:
            ``[B, V]`` logits of ``logits``' dtype, ``-inf`` where masked.
        """
        logits = logits.contiguous()
        k = k.contiguous()
        batch, vocab = logits.shape
        call = SamplingCall(
            device=logits.device, batch=batch, vocab=vocab, dtype=logits.dtype, tune=self.tune
        )
        kernel = self.kernel_for("top_k_mask", (logits, k), call)
        return kernel(logits, k)
