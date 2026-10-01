"""The top-k logit filter op."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.sampling import SamplingCall, TopKMaskFwdInterface, TopKMaskFwdKernel
from tileops.ops.op_base import Op

__all__ = ["TopKMaskFwdOp"]


class TopKMaskFwdOp(Op):
    """Top-k logit filter: keeps the ``k[b]`` largest logits of row ``b``, the rest ``-inf``.

    Every logit equal to the ``k[b]``-th largest is kept, so a tie at the threshold keeps
    more than ``k[b]``, and a row with ``k[b] >= V`` is returned unchanged, as FlashInfer's
    ``top_k_mask_logits`` does. The output has the shape and dtype of ``logits``, and each
    kept logit is passed through bit for bit.

    NaN ranks above every number, as ``torch.sort`` places it, and is masked wherever the
    row is filtered, as ``logits >= kth`` is false at a NaN; a row with ``k[b] >= V`` keeps
    its NaNs.
    """

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"top_k_mask_fwd": TopKMaskFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "top_k_mask_fwd": TopKMaskFwdInterface
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

    def forward(self, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Mask each row of ``logits`` to its ``k`` largest.

        Args:
            logits: ``[B, V]`` logits, float16, bfloat16 or float32.
            k: ``[B]`` int32 number of logits each row keeps, at least 1.

        Returns:
            ``[B, V]`` logits of ``logits``' dtype, ``-inf`` where masked.
        """
        return self._call_boundary(logits, k)

    def _eager_forward(self, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        logits = logits.contiguous()
        k = k.contiguous()
        batch, vocab = logits.shape
        call = SamplingCall(device=logits.device, batch=batch, vocab=vocab, dtype=logits.dtype)
        kernel = self.kernel_for("top_k_mask_fwd", call)
        return kernel(logits, k)
