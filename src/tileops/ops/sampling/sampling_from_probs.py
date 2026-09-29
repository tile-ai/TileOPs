"""The categorical draw from probabilities op."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sampling.call_spec import SamplingCall
from tileops.ops.op_base import Op

__all__ = ["SamplingFromProbsFwdOp"]


class SamplingFromProbsFwdOp(Op):
    """Categorical draw: one index per row, with probability proportional to ``probs[b]``.

    The row need not sum to 1: the draw is scaled by the row's total. A zero-weight token is
    never drawn. Each row finite, non-negative and with a positive total is the caller's
    obligation. The draw is a function of the Philox state ``(seed, offset)``: the same pair
    and inputs give the same samples. Which stream a pair selects belongs to the
    implementation, so two implementations need not draw the same samples from one pair.

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

    def forward(
        self, probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """Draw one token per row of ``probs``.

        Args:
            probs: ``[B, V]`` float32 unnormalized probabilities.
            seed: ``[1]`` int64 Philox seed.
            offset: ``[1]`` int64 Philox offset.

        Returns:
            ``[B]`` int32 drawn indices.
        """
        probs = probs.contiguous()
        seed = seed.contiguous()
        offset = offset.contiguous()
        batch, vocab = probs.shape
        call = SamplingCall(
            device=probs.device, batch=batch, vocab=vocab, dtype=probs.dtype, tune=self.tune
        )
        kernel = self.kernel_for("sampling_from_probs", (probs, seed, offset), call)
        return kernel(probs, seed, offset)
