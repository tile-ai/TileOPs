"""The categorical draw from probabilities op."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.sampling import (
    SamplingCall,
    SamplingFromProbsFwdInterface,
    SamplingFromProbsFwdKernel,
)
from tileops.ops.op_base import Op

__all__ = ["SamplingFromProbsFwdOp"]


class SamplingFromProbsFwdOp(Op):
    """Categorical draw: one index per row, with probability proportional to ``probs[b]``.

    The row need not sum to 1: the draw is scaled by the row's total. A zero-weight token is
    never drawn. Each row finite, non-negative and with a positive total is the caller's
    obligation. The draw is a function of the Philox state ``(seed, offset)``: the same pair
    and inputs give the same samples. Which stream a pair selects belongs to the
    implementation, so two implementations need not draw the same samples from one pair.

    The in-tree kernel takes the stream ``workloads/sampling.py`` states: the row's uniform
    is the first output word of Philox4x32-10 keyed by ``(seed, offset)`` and counted by the
    row index. The counter names no launch fact, so the same pair and inputs give the same
    samples whatever the batch and whatever grid the call takes.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "sampling_from_probs": SamplingFromProbsFwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "sampling_from_probs": SamplingFromProbsFwdInterface
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
        return self._call_boundary(probs, seed, offset)

    def _eager_forward(
        self, probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        probs = probs.contiguous()
        seed = seed.contiguous()
        offset = offset.contiguous()
        batch, vocab = probs.shape
        call = SamplingCall(device=probs.device, batch=batch, vocab=vocab, dtype=probs.dtype)
        kernel = self.kernel_for("sampling_from_probs", call)
        return kernel(probs, seed, offset)
