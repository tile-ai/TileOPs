"""Call records for sampling kernels."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = ["SamplingCall", "TopKMaskFwdInterface"]


@dataclasses.dataclass(frozen=True)
class SamplingCall(CallSpec):
    """The facts that select a logit filter or token draw implementation and build it.

    Each call works on ``batch`` rows of ``vocab`` entries. ``dtype`` is the dtype of the
    rows: the logits of a filter, the probabilities of a draw. ``num_draft`` is the number
    of draft tokens per row a speculative verification checks, and 0 for every other op.
    """

    batch: int = 0
    vocab: int = 0
    dtype: torch.dtype = torch.float32
    num_draft: int = 0


class TopKMaskFwdInterface(KernelInterface):
    """Top-k logit mask: keep each row's values at least its ``k[b]``-th largest, the rest ``-inf``."""

    request = SamplingCall

    @abstractmethod
    def forward(self, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Mask each of the ``call.batch`` rows of *logits*; nothing is written in place.

        Args:
            logits: ``[call.batch, call.vocab]``, contiguous, in ``call.dtype`` on ``call.device``.
            k: ``[call.batch]`` ``int32``, each at least 1, on ``call.device``.

        Returns:
            A new tensor shaped like *logits*, ``-inf`` where masked.
        """
