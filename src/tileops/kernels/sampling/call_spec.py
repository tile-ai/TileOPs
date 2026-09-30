"""Call records for sampling kernels."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = [
    "MinPMaskFwdInterface",
    "SamplingCall",
    "SamplingFromProbsFwdInterface",
    "TopKMaskFwdInterface",
    "TopKTopPMaskFwdInterface",
    "TopPMaskFwdInterface",
]


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


class MinPMaskFwdInterface(KernelInterface):
    """Min-p logit mask: keep each row's logits at least its row max plus ``log(min_p[b])``."""

    request = SamplingCall

    @abstractmethod
    def forward(self, logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        """Mask each of the ``call.batch`` rows of *logits*; nothing is written in place.

        Args:
            logits: ``[call.batch, call.vocab]``, contiguous, in ``call.dtype`` on ``call.device``.
            min_p: ``[call.batch]`` ``float32`` in ``[0, 1]``, on ``call.device``.

        Returns:
            A new tensor shaped like *logits*, ``-inf`` where masked.
        """


class TopKTopPMaskFwdInterface(KernelInterface):
    """Top-k then top-p logit mask: keep each row's top ``k[b]``, then their nucleus ``p[b]``."""

    request = SamplingCall

    @abstractmethod
    def forward(self, logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Mask each of the ``call.batch`` rows of *logits*; nothing is written in place.

        Args:
            logits: ``[call.batch, call.vocab]``, contiguous, in ``call.dtype`` on ``call.device``.
            k: ``[call.batch]`` ``int32``, each at least 1, on ``call.device``.
            p: ``[call.batch]`` ``float32`` in ``(0, 1)``, on ``call.device``.

        Returns:
            A new tensor shaped like *logits*, ``-inf`` where masked.
        """


class TopPMaskFwdInterface(KernelInterface):
    """Top-p logit mask: keep each row's nucleus, the tokens the mass ``p[b]`` reaches."""

    request = SamplingCall

    @abstractmethod
    def forward(self, logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Mask each of the ``call.batch`` rows of *logits*; nothing is written in place.

        Args:
            logits: ``[call.batch, call.vocab]``, contiguous, in ``call.dtype`` on ``call.device``.
            p: ``[call.batch]`` ``float32`` in ``[0, 1]``, on ``call.device``.

        Returns:
            A new tensor shaped like *logits*, ``-inf`` where masked.
        """


class SamplingFromProbsFwdInterface(KernelInterface):
    """Categorical draw: one token index per row, with probability proportional to the row."""

    request = SamplingCall

    @abstractmethod
    def forward(
        self, probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """Draw one index from each of the ``call.batch`` rows of *probs*.

        Args:
            probs: ``[call.batch, call.vocab]``, contiguous, ``float32`` on ``call.device``,
                each row finite, non-negative and with a positive total.
            seed: ``[1]`` ``int64`` Philox seed, on ``call.device``.
            offset: ``[1]`` ``int64`` Philox offset, on ``call.device``.

        Returns:
            A new ``[call.batch]`` ``int32`` tensor of drawn indices.
        """
