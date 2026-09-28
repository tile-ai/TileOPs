"""Call records for sampling kernels."""

from __future__ import annotations

import dataclasses

import torch

from tileops.kernels.call_spec import CallSpec

__all__ = ["SamplingCall"]


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
