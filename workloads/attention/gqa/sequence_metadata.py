from itertools import accumulate

import torch

from workloads.device import run_device

__all__ = ["make_cu_seqlens"]


def make_cu_seqlens(lengths: list[int]) -> torch.Tensor:
    """Exclusive prefix sum of *lengths*, the packed-varlen offset vector."""
    return torch.tensor([0, *accumulate(lengths)], device=run_device(), dtype=torch.int32)


def _dtype(call, tensor: str) -> torch.dtype:
    return getattr(torch, call.tensors[tensor][1])


def _segments(bounds: list[int]) -> list[int]:
    return [end - start for start, end in zip(bounds, bounds[1:], strict=False)]
