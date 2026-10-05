import torch


def _dtype(call, tensor: str) -> torch.dtype:
    return getattr(torch, call.tensors[tensor][1])


def _segments(bounds: list[int]) -> list[int]:
    return [end - start for start, end in zip(bounds, bounds[1:], strict=False)]
