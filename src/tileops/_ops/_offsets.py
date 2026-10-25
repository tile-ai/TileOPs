"""The ``validate_inputs`` check every packed (varlen) op runs on its cumulative offsets."""

import torch

__all__ = ["check_offsets"]


def check_offsets(name: str, offsets: torch.Tensor, total: int) -> None:
    """Check on the CPU that *offsets* start at 0, never decrease and end at *total*.

    Raises:
        ValueError: The offsets break one of the three rules; the message names *name*.
    """
    bounds = [int(value) for value in offsets.detach().cpu().tolist()]
    if bounds[0] != 0:
        raise ValueError(f"{name}[0] must equal 0")
    if bounds[-1] != total:
        raise ValueError(f"{name}[-1] must equal {total}")
    if any(end < start for start, end in zip(bounds[:-1], bounds[1:], strict=True)):
        raise ValueError(f"{name} must be non-decreasing")
