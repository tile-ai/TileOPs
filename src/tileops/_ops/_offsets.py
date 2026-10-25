"""The ``validate_inputs`` check every op taking caller-supplied cumulative token offsets runs.

The rule covers token offsets: a ``[N + 1]`` tensor whose manifest input requires
``prefix_offsets``. Group offsets such as ``GroupedGEMMFwdOp``'s ``batch_offsets`` are not
token offsets and are not covered.
"""

from typing import Callable, Optional

import torch

__all__ = ["check_offsets", "packed_sequence_check"]


def check_offsets(
    name: str, offsets: torch.Tensor, total: int, mirror: Optional[torch.Tensor] = None
) -> None:
    """Check on the CPU that *offsets* start at 0, never decrease and end at *total*, and
    that *mirror*, a CPU copy of them when given, equals them.

    Raises:
        ValueError: The offsets break one of the rules; the message names *name*.
    """
    bounds = [int(value) for value in offsets.detach().cpu().tolist()]
    if bounds[0] != 0:
        raise ValueError(f"{name}[0] must equal 0")
    if bounds[-1] != total:
        raise ValueError(f"{name}[-1] must equal {total}")
    if any(end < start for start, end in zip(bounds[:-1], bounds[1:], strict=True)):
        raise ValueError(f"{name} must be non-decreasing")
    if mirror is not None and [int(value) for value in mirror.tolist()] != bounds:
        raise ValueError(f"{name}_cpu must equal {name}")


def packed_sequence_check(position: int) -> Callable[..., None]:
    """The ``_check_input_values`` of an op whose inputs *position* and *position* + 1 are
    ``cu_seqlens`` and its CPU mirror, offsets into axis 1 of input 0."""

    def check(op, *inputs: Optional[torch.Tensor]) -> None:
        offsets = inputs[position]
        if op.validate_inputs and offsets is not None:
            check_offsets("cu_seqlens", offsets, inputs[0].shape[1], inputs[position + 1])

    return check
