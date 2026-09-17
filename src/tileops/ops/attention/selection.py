"""The attention call record, re-exported at the op layer.

Which implementation serves a call is stated by the implementations themselves,
through ``applies`` / ``refusal``; see docs/design/ops-design.md § Kernel selection.
"""

from typing import Optional, Sequence

import torch

from tileops.kernels.attention.call_spec import AttentionCall, fp8_dtype

__all__ = [
    "AttentionCall",
    "device_of",
    "fp8_dtype",
]


def device_of(inputs: "Sequence[Optional[torch.Tensor]]") -> Optional[torch.device]:
    """The device the call runs on: the first tensor's, which the op has checked agree."""
    for tensor in inputs:
        if tensor is not None:
            return tensor.device
    return None
