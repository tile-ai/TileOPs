"""The L2 flush on CUDA: one read of a buffer, and nothing else the next call can feel.

torch's reduction kernels leave the device in a state that slows some kernels launched
after them, whatever L2 holds; a single pass of loads folded into one atomic maximum
does not, so the timed call inherits a cold L2 and no other trace of the flush.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 4096
# One per device, so a flush allocates nothing the timed call's allocations could land on.
_sinks: dict[torch.device, torch.Tensor] = {}


@triton.jit
def _fold_max(buffer, sink, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(buffer + offsets, mask=offsets < numel, other=0)
    tl.atomic_max(sink, tl.max(values, 0))


def read_sweep(buffer: torch.Tensor) -> None:
    """Load every element of *buffer* once; the only store is one atomic per block."""
    sink = _sinks.get(buffer.device)
    if sink is None:
        sink = _sinks[buffer.device] = torch.zeros(1, dtype=buffer.dtype, device=buffer.device)
    _fold_max[(triton.cdiv(buffer.numel(), _BLOCK),)](buffer, sink, buffer.numel(), BLOCK=_BLOCK)
