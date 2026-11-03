"""The L2 flush on CUDA: one load of every element of a buffer, and one atomic per block.

The flush is not torch's own reduction: run before a timed call, that kernel slows some
calls (Marlin's w4a16 GEMM among them) that this read leaves at the speed a write sweep
gives them. Why the reduction does so is not established.
"""

import torch
import triton
import triton.language as tl

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
    # 16 KB of int32 per block keeps the atomics to one per 128 lines read.
    block = 4096
    _fold_max[(triton.cdiv(buffer.numel(), block),)](buffer, sink, buffer.numel(), BLOCK=block)
