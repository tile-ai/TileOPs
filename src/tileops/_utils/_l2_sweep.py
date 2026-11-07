"""CUDA L2 read sweep with one atomic result per block.

A persistent scalar sink keeps reads observable without a buffer-sized write footprint.
"""

import torch
import triton
import triton.language as tl

# Persistent device storage avoids allocator traffic on repeated sweeps.
_SINKS: dict[torch.device, torch.Tensor] = {}


@triton.jit
def _fold_max(buffer, sink, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(buffer + offsets, mask=offsets < numel, other=0)
    tl.atomic_max(sink, tl.max(values, 0))


def read_sweep(buffer: torch.Tensor) -> None:
    """Load every element of *buffer* once; the only store is one atomic per block."""
    sink = _SINKS.get(buffer.device)
    if sink is None:
        sink = _SINKS[buffer.device] = torch.zeros(1, dtype=buffer.dtype, device=buffer.device)
    # 16 KB of int32 per block keeps the atomics to one per 128 lines read.
    block = 4096
    _fold_max[(triton.cdiv(buffer.numel(), block),)](buffer, sink, buffer.numel(), BLOCK=block)
