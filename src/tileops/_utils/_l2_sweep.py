"""CUDA L2 read sweep with one atomic result per block.

A persistent scalar sink keeps reads observable without a buffer-sized write footprint.

The sweep reads with L2 evict-last priority. Measured on SM90 (H200), where the PTX
documentation does not specify it: a normal-priority fill does not replace an evict-last
line, so a plain sweep leaves the lines a previous kernel read evict-last (as
torch.compile's Triton kernels do) in L2, and an evict-last fill replaces the oldest lines
whatever their priority. The buffer covers ``_L2_FLUSH_FACTOR`` times L2, so only lines
from its last L2's worth remain, and the sweep sets those back to normal priority. The next
kernel then finds the whole L2 available, as after a plain sweep behind
``cudaCtxResetPersistingL2Cache``.
"""

import torch
import triton
import triton.language as tl

from tileops._utils.cupti import _L2_FLUSH_FACTOR

# Persistent device storage avoids allocator traffic on repeated sweeps.
_SINKS: dict[torch.device, torch.Tensor] = {}

_L2_LINE_BYTES = 128
# Demotes the L2 line holding address $1 to normal priority.
_DEMOTE_ASM = tl.constexpr(
    f"applypriority.global.L2::evict_normal [$1], {_L2_LINE_BYTES}; mov.u32 $0, $2;"
)


@triton.jit
def _fold_max(buffer, sink, numel, demote_from, BLOCK: tl.constexpr, LINE: tl.constexpr):
    start = tl.program_id(0) * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    values = tl.load(buffer + offsets, mask=offsets < numel, other=0, eviction_policy="evict_last")
    top = tl.max(values, 0)
    tl.atomic_max(sink, top)
    if start + BLOCK > demote_from:
        # One address in each line this block read, clamped to the buffer's last line.
        lines = tl.minimum(start + tl.arange(0, BLOCK // LINE) * LINE, (numel - 1) // LINE * LINE)
        # The block-wide max ends in a barrier that waits for every load of the block, so
        # the demotion issues after the block's fills.
        tl.inline_asm_elementwise(
            _DEMOTE_ASM,
            "=r,l,r",
            [buffer + lines, top.to(tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


def read_sweep(buffer: torch.Tensor) -> None:
    """Load every element of *buffer* once; the only store is one atomic per block.

    *buffer* covers ``_L2_FLUSH_FACTOR`` times L2 (``l2_flush_buffer``). The sweep leaves no
    line at evict-last priority: its fills replace older evict-last lines, and the last L2's
    worth, the only lines left, return to normal priority.
    """
    sink = _SINKS.get(buffer.device)
    if sink is None:
        sink = _SINKS[buffer.device] = torch.zeros(1, dtype=buffer.dtype, device=buffer.device)
    # 16 KB of int32 per block keeps the atomics to one per 128 lines read.
    block = 4096
    _fold_max[(triton.cdiv(buffer.numel(), block),)](
        buffer,
        sink,
        buffer.numel(),
        # The last L2's worth of lines, the only ones the fills before them do not replace.
        buffer.numel() - buffer.numel() // _L2_FLUSH_FACTOR,
        BLOCK=block,
        LINE=_L2_LINE_BYTES // buffer.element_size(),
    )
