"""The facts of one attention call, and the regions kernels answer for.

``AttentionCall`` is what an op states about a call; the region helpers are the
predicates kernel classes answer ``applies`` with, kept here because more than
one class reads each. See docs/design/ops-design.md § Kernel selection.
"""

import dataclasses
import math
from typing import Optional

import torch

from ..call_spec import CallSpec

__all__ = [
    "ATTENTION_DTYPES",
    "WS_ARCH",
    "AttentionCall",
    "dense_decode_region",
    "dense_long_context_decode_region",
    "dense_fp8_decode_region",
    "dense_sliding_window_region",
    "dense_ws_region",
    "decode_bs1_region",
    "dense_prefill_region",
    "fp8_dtype",
    "paged_decode_ws_region",
    "square_ws_prefill_region",
    "uses_sliding_window",
]

ATTENTION_DTYPES = (torch.float16, torch.bfloat16)

_WS_BLOCK_M = 128
# Architecture the warp-specialized prefill kernels are written for. The
# classes declare it as their ``supported_archs`` and the region below reads
# the same name, so the two statements of one fact cannot drift apart.
WS_ARCH = 90


def fp8_dtype() -> Optional[torch.dtype]:
    """Return ``torch.float8_e4m3fn`` when the torch build carries it."""
    return getattr(torch, "float8_e4m3fn", None)


@dataclasses.dataclass(frozen=True)
class AttentionCall(CallSpec):
    """What one attention call is, as the op knows it.

    Assembled in ``forward`` from op state plus what only the call knows: the
    element type, whether the packed ranges are uniform, whether the inputs are
    FP8. The device fields come from ``CallSpec``.
    """

    dtype: Optional[torch.dtype] = None
    batch: int = 0
    heads: int = 0
    heads_kv: int = 0
    dim: int = 0
    max_seqlen_q: int = 0
    max_seqlen_kv: int = 0
    seqlen_kv: int = 0
    page_size: int = 0
    max_pages_per_req: int = 0
    is_causal: bool = False
    sm_scale: Optional[float] = None
    softcap: float = 0.0
    window_size_left: int = -1
    window_size_right: int = -1
    backend: str = "auto"
    is_fp8: bool = False
    is_uniform: bool = True
    # Every packed KV range is empty, so a TMA descriptor over K/V has no extent.
    empty_kv: bool = False
    cache_dtype: Optional[torch.dtype] = None
    fuse_rope: bool = False
    max_position: Optional[int] = None
    rotary_dim: Optional[int] = None
    rope_layout: str = "neox"
    accum_dtype: torch.dtype = torch.float32


def uses_sliding_window(call: AttentionCall) -> bool:
    """Whether either window bound is set, which restricts what may serve the call."""
    return call.window_size_left != -1 or call.window_size_right != -1


def dense_prefill_region(call: AttentionCall) -> bool:
    """What every dense packed-prefill implementation requires of a call.

    A dense implementation computes on a BSHD view of the packed tensors, so it
    serves a uniform request only. FP8 and sliding-window calls are regions of
    their own with their own implementations, and an explicit ``backend`` naming
    one of those asks for it by name.
    """
    return (
        not call.is_fp8
        and not uses_sliding_window(call)
        and call.is_uniform
        and call.backend in ("auto", "dense")
    )


def square_ws_prefill_region(call: AttentionCall) -> bool:
    """The H200 square causal packed-prefill region.

    Owned by ``GQAFwdWsPersistentCausalKernel``; the warp-specialized causal
    kernel behind it excludes exactly this region so the two never both apply.
    """
    if not dense_prefill_region(call):
        return False
    if call.dtype not in ATTENTION_DTYPES:
        return False
    if not call.h200 or call.dim != 128:
        return False
    if call.heads_kv <= 0 or call.heads % call.heads_kv != 0:
        return False
    if call.max_seqlen_q % _WS_BLOCK_M != 0:
        return False
    m_blocks = math.ceil(call.max_seqlen_q / _WS_BLOCK_M)
    if m_blocks % 2 != 0:
        return False
    if not call.is_causal or call.max_seqlen_q != call.max_seqlen_kv:
        return False
    groups = call.heads // call.heads_kv
    work_items = call.batch * call.heads_kv * (m_blocks // 2) * groups
    # A persistent kernel earns its prologue once the work fills the grid, so the
    # bar is the device's own SM count rather than the number the board this was
    # fitted on happens to report.
    return work_items >= call.sm_count


# Tile heights the warp-specialized paged decode kernel can pick from. A tile
# divides the page size, so one tile never straddles two pages, and it splits
# evenly across the four consumer warps.
_WS_DECODE_TILES = (16, 32, 64, 128)
# Head dims that map onto one warp: the score reduction is a shuffle chain over
# 32 lanes, so a lane owns ``dim / 32`` elements of the head vector.
_WS_DECODE_LANES = 32


def paged_decode_ws_region(call: AttentionCall) -> bool:
    """The paged-decode region the warp-specialized MHA kernel serves.

    Stated positively, and only in terms the call already carries. What is left
    to the general kernel: a query longer than one token (this kernel's whole
    reason for skipping the tensor cores is that ``seqlen_q`` is 1), a head dim
    that does not divide across a warp, a page size no tile height divides, a
    softcap, and a causal request -- which for a one-token query against a
    finished cache is not the same computation.
    """
    if call.max_seqlen_q != 1 or call.is_causal or call.softcap != 0.0:
        return False
    if call.dtype not in ATTENTION_DTYPES or call.is_fp8:
        return False
    if uses_sliding_window(call):
        return False
    if call.dim % _WS_DECODE_LANES != 0 or not 0 < call.dim <= 256:
        return False
    if call.page_size <= 0 or call.seqlen_kv <= 0:
        return False
    return any(
        tile <= call.page_size and call.page_size % tile == 0 and tile <= call.seqlen_kv
        for tile in _WS_DECODE_TILES
    )


def dense_decode_region(call: AttentionCall) -> bool:
    """The contiguous decode region: one query position, no window, not FP8."""
    return not call.is_fp8 and call.max_seqlen_q == 1 and not uses_sliding_window(call)


def dense_long_context_decode_region(call: AttentionCall) -> bool:
    """The one decode shape the long-context split serves."""
    return (
        dense_decode_region(call)
        and not call.fuse_rope
        and call.seqlen_kv >= 1024
        and call.batch == 1
        and call.heads == 32
        and call.heads_kv == 4
        and call.dim == 128
        and call.dtype == torch.float16
        and call.softcap == 0.0
    )


def dense_fp8_decode_region(call: AttentionCall) -> bool:
    """The FP8 decode region: batch 1, one query position, a long cache."""
    return (
        call.is_fp8
        and call.batch == 1
        and call.max_seqlen_q == 1
        and call.seqlen_kv >= 2048
        and call.heads_kv > 0
        and call.heads // call.heads_kv <= 16
        and not uses_sliding_window(call)
        and not call.fuse_rope
    )


def dense_sliding_window_region(call: AttentionCall) -> bool:
    """The contiguous windowed region, which FP8 has its own implementation for."""
    return not call.is_fp8 and uses_sliding_window(call)


def dense_ws_region(call: AttentionCall) -> bool:
    """The contiguous prefill region: more than one query position, no window."""
    return not call.is_fp8 and call.max_seqlen_q != 1 and not uses_sliding_window(call)


def decode_bs1_region(call: AttentionCall) -> bool:
    """The SM90 batch-1 decode region, shared by contiguous and paged decode.

    Owned by the batch-1 kernels; the general decode kernels behind them exclude
    exactly this region, and the paged batch-1 kernel narrows it further with a
    page-tile condition only it can answer.
    """
    if not (
        call.batch == 1 and call.dtype == torch.float16 and call.dim == 128 and call.softcap == 0.0
    ):
        return False
    if call.heads_kv <= 0 or call.heads % call.heads_kv != 0:
        return False
    return 1 <= call.heads // call.heads_kv <= 64
