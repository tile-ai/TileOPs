"""The facts of one attention call, and the regions kernels answer for.

``AttentionCall`` is what an op states about a call; the region helpers are the
predicates kernel classes answer ``applies`` with, kept here because more than
one class reads each. See docs/design/ops-design.md § Kernel selection.
"""

import dataclasses
import math
from typing import Optional

import torch

__all__ = [
    "ATTENTION_DTYPES",
    "WS_ARCH",
    "AttentionCall",
    "causal_ws_prefill_region",
    "decode_bs1_region",
    "dense_prefill_region",
    "fp8_dtype",
    "square_ws_prefill_region",
    "uses_sliding_window",
]

ATTENTION_DTYPES = (torch.float16, torch.bfloat16)

_WS_BLOCK_M = 128
_H200_SMS = 132
#: Architecture the warp-specialized prefill kernels are written for. The
#: classes declare it as their ``supported_archs`` and the region below reads
#: the same name, so the two statements of one fact cannot drift apart.
WS_ARCH = 90


def fp8_dtype() -> Optional[torch.dtype]:
    """Return ``torch.float8_e4m3fn`` when the torch build carries it."""
    return getattr(torch, "float8_e4m3fn", None)


@dataclasses.dataclass(frozen=True)
class AttentionCall:
    """What one attention call is, as the op knows it.

    Assembled in ``forward`` from op state plus what only the call knows: the
    element type, whether the packed ranges are uniform, whether the inputs are
    FP8. ``arch`` and ``h200`` describe the device the call will run on; they are
    read when the call is made, never at construction, so an op constructs on a
    machine that cannot run it and is refused when a kernel is selected.
    """

    dtype: Optional[torch.dtype] = None
    arch: int = -1
    h200: bool = False
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
    cache_dtype: Optional[torch.dtype] = None
    fuse_rope: bool = False
    max_position: Optional[int] = None
    rotary_dim: Optional[int] = None
    accum_dtype: torch.dtype = torch.float32
    tune: bool = False

    def __str__(self) -> str:
        """The facts of the call, without the fields nobody set.

        A selection failure names the call, and a record of two dozen mostly
        default fields buries the two that decided it.
        """
        default = AttentionCall(arch=self.arch, h200=self.h200)
        stated = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(self)
            if getattr(self, f.name) != getattr(default, f.name)
        ]
        return ", ".join([f"arch={self.arch}", f"h200={self.h200}", *stated])

    def __post_init__(self) -> None:
        if self.arch < 0:
            from tileops.utils import get_sm_version, is_h200
            object.__setattr__(self, "arch", get_sm_version())
            object.__setattr__(self, "h200", is_h200())


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
    return work_items >= _H200_SMS


def causal_ws_prefill_region(call: AttentionCall) -> bool:
    """The warp-specialized causal packed-prefill region: head dim 128, 16-bit.

    Owned by ``GQAPrefillFwdWsPersistentCausalKernel``; the general dense kernel
    behind it excludes exactly this region.

    The architecture is part of the region, not only of the owning class's
    ``supported_archs``: the general kernel steps aside for this region, so on a
    device where the warp-specialized kernel cannot run the region must be empty
    rather than unserved.
    """
    return (
        dense_prefill_region(call)
        and call.is_causal
        and call.dim == 128
        and call.dtype in ATTENTION_DTYPES
    )


def decode_bs1_region(call: AttentionCall) -> bool:
    """The Hopper batch-1 decode region, shared by contiguous and paged decode.

    Owned by the batch-1 kernels; the general decode kernels behind them exclude
    exactly this region, and the paged batch-1 kernel narrows it further with a
    page-tile condition only it can answer.
    """
    if not (
        call.batch == 1
        and call.dtype == torch.float16
        and call.dim == 128
        and call.softcap == 0.0
    ):
        return False
    if call.heads_kv <= 0 or call.heads % call.heads_kv != 0:
        return False
    return 1 <= call.heads // call.heads_kv <= 64
