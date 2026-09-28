"""The facts of one attention call, and the regions kernels answer for.

``AttentionCall`` is what an op states about a call; the region helpers are the
predicates kernel classes answer ``applies`` with, kept here because more than
one class reads each. See docs/design/ops-design.md § Kernel selection.
"""

import dataclasses
from typing import Optional

import torch

from ..call_spec import CallSpec

__all__ = [
    "ATTENTION_DTYPES",
    "AttentionCall",
    "dense_decode_limit_refusal",
    "dense_decode_refusal",
    "dense_decode_region",
    "dense_fp8_decode_refusal",
    "dense_long_context_decode_refusal",
    "dense_fp8_limit_refusal",
    "dense_fp8_refusal",
    "dense_long_context_decode_region",
    "dense_fp8_decode_region",
    "dense_sliding_window_refusal",
    "dense_sliding_window_region",
    "dense_ws_refusal",
    "dense_ws_region",
    "decode_bs1_region",
    "paged_decode_region",
    "paged_decode_refusal",
    "uses_sliding_window",
]

ATTENTION_DTYPES = (torch.float16, torch.bfloat16)


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


def dense_decode_region(call: AttentionCall) -> bool:
    """The contiguous decode region: one query position, no window, not FP8."""
    return not call.is_fp8 and call.max_seqlen_q == 1 and not uses_sliding_window(call)


def paged_decode_refusal(call: AttentionCall) -> Optional[str]:
    """Why the paged-decode kernels cannot serve *call*, or ``None`` when they can.

    They serve one query length shared by every request against a 16-bit cache of
    the query's dtype, with no window, RoPE or FP8.
    """
    if call.max_seqlen_q < 1 or not call.is_uniform:
        return "requires the same query length for every request"
    if call.dtype not in ATTENTION_DTYPES:
        return "requires float16 or bfloat16 Q"
    if call.cache_dtype != call.dtype:
        return "requires Q and KV to share a dtype"
    if call.is_fp8:
        return "does not serve FP8"
    if uses_sliding_window(call):
        return "does not serve sliding windows"
    if call.fuse_rope:
        return "does not serve RoPE"
    return None


def paged_decode_region(call: AttentionCall) -> bool:
    """The region the paged-decode kernels share; see :func:`paged_decode_refusal`."""
    return paged_decode_refusal(call) is None


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


# What the contiguous kernels refuse inside their regions. Each returns the limit a
# call fails, or ``None``; outside the region it answers "does not serve this call".
_OUTSIDE_REGION = "does not serve this call"


def _tensor_core_dim_refusal(dim: int) -> Optional[str]:
    """The prefill and windowed kernels step the head dimension by one MMA k-slice."""
    return None if dim % 16 == 0 else "requires head dimension a multiple of 16"


def dense_decode_limit_refusal(*, dim: int, seq_len_kv: int) -> Optional[str]:
    """Why the contiguous decode kernel cannot build this shape, or ``None``.

    Read by the kernel's region and by its constructor, which a caller can reach directly.
    Past 128 the split-KV tile has no register layout.
    """
    if seq_len_kv <= 0:
        return "requires a non-empty KV cache"
    if dim % 16 != 0 or not 16 <= dim <= 128:
        return "requires head dimension a multiple of 16 in [16, 128]"
    return None


def dense_ws_refusal(call: AttentionCall) -> Optional[str]:
    """Why the contiguous prefill kernel cannot serve *call*, or ``None``."""
    if not dense_ws_region(call):
        return _OUTSIDE_REGION
    return _tensor_core_dim_refusal(call.dim)


def dense_sliding_window_refusal(call: AttentionCall) -> Optional[str]:
    """Why the contiguous windowed kernel cannot serve *call*, or ``None``."""
    if not dense_sliding_window_region(call):
        return _OUTSIDE_REGION
    if call.max_seqlen_q != call.seqlen_kv:
        return "a sliding window requires equal Q and KV lengths"
    return _tensor_core_dim_refusal(call.dim)


def dense_decode_refusal(call: AttentionCall) -> Optional[str]:
    """Why the contiguous decode kernels cannot serve *call*, or ``None``."""
    if not dense_decode_region(call):
        return _OUTSIDE_REGION
    return dense_decode_limit_refusal(dim=call.dim, seq_len_kv=call.seqlen_kv)


def dense_long_context_decode_refusal(call: AttentionCall) -> Optional[str]:
    """Why the long-context decode kernel cannot serve *call*, or ``None``."""
    return None if dense_long_context_decode_region(call) else _OUTSIDE_REGION


def dense_fp8_decode_refusal(call: AttentionCall) -> Optional[str]:
    """Why the FP8 decode kernel cannot serve *call*, or ``None``."""
    if not dense_fp8_decode_region(call):
        return _OUTSIDE_REGION
    # Each of the four consumer warps takes a quarter of the head dimension in 8-wide tiles.
    if call.dim % 32 != 0 or not 32 <= call.dim <= 128:
        return "requires head dimension a multiple of 32 in [32, 128]"
    return None


def dense_fp8_refusal(call: AttentionCall) -> Optional[str]:
    """Why the FP8 main kernel cannot serve *call*, or ``None``."""
    if not call.is_fp8 or dense_fp8_decode_region(call):
        return _OUTSIDE_REGION
    return dense_fp8_limit_refusal(
        dim=call.dim,
        seq_len_q=call.max_seqlen_q,
        seq_len_kv=call.seqlen_kv,
        is_causal=call.is_causal,
        softcap=call.softcap,
        window=(call.window_size_left, call.window_size_right),
        fuse_rope=call.fuse_rope,
    )


def dense_fp8_limit_refusal(
    *,
    dim: int,
    seq_len_q: int,
    seq_len_kv: int,
    is_causal: bool,
    softcap: float,
    window: tuple[int, int],
    fuse_rope: bool,
) -> Optional[str]:
    """Why the FP8 main kernel cannot build this shape, or ``None``.

    Read by the kernel's region and by its constructor, which a caller can reach directly.
    """
    if window != (-1, -1):
        return "does not serve sliding windows"
    if fuse_rope and seq_len_q == 1:
        return "does not serve RoPE with one query position"
    if not is_causal and seq_len_q != seq_len_kv:
        return "non-causal attention requires equal Q and KV lengths"
    if not is_causal and softcap != 0.0:
        return "does not serve a softcap without the causal mask"
    if dim != 128:
        return "requires head dimension 128"
    return None


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
