"""How each contiguous GQA implementation is built from a call, and what keys it.

Four identity shapes, because the implementations specialize on different facts:
the decode programs take the cache length at runtime, the sliding and RoPE
programs compile exact extents, and the long-context split also tiers on length.
"""

from typing import Optional

from tileops.kernels.attention.call_spec import AttentionCall
from tileops.kernels.kernel_base import Entry

__all__ = [
    "dense_decode_entry",
    "dense_fp8_decode_entry",
    "dense_fp8_entry",
    "dense_sliding_window_entry",
    "dense_ws_entry",
]


def _rope_kwargs(call: AttentionCall) -> dict:
    return {
        "fuse_rope": call.fuse_rope,
        "max_position": call.max_position if call.max_position is not None else 1,
        "rotary_dim": call.rotary_dim if call.rotary_dim is not None else 0,
        "rope_layout": call.rope_layout,
    }


def _index(call: AttentionCall) -> Optional[int]:
    return call.device.index if call.device is not None else None


def dense_fp8_decode_entry(cls: type, call: AttentionCall) -> Entry:
    """Skv is dynamic in this program, so one object serves every cache length."""
    args = dict(
        batch=call.batch,
        heads=call.heads,
        heads_kv=call.heads_kv,
        dim=call.dim,
        dtype=call.dtype,
        sm_scale=call.sm_scale,
        softcap=call.softcap,
        device_index=_index(call),
        tune=call.tune,
    )
    return tuple(args.values()), lambda: cls(**args)


def dense_fp8_entry(cls: type, call: AttentionCall) -> Entry:
    """Compiles exact extents, so both sequence lengths are in the identity."""
    args = dict(
        batch=call.batch,
        heads=call.heads,
        heads_kv=call.heads_kv,
        seq_len_q=call.max_seqlen_q,
        seq_len_kv=call.seqlen_kv,
        dim=call.dim,
        is_causal=call.is_causal,
        window_size_left=call.window_size_left,
        window_size_right=call.window_size_right,
        dtype=call.dtype,
        sm_scale=call.sm_scale,
        softcap=call.softcap,
        **_rope_kwargs(call),
        device_index=_index(call),
        tune=call.tune,
    )
    return tuple(args.values()), lambda: cls(**args)


def dense_decode_entry(cls: type, call: AttentionCall) -> Entry:
    """Decode takes the cache length at runtime; the split tier is what varies."""
    args = dict(
        batch=call.batch,
        heads=call.heads,
        heads_kv=call.heads_kv,
        seq_len_kv=call.seqlen_kv,
        dim=call.dim,
        dtype=call.dtype,
        sm_scale=call.sm_scale,
        softcap=call.softcap,
        **_rope_kwargs(call),
        device_index=_index(call),
        tune=call.tune,
    )
    # Every construction argument but the cache length, which this program takes at
    # runtime: what it compiles for is the split tier the length falls in.
    identity = (*(v for k, v in args.items() if k != "seq_len_kv"), *cls.split_tier(call))
    return identity, lambda: cls(**args)


def dense_sliding_window_entry(cls: type, call: AttentionCall) -> Entry:
    """Compiles exact extents, so the query length is in the identity."""
    args = dict(
        batch=call.batch,
        heads=call.heads,
        heads_kv=call.heads_kv,
        seq_len=call.max_seqlen_q,
        dim=call.dim,
        is_causal=call.is_causal,
        window_size_left=call.window_size_left,
        window_size_right=call.window_size_right,
        dtype=call.dtype,
        sm_scale=call.sm_scale,
        softcap=call.softcap,
        **_rope_kwargs(call),
        device_index=_index(call),
        tune=call.tune,
    )
    return tuple(args.values()), lambda: cls(**args)


def dense_ws_entry(cls: type, call: AttentionCall) -> Entry:
    """Accepts its sequence extents at runtime unless RoPE compiles them in."""
    args = dict(
        batch=call.batch,
        heads=call.heads,
        heads_kv=call.heads_kv,
        seq_len_q=call.max_seqlen_q,
        seq_len_kv=call.seqlen_kv,
        dim=call.dim,
        is_causal=call.is_causal,
        dtype=call.dtype,
        sm_scale=call.sm_scale,
        softcap=call.softcap,
        **_rope_kwargs(call),
        device_index=_index(call),
        tune=call.tune,
    )
    if call.fuse_rope:
        return tuple(args.values()), lambda: cls(**args)
    # Without RoPE this program takes its sequence extents at runtime.
    dynamic = ("seq_len_q", "seq_len_kv")
    return tuple(v for k, v in args.items() if k not in dynamic), lambda: cls(**args)
