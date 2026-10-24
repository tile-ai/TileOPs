"""TileLang wrappers over ``csrc/streaming_load.h``; a caller's program pre-includes the header."""

import tilelang.language as T

__all__ = ["load16_evict_last"]


@T.macro
def load16_evict_last(dst, slot, src, at):
    """Read the 16-byte vector of *src* at *at* into local *dst* from *slot*, evict-last in L2."""
    T.call_extern(
        "handle", "tileops::load16_evict_last", T.address_of(dst[slot]), T.address_of(src[at])
    )
